from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from aspect_infer import collect_labels_for_row, summarize_aspects
from config import BuilderConfig
from domain_infer import infer_domain
from episodic_builder import build_episodic_rows
from schema_detect import detect_schema, load_file_rows
from splitter import assign_splits, leakage_ids, split_rows
from utils import normalize_text, stable_hash, write_json, write_jsonl
from validators import aspect_frequency, few_shot_warnings, validate_jsonl, validate_review_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(raw: Path) -> Path:
    if raw.is_absolute():
        return raw
    if len(raw.parts) >= 2 and raw.parts[0] == "dataset_builder":
        return PROJECT_ROOT / Path(*raw.parts[1:])
    return PROJECT_ROOT / raw


def list_input_files(input_dir: Path) -> List[Path]:
    supported = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls"}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported]
    return sorted(files)


def extract_aspect_values(row: Dict[str, Any], aspect_cols: List[str]) -> List[str]:
    values: List[str] = []
    for col in aspect_cols:
        raw = row.get(col, "")
        if isinstance(raw, list):
            values.extend(normalize_text(x) for x in raw)
        elif isinstance(raw, str):
            if "," in raw:
                values.extend(normalize_text(x) for x in raw.split(","))
            elif ";" in raw:
                values.extend(normalize_text(x) for x in raw.split(";"))
            else:
                values.append(normalize_text(raw))
        elif raw:
            values.append(normalize_text(raw))
    return [v for v in values if v]


def ensure_output_dirs(output_dir: Path) -> None:
    (output_dir / "reviewlevel").mkdir(parents=True, exist_ok=True)
    (output_dir / "episodic").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)


def compact_open_aspects(review_rows: List[Dict[str, Any]], min_count: int = 5) -> List[Dict[str, Any]]:
    freq = Counter()
    for row in review_rows:
        for lab in row.get("labels", []):
            freq[str(lab.get("aspect", ""))] += 1

    for row in review_rows:
        domain = str(row.get("domain", "generic")).strip().lower() or "generic"
        merged = {}
        for lab in row.get("labels", []):
            aspect = str(lab.get("aspect", "")).strip()
            mode = str(lab.get("metadata", {}).get("mapping_mode", "")).strip()
            if mode == "open_aspect" and freq.get(aspect, 0) < min_count:
                lab = dict(lab)
                lab["aspect"] = f"other_{domain}"
                lab["metadata"] = dict(lab.get("metadata", {}))
                lab["metadata"]["compacted_from"] = aspect
                lab["metadata"]["mapping_mode"] = "open_aspect_compacted"
                lab["confidence"] = max(float(lab.get("confidence", 0.0)), 0.5)
            key = (lab.get("aspect", ""), lab.get("sentiment", ""), lab.get("evidence_sentence", ""))
            if key not in merged or lab.get("confidence", 0) > merged[key].get("confidence", 0):
                merged[key] = lab
        row["labels"] = list(merged.values())
    return review_rows


def _text_signature(text: str) -> str:
    tokens = [t for t in normalize_text(text).lower().split() if len(t) > 2]
    return " ".join(tokens[:24])


def main() -> None:
    parser = argparse.ArgumentParser(description="ReviewOps dataset builder")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "input")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--max-aspects", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--prefer-open-aspect", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = [float(x.strip()) for x in args.split_ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("--split-ratios must have 3 values that sum to 1.0")

    cfg = BuilderConfig(
        input_dir=_resolve_project_path(args.input_dir),
        output_dir=_resolve_project_path(args.output_dir),
        split_ratios={"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        random_seed=args.seed,
        max_aspects_per_review=args.max_aspects,
        confidence_threshold=args.confidence_threshold,
        prefer_canonical=not args.prefer_open_aspect,
        dry_run=args.dry_run,
    )

    files = list_input_files(cfg.input_dir)
    if not files:
        print(f"No supported input files found in: {cfg.input_dir}")
        return

    review_by_id: Dict[str, Dict[str, Any]] = {}
    seen_text_signatures: List[str] = []
    skipped: List[Dict[str, Any]] = []
    schema_reports: List[Dict[str, Any]] = []
    rows_read = 0

    for file_path in files:
        try:
            rows, file_type, columns = load_file_rows(file_path)
        except Exception as exc:
            skipped.append({"file": str(file_path), "row": None, "reason": f"file_load_error: {exc}"})
            continue

        rows_read += len(rows)
        schema = detect_schema(file_path, columns, file_type)
        if not schema.text_col:
            skipped.append({"file": str(file_path), "row": None, "reason": "missing_text_column"})
            continue
        sample_texts = [normalize_text(r.get(schema.text_col, "")) for r in rows[:150] if schema.text_col]
        sample_aspects: List[str] = []
        if schema.aspect_cols:
            for r in rows[:200]:
                sample_aspects.extend(extract_aspect_values(r, schema.aspect_cols))
        domain_guess = infer_domain(schema.file_path, schema.columns, sample_texts, sample_aspects)

        schema_reports.append(
            {
                "file": str(file_path),
                "file_type": file_type,
                "rows": len(rows),
                "schema": schema.__dict__,
                "inferred_domain": domain_guess,
            }
        )

        for idx, row in enumerate(rows):
            review_text = normalize_text(row.get(schema.text_col, "")) if schema.text_col else ""
            if not review_text:
                skipped.append({"file": str(file_path), "row": idx, "reason": "empty_review_text"})
                continue
            sig = _text_signature(review_text)
            if any(sig == prev or len(set(sig.split()) & set(prev.split())) / max(1, len(set(sig.split()) | set(prev.split()))) >= 0.88 for prev in seen_text_signatures):
                skipped.append({"file": str(file_path), "row": idx, "reason": "near_duplicate_review_text"})
                continue
            seen_text_signatures.append(sig)

            source = file_path.name
            raw_id = normalize_text(row.get(schema.id_col, "")) if schema.id_col else ""
            if raw_id:
                row_id = f"{source}__{raw_id}"
            else:
                row_id = f"{source}__gen_{stable_hash(source, str(idx), review_text)}"

            split = normalize_text(row.get(schema.split_col, "")).lower() if schema.split_col else ""
            split = split if split in {"train", "val", "test"} else ""

            domain = normalize_text(row.get(schema.domain_col, "")).lower() if schema.domain_col else ""
            if not domain:
                domain = domain_guess

            aspect_values = extract_aspect_values(row, schema.aspect_cols)
            sentiment_value = row.get(schema.sentiment_col, "") if schema.sentiment_col else ""
            evidence_value = row.get(schema.evidence_col, "") if schema.evidence_col else ""
            span_from_value = row.get(schema.span_from_col, "") if schema.span_from_col else ""
            span_to_value = row.get(schema.span_to_col, "") if schema.span_to_col else ""

            labels = collect_labels_for_row(
                row=row,
                review_text=review_text,
                domain=domain,
                aspect_values=aspect_values,
                sentiment_value=sentiment_value,
                evidence_value=evidence_value,
                span_from_value=span_from_value,
                span_to_value=span_to_value,
                prefer_canonical=cfg.prefer_canonical,
                confidence_threshold=cfg.confidence_threshold,
            )

            if not labels:
                skipped.append({"file": str(file_path), "row": idx, "reason": "no_labels_after_inference"})
                continue

            if any(lab.get("type") == "implicit" and str(lab.get("aspect", "")).replace("_", " ") in normalize_text(review_text).lower() for lab in labels):
                labels = [lab for lab in labels if not (lab.get("type") == "implicit" and str(lab.get("aspect", "")).replace("_", " ") in normalize_text(review_text).lower())]
            if not labels:
                skipped.append({"file": str(file_path), "row": idx, "reason": "implicit_explicit_collision"})
                continue

            if len(labels) > cfg.max_aspects_per_review:
                labels = sorted(labels, key=lambda x: x.get("confidence", 0.0), reverse=True)[: cfg.max_aspects_per_review]

            entry = review_by_id.get(row_id)
            if entry is None:
                review_by_id[row_id] = {
                    "id": row_id,
                    "review_text": review_text,
                    "domain": domain,
                    "source": source,
                    "split": split,
                    "labels": labels,
                }
            else:
                if normalize_text(entry.get("review_text", "")) != review_text:
                    row_id = f"{row_id}__{stable_hash(review_text)}"
                    entry = review_by_id.get(row_id)
                if entry is None:
                    review_by_id[row_id] = {
                        "id": row_id,
                        "review_text": review_text,
                        "domain": domain,
                        "source": source,
                        "split": split,
                        "labels": labels,
                    }
                    continue
                combined = entry["labels"] + labels
                dedup = {}
                for lab in combined:
                    key = (lab["aspect"], lab.get("sentiment", ""), lab.get("evidence_sentence", ""))
                    if key not in dedup or lab.get("confidence", 0) > dedup[key].get("confidence", 0):
                        dedup[key] = lab
                entry["labels"] = list(dedup.values())[: cfg.max_aspects_per_review]
                if not entry.get("split"):
                    entry["split"] = split

    review_rows = list(review_by_id.values())
    review_rows = compact_open_aspects(review_rows, min_count=5)
    review_rows = assign_splits(review_rows, cfg.split_ratios, seed=cfg.random_seed)
    split_review = split_rows(review_rows)

    all_episode_rows = build_episodic_rows(review_rows)
    split_episode = {k: [r for r in all_episode_rows if r.get("split") == k] for k in ["train", "val", "test"]}

    summary = {
        "datasets_processed": len(schema_reports),
        "total_rows_read": rows_read,
        "schema_reports": schema_reports,
        "reviewlevel_rows": len(review_rows),
        "episodic_rows": len(all_episode_rows),
        "explicit_labels": sum(1 for r in review_rows for l in r.get("labels", []) if l.get("type") == "explicit"),
        "implicit_labels": sum(1 for r in review_rows for l in r.get("labels", []) if l.get("type") == "implicit"),
        "unique_aspects": sorted(list({l.get("aspect") for r in review_rows for l in r.get("labels", []) if l.get("aspect")})),
        "top_aspects": summarize_aspects(review_rows),
        "split_sizes": {
            "reviewlevel": {k: len(v) for k, v in split_review.items()},
            "episodic": {k: len(v) for k, v in split_episode.items()},
        },
        "duplicate_id_count": validate_review_rows(review_rows).get("duplicate_id", 0),
        "rows_skipped": len(skipped),
        "rows_skipped_reasons": dict(Counter(s["reason"] for s in skipped)),
        "validation": {
            "reviewlevel": validate_review_rows(review_rows),
            "leakage_ids": leakage_ids(split_review),
            "few_shot_warnings": few_shot_warnings(all_episode_rows),
            "aspect_frequency": aspect_frequency(review_rows),
        },
    }

    print("=== Dataset Builder Summary ===")
    print(f"Datasets processed: {summary['datasets_processed']}")
    print(f"Rows read: {summary['total_rows_read']}")
    print(f"Review-level rows: {summary['reviewlevel_rows']}")
    print(f"Episodic rows: {summary['episodic_rows']}")
    print(f"Split sizes: {summary['split_sizes']}")
    print(f"Skipped rows: {summary['rows_skipped']} -> {summary['rows_skipped_reasons']}")

    preview_reviews = review_rows[: cfg.sample_preview_count]
    preview_episodic = all_episode_rows[: cfg.sample_preview_count]
    print("\nReview-level preview:")
    for row in preview_reviews:
        print({"id": row["id"], "domain": row["domain"], "split": row["split"], "labels": row["labels"][:2]})

    print("\nEpisodic preview:")
    for row in preview_episodic:
        print({k: row[k] for k in ["example_id", "parent_review_id", "aspect", "implicit_aspect", "split"]})

    if cfg.dry_run:
        print("\nDry run mode: no files written.")
        return

    ensure_output_dirs(cfg.output_dir)
    out_review = cfg.output_dir / "reviewlevel"
    out_epi = cfg.output_dir / "episodic"
    out_reports = cfg.output_dir / "reports"

    for split in ["train", "val", "test"]:
        write_jsonl(out_review / f"{split}.jsonl", split_review[split])
        write_jsonl(out_epi / f"{split}.jsonl", split_episode[split])

    write_json(out_reports / "build_report.json", summary)
    write_jsonl(out_reports / "skipped_rows.jsonl", skipped)

    jsonl_errors = []
    for split in ["train", "val", "test"]:
        jsonl_errors.extend(validate_jsonl(out_review / f"{split}.jsonl"))
        jsonl_errors.extend(validate_jsonl(out_epi / f"{split}.jsonl"))
    if jsonl_errors:
        print("\nValidation JSONL errors found:")
        for err in jsonl_errors[:15]:
            print(" -", err)
    else:
        print("\nJSONL validation passed.")


if __name__ == "__main__":
    main()
