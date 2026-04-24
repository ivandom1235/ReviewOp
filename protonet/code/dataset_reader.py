from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .config import ProtonetConfig, split_file
    from .progress import track
except ImportError:
    import importlib.util
    import sys

    _config_path = Path(__file__).resolve().with_name("config.py")
    _config_spec = importlib.util.spec_from_file_location("protonet_local_config", _config_path)
    if _config_spec is None or _config_spec.loader is None:  # pragma: no cover
        raise
    _config_module = importlib.util.module_from_spec(_config_spec)
    sys.modules[_config_spec.name] = _config_module
    _config_spec.loader.exec_module(_config_module)
    ProtonetConfig = _config_module.ProtonetConfig
    split_file = _config_module.split_file
    from progress import track


VALID_SPLITS = ("train", "val", "test")
REQUIRED_BENCHMARK_FILES = tuple(f"{split}.jsonl" for split in VALID_SPLITS) + ("manifest.json",)


@dataclass
class DatasetSummary:
    split_sizes: Dict[str, int]
    detected_format: str
    input_type: str


def _normalize_sentiment(value: Any, fallback: str = "neutral") -> str:
    candidate = value
    if isinstance(candidate, list):
        candidate = candidate[0] if candidate else fallback
    sentiment = str(candidate or fallback).strip().lower()
    return sentiment or fallback


def _normalize_evidence_span(evidence_span: Any, review_text: str, evidence_text: str) -> tuple[list[int], bool]:
    if isinstance(evidence_span, list) and len(evidence_span) == 2:
        try:
            start = int(evidence_span[0] if evidence_span[0] is not None else -1)
            end = int(evidence_span[1] if evidence_span[1] is not None else -1)
            if 0 <= start <= end <= len(review_text):
                return [start, end], False
        except (TypeError, ValueError):
            pass
    if evidence_text and review_text:
        start = review_text.lower().find(evidence_text.lower())
        if start >= 0:
            return [int(start), int(start + len(evidence_text))], True
    return [-1, -1], True


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object rows in {path}, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_split_rows(path: Path, *, progress_enabled: bool) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input split file: {path}")
    rows = load_jsonl(path)
    return list(track(rows, total=len(rows), desc=f"load:{path.stem}", enabled=progress_enabled))


def validate_benchmark_artifacts(input_dir: Path) -> None:
    missing = [input_dir / name for name in REQUIRED_BENCHMARK_FILES if not (input_dir / name).exists()]
    if not missing:
        return

    if input_dir.name == "ambiguity_grounded" and input_dir.parent.name == "benchmark":
        output_dir_hint = input_dir.parent.parent
    else:
        output_dir_hint = Path("dataset_builder") / "output"
    missing_lines = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(
        "Missing benchmark artifacts required by protonet.\n"
        f"Input directory: {input_dir}\n"
        f"Missing files:\n{missing_lines}\n"
        "Generate them with:\n"
        f"python dataset_builder\\scripts\\build_benchmark.py --input dataset_builder\\input --output-dir {output_dir_hint}"
    )


def _label_from_interpretation(
    instance_id: str,
    review_text: str,
    domain: str,
    interp: Dict[str, Any],
    split: str,
    idx: int,
    gold_joint_labels: List[str],
    split_protocol: Dict[str, Any],
    ambiguity_score: float,
    abstain_acceptable: bool,
    ambiguity_type: str | None,
    novel_acceptable: bool,
    novelty_status: str,
    novel_cluster_id: str | None,
    novel_alias: str | None,
    novel_evidence_text: str | None,
) -> Dict[str, Any]:
    sentiment = _normalize_sentiment(interp.get("sentiment"))
    evidence_text = str(interp.get("evidence_text") or "").strip()
    if not evidence_text:
        evidence_text = review_text
    evidence_span, evidence_fallback_used = _normalize_evidence_span(interp.get("evidence_span"), review_text, evidence_text)
    aspect_raw = str(interp.get("aspect_raw") or interp.get("aspect_label") or interp.get("aspect") or "unknown").strip()
    aspect_canonical = str(
        interp.get("aspect_canonical")
        or interp.get("domain_canonical_aspect")
        or interp.get("aspect_label")
        or interp.get("aspect")
        or aspect_raw
    ).strip()
    latent_family = str(interp.get("latent_family") or "unknown").strip()
    label_type = str(interp.get("label_type") or interp.get("interpretation_type") or "implicit").strip().lower()
    if label_type not in {"explicit", "implicit", "verified"}:
        label_type = "implicit"
    return {
        "example_id": f"{instance_id}_g{idx}",
        "parent_review_id": instance_id,
        "review_text": review_text,
        "evidence_sentence": evidence_text,
        "evidence_text": evidence_text,
        "evidence_span": evidence_span,
        "evidence_fallback_used": evidence_fallback_used or evidence_span == [-1, -1],
        "domain": domain,
        "domain_family": str(interp.get("domain_family") or ""),
        "group_id": str(interp.get("group_id") or ""),
        "hardness_tier": str(interp.get("hardness_tier") or "H0"),
        "annotation_source": str(interp.get("annotation_source") or "unknown"),
        "aspect": aspect_canonical,
        "implicit_aspect": aspect_canonical,
        "aspect_raw": aspect_raw,
        "latent_family": latent_family,
        "aspect_canonical": aspect_canonical,
        "sentiment": sentiment,
        "label_type": label_type,
        "confidence": float(interp.get("canonical_confidence", interp.get("annotator_support", 1))),
        "support_type": str(interp.get("support_type") or "unknown"),
        "mapping_source": str(interp.get("mapping_source") or "unknown"),
        "quality_flags": list(interp.get("quality_flags") or []),
        "split": split,
        "abstain_acceptable": bool(abstain_acceptable),
        "ambiguity_type": str(ambiguity_type or "").strip() or None,
        "novel_acceptable": bool(novel_acceptable),
        "novelty_status": novelty_status,
        "source_type": str(interp.get("source_type") or "unknown"),
        "novel_cluster_id": str(novel_cluster_id or "").strip() or None,
        "novel_alias": str(novel_alias or "").strip() or None,
        "novel_evidence_text": str(novel_evidence_text or "").strip() or None,
        "gold_joint_labels": gold_joint_labels,
        "split_protocol": split_protocol,
        "benchmark_ambiguity_score": float(ambiguity_score),
    }


def validate_benchmark_rows(rows: List[Dict[str, Any]], split: str) -> tuple[List[Dict[str, Any]], str]:
    if not rows:
        raise ValueError(f"No benchmark rows found for split {split}")
    expanded: List[Dict[str, Any]] = []
    evidence_fallback_counter = 0
    novel_counter = 0
    abstain_counter = 0
    for index, row in enumerate(rows):
        instance_id = str(row.get("instance_id") or row.get("review_id") or "").strip()
        review_text = str(row.get("review_text") or "").strip()
        domain = str(row.get("domain") or "unknown")
        if not instance_id or not review_text:
            raise ValueError(f"benchmark row {index} in split {split} is missing instance_id/review_text")
        interpretations = row.get("gold_interpretations")
        if not isinstance(interpretations, list) or not interpretations:
            raise ValueError(f"benchmark row {index} in split {split} has no gold_interpretations")
        split_protocol = row.get("split_protocol") if isinstance(row.get("split_protocol"), dict) else {}
        if "grouped" not in split_protocol and "source_holdout" in split_protocol:
            split_protocol["grouped"] = split_protocol.get("source_holdout")
        ambiguity_score = float(row.get("ambiguity_score", 0.0))
        abstain_acceptable = bool(row.get("abstain_acceptable", False))
        novelty_status = str(row.get("novelty_status") or "known").strip().lower()
        novel_acceptable = bool(row.get("novel_acceptable", False)) or novelty_status == "novel"
        novel_cluster_id = str(row.get("novel_cluster_id") or "").strip() or None
        novel_alias = str(row.get("novel_alias") or "").strip() or None
        novel_evidence_text = str(row.get("novel_evidence_text") or "").strip() or None
        group_id = str(row.get("group_id") or "").strip()
        domain_family = str(row.get("domain_family") or "").strip()
        hardness_tier = str(row.get("hardness_tier") or "H0").strip().upper()
        annotation_source = str(row.get("annotation_source") or "unknown").strip()
        if novel_acceptable:
            novel_counter += 1
        if abstain_acceptable:
            abstain_counter += 1
        gold_joint_labels = []
        for interp in interpretations:
            if not isinstance(interp, dict):
                continue
            aspect = str(
                interp.get("aspect_canonical")
                or interp.get("domain_canonical_aspect")
                or interp.get("aspect_label")
                or interp.get("aspect")
                or interp.get("aspect_raw")
                or "unknown"
            ).strip()
            sentiment = _normalize_sentiment(interp.get("sentiment"))
            gold_joint_labels.append(f"{aspect}__{sentiment}")
        for interp_idx, interp in enumerate(interpretations, start=1):
            if not isinstance(interp, dict):
                continue
            payload = {
                **interp,
                "group_id": group_id,
                "domain_family": domain_family,
                "hardness_tier": hardness_tier,
                "annotation_source": annotation_source,
            }
            built = _label_from_interpretation(
                instance_id,
                review_text,
                domain,
                payload,
                split,
                interp_idx,
                gold_joint_labels,
                split_protocol,
                ambiguity_score,
                abstain_acceptable,
                str(interp.get("ambiguity_type") or "").strip() or None,
                novel_acceptable,
                novelty_status,
                novel_cluster_id,
                novel_alias,
                novel_evidence_text,
            )
            if bool(built.get("evidence_fallback_used", False)):
                evidence_fallback_counter += 1
            expanded.append(built)
    if split in {"val", "test"} and novel_counter == 0:
        print(f"[warn] {split} split contains zero novel positives; novelty calibration/eval will be limited.")
    if split in {"val", "test"} and abstain_counter == 0:
        print(f"[warn] {split} split contains zero abstain-acceptable rows; abstention eval will be limited.")
    if evidence_fallback_counter > 0:
        print(f"[warn] {split} split required evidence span fallback for {evidence_fallback_counter} examples.")
    return expanded, "benchmark_examples"


def load_input_dataset(cfg: ProtonetConfig) -> tuple[Dict[str, List[Dict[str, Any]]], DatasetSummary]:
    if cfg.input_type != "benchmark":
        raise ValueError("V6 runtime only supports input_type='benchmark'")
    validate_benchmark_artifacts(cfg.input_dir)
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    detected_format = "unknown"
    for split in VALID_SPLITS:
        path = split_file(cfg.input_dir, split)
        raw_rows = read_split_rows(path, progress_enabled=cfg.progress_enabled)
        rows, detected_format = validate_benchmark_rows(raw_rows, split)
        rows_by_split[split] = rows
    summary = DatasetSummary(
        split_sizes={split: len(rows) for split, rows in rows_by_split.items()},
        detected_format=detected_format,
        input_type=cfg.input_type,
    )
    return rows_by_split, summary
