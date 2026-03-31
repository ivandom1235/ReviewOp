from __future__ import annotations

import argparse
from dataclasses import asdict
from collections import Counter
import random
from pathlib import Path
from typing import Any

import pandas as pd

from contracts import BuilderConfig
from evaluation import aspect_metrics
from exporters import write_compat_exports, write_split_outputs
from explicit_features import build_explicit_row, fit_explicit_artifacts
from implicit_pipeline import _is_valid_latent_aspect, build_implicit_row, collect_diagnostics, discover_aspects
from io_utils import load_inputs
from research_stack import build_research_manifest, resolve_benchmark, resolve_model_family
from schema_detect import detect_schema
from splitter import choose_stratify_values, preliminary_split, split_holdout
from utils import stable_id, utc_now_iso, write_json


def _assign_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ids = []
    for idx, row in out.iterrows():
        ids.append(stable_id(row.get("source_file", "source"), idx, row.to_json()))
    out["id"] = ids
    return out


def _canonical_domain(source_file: str | None) -> str:
    name = Path(str(source_file or "unknown")).stem.lower()
    for suffix in ("_train", "_test", "_val", "-train", "-test", "-val"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _chunk_rows(rows: list[dict[str, Any]], cfg: BuilderConfig) -> list[dict[str, Any]]:
    ordered = list(rows)
    rng = random.Random(cfg.random_seed)
    rng.shuffle(ordered)
    return ordered


def _select_working_rows(rows: list[dict[str, Any]], cfg: BuilderConfig) -> list[dict[str, Any]]:
    ordered = _chunk_rows(rows, cfg)
    if cfg.sample_size is not None:
        ordered = ordered[: max(0, cfg.sample_size)]
    if cfg.chunk_size is not None:
        start = max(0, cfg.chunk_offset)
        end = start + max(0, cfg.chunk_size)
        ordered = ordered[start:end]
    return ordered


_NON_FEATURE_COLUMNS = {
    "id",
    "aspect",
    "aspect_term",
    "from",
    "to",
    "label",
    "labels",
    "polarity",
    "sentiment",
    "target",
    "target_aspect",
    "gold_aspect",
    "gold_labels",
}


def _feature_columns(columns: list[str], *, text_column: str, target_column: str | None = None) -> list[str]:
    excluded = {text_column}
    if target_column:
        excluded.add(target_column)
    excluded.update(_NON_FEATURE_COLUMNS)
    return [column for column in columns if column not in excluded]


def _split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split", "train"))
        grouped.setdefault(split, []).append(row)
    return grouped


def _aspect_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for aspect in row.get("implicit", {}).get("aspects", []):
            if aspect != "general":
                counts[str(aspect)] += 1
    return counts


def _row_count(rows: list[dict[str, Any]], predicate: Any) -> int:
    return sum(1 for row in rows if predicate(row))


def _fallback_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(_row_count(rows, lambda row: row.get("implicit", {}).get("aspects") == ["general"]) / len(rows), 4)


def _quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_by_split = _split_rows(rows)
    grouped_by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_by_domain.setdefault(str(row.get("domain", "unknown")), []).append(row)

    aspect_counts = _aspect_counts(rows)
    generic_aspect_count = sum(count for aspect, count in aspect_counts.items() if not _is_valid_latent_aspect(aspect))
    rejected_aspect_count = sum(count for aspect, count in aspect_counts.items() if aspect != "general" and not _is_valid_latent_aspect(aspect))

    return {
        "canonical_domains": sorted(grouped_by_domain),
        "fallback_only_rows": _row_count(rows, lambda row: row.get("implicit", {}).get("aspects") == ["general"]),
        "fallback_only_rate": _fallback_rate(rows),
        "needs_review_rows": _row_count(rows, lambda row: bool(row.get("implicit", {}).get("needs_review"))),
        "generic_implicit_aspects": generic_aspect_count,
        "rejected_implicit_aspects": rejected_aspect_count,
        "span_support": {
            "exact": sum(
                1
                for row in rows
                for span in row.get("implicit", {}).get("spans", [])
                if span.get("support_type") == "exact"
            ),
            "near_exact": sum(
                1
                for row in rows
                for span in row.get("implicit", {}).get("spans", [])
                if span.get("support_type") == "near_exact"
            ),
        },
        "top_implicit_aspects": aspect_counts.most_common(10),
        "top_implicit_aspects_by_split": {
            split: _aspect_counts(split_rows).most_common(10) for split, split_rows in grouped_by_split.items()
        },
        "top_implicit_aspects_by_domain": {
            domain: _aspect_counts(domain_rows).most_common(10) for domain, domain_rows in grouped_by_domain.items()
        },
        "fallback_only_rate_by_split": {
            split: _fallback_rate(split_rows) for split, split_rows in grouped_by_split.items()
        },
        "fallback_only_rate_by_domain": {
            domain: _fallback_rate(domain_rows) for domain, domain_rows in grouped_by_domain.items()
        },
    }


def run_pipeline(cfg: BuilderConfig) -> dict[str, Any]:
    cfg.ensure_dirs()
    frame = load_inputs(cfg.input_dir)
    if frame.empty:
        raise ValueError(f"No supported input files found under {cfg.input_dir}")

    schema = detect_schema(frame, text_column_override=cfg.text_column_override)
    text_column = schema.primary_text_column
    if text_column is None:
        raise ValueError("No text column detected")

    clean_frame = frame.copy().reset_index(drop=True)
    clean_frame = _assign_ids(clean_frame)
    clean_frame[text_column] = clean_frame[text_column].fillna("").astype(str)
    clean_frame = clean_frame[clean_frame[text_column].str.split().map(len) >= cfg.min_text_tokens].reset_index(drop=True)

    working_rows = _select_working_rows(clean_frame.to_dict(orient="records"), cfg)
    if not working_rows:
        raise ValueError("No rows selected after applying sampling and chunk constraints")
    sample_frame = pd.DataFrame(working_rows)
    feature_numeric_columns = _feature_columns(schema.numeric_columns, text_column=text_column, target_column=schema.target_column)
    feature_categorical_columns = _feature_columns(schema.categorical_columns, text_column=text_column, target_column=schema.target_column)
    stratify_key, stratify_values = choose_stratify_values(
        sample_frame.to_dict(orient="records"),
        preferred_key=schema.target_column,
        fallback_key=text_column,
    )
    train_frame, holdout_frame = preliminary_split(
        sample_frame,
        train_ratio=cfg.train_ratio,
        random_seed=cfg.random_seed,
        stratify_values=stratify_values,
    )

    train_rows = train_frame.to_dict(orient="records")
    holdout_rows = holdout_frame.to_dict(orient="records")
    val_rows, test_rows = split_holdout(
        holdout_rows,
        val_ratio_within_holdout=cfg.val_ratio / max(cfg.val_ratio + cfg.test_ratio, 1e-9),
        random_seed=cfg.random_seed + 1,
        stratify_values=[str(row.get(stratify_key, "unknown")) for row in holdout_rows] if stratify_key else None,
    )

    candidate_aspects = discover_aspects(train_rows, text_column=text_column, max_aspects=cfg.max_aspects, implicit_mode=cfg.implicit_mode)
    artifacts = fit_explicit_artifacts(
        train_frame,
        feature_numeric_columns,
        feature_categorical_columns,
    )

    def build_rows(rows: list[dict[str, Any]], split_name: str, offset: int = 0) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            explicit = build_explicit_row(
                {**row, "split": split_name},
                artifacts=artifacts,
                numeric_columns=feature_numeric_columns,
                categorical_columns=feature_categorical_columns,
                datetime_columns=schema.datetime_columns,
                text_column=text_column,
            )
            implicit = build_implicit_row(
                {**row, "split": split_name},
                text_column=text_column,
                candidate_aspects=candidate_aspects,
                confidence_threshold=cfg.confidence_threshold,
                row_index=idx,
                domain=_canonical_domain(str(row.get("source_file", "unknown"))),
                implicit_mode=cfg.implicit_mode,
                chunk_offset=offset,
            )
            built.append({
                "id": row.get("id"),
                "split": split_name,
                "source_file": row.get("source_file"),
                "source_text": row.get(text_column, ""),
                "domain": _canonical_domain(str(row.get("source_file", "unknown"))),
                "language": "en",
                "gold_labels": row.get("gold_labels", []),
                "explicit": explicit["explicit"],
                "implicit": implicit["implicit"],
                "diagnostics": {"schema_fingerprint": schema.schema_fingerprint, "text_column": text_column},
            })
        return built

    train_built = build_rows(train_rows, "train", 0)
    val_built = build_rows(val_rows, "val", len(train_rows))
    test_built = build_rows(test_rows, "test", len(train_rows) + len(val_rows))

    chunked_preview = _chunk_rows(train_built, cfg)
    finalized_rows = train_built + val_built + test_built
    benchmark_spec = resolve_benchmark(
        benchmark_key=cfg.benchmark_key,
        domains=[row.get("domain") for row in finalized_rows],
        languages=[row.get("language") for row in finalized_rows],
        source_files=[row.get("source_file") for row in finalized_rows],
    )
    model_spec = resolve_model_family(cfg.model_family)
    diagnostics = collect_diagnostics(finalized_rows, text_column=text_column, candidate_aspects=candidate_aspects)
    quality_summary = _quality_summary(finalized_rows)
    report = {
        "pipeline_version": "1.0-clean-room",
        "generated_at": utc_now_iso(),
        "config": asdict(cfg),
        "schema": asdict(schema),
        "implicit_mode": cfg.implicit_mode,
        "research": {
            "benchmark": benchmark_spec.key,
            "benchmark_family": benchmark_spec.family,
            "model_family": model_spec.key,
            "model_kind": model_spec.kind,
            "prompt_mode": cfg.prompt_mode,
            "augmentation_mode": cfg.augmentation_mode,
        },
        "stratification_choice": stratify_key,
        "candidate_aspects": candidate_aspects,
        "split_sizes": {"train": len(train_built), "val": len(val_built), "test": len(test_built)},
        "chunk_preview_size": len(chunked_preview),
        "chunk_sampling_strategy": "seeded_shuffle_then_slice",
        "implicit_diagnostics": diagnostics,
        "output_quality": quality_summary,
        "explicit_metrics": aspect_metrics(train_built),
    }
    research_manifest = build_research_manifest(
        dataset={
            "input_dir": cfg.input_dir,
            "output_dir": cfg.output_dir,
            "rows_in": len(frame),
            "rows_out": len(clean_frame),
            "text_column": text_column,
            "schema_fingerprint": schema.schema_fingerprint,
        },
        benchmark=benchmark_spec,
        model_family=model_spec,
        metrics=quality_summary,
        prompt_mode=cfg.prompt_mode,
        augmentation_mode=cfg.augmentation_mode,
    )

    if not cfg.dry_run:
        write_split_outputs(cfg.explicit_dir, {
            "train": [
                {
                    **row["explicit"],
                    "id": row["id"],
                    "split": row["split"],
                    "source_file": row["source_file"],
                    "source_text": row["source_text"],
                    "domain": row["domain"],
                    "language": row["language"],
                }
                for row in train_built
            ],
            "val": [
                {
                    **row["explicit"],
                    "id": row["id"],
                    "split": row["split"],
                    "source_file": row["source_file"],
                    "source_text": row["source_text"],
                    "domain": row["domain"],
                    "language": row["language"],
                }
                for row in val_built
            ],
            "test": [
                {
                    **row["explicit"],
                    "id": row["id"],
                    "split": row["split"],
                    "source_file": row["source_file"],
                    "source_text": row["source_text"],
                    "domain": row["domain"],
                    "language": row["language"],
                }
                for row in test_built
            ],
        })
        write_split_outputs(cfg.implicit_dir, {
            "train": train_built,
            "val": val_built,
            "test": test_built,
        })
        write_json(cfg.reports_dir / "build_report.json", report)
        write_json(cfg.reports_dir / "data_quality_report.json", {
            "rows_in": len(frame),
            "rows_out": len(clean_frame),
            "text_column": text_column,
            "schema_fingerprint": schema.schema_fingerprint,
            "candidate_aspects": candidate_aspects,
            "implicit_mode": cfg.implicit_mode,
            "research": report["research"],
            "output_quality": quality_summary,
            "chunked_execution": {
                "sample_size": cfg.sample_size,
                "chunk_size": cfg.chunk_size,
                "chunk_offset": cfg.chunk_offset,
                "strategy": "seeded_shuffle_then_slice",
            },
        })
        write_json(cfg.reports_dir / "research_manifest.json", research_manifest)
        write_compat_exports(cfg.output_dir / "compat" / "protonet" / "reviewlevel", {"train": train_built, "val": val_built, "test": test_built})
        write_compat_exports(cfg.output_dir / "compat" / "protonet" / "episodic", {"train": train_built, "val": val_built, "test": test_built})
        write_compat_exports(cfg.output_dir / "compat" / "backend", {"train": train_built, "val": val_built, "test": test_built})
    report["research_manifest"] = research_manifest
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean-room dataset builder")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--max-aspects", type=int, default=20)
    parser.add_argument("--min-text-tokens", type=int, default=4)
    parser.add_argument("--implicit-mode", type=str, default="heuristic", choices=["heuristic", "benchmark"])
    parser.add_argument("--benchmark-key", type=str, default=None)
    parser.add_argument("--model-family", type=str, default="heuristic_latent")
    parser.add_argument("--augmentation-mode", type=str, default="none")
    parser.add_argument("--prompt-mode", type=str, default="constrained")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = BuilderConfig(
        input_dir=args.input_dir or BuilderConfig().input_dir,
        output_dir=args.output_dir or BuilderConfig().output_dir,
        random_seed=args.seed,
        text_column_override=args.text_column,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_offset=args.chunk_offset,
        dry_run=args.dry_run or args.preview,
        preview_only=args.preview,
        confidence_threshold=args.confidence_threshold,
        max_aspects=args.max_aspects,
        min_text_tokens=args.min_text_tokens,
        implicit_mode=args.implicit_mode,
        benchmark_key=args.benchmark_key,
        model_family=args.model_family,
        augmentation_mode=args.augmentation_mode,
        prompt_mode=args.prompt_mode,
    )
    report = run_pipeline(cfg)
    print(f"Build complete: {report['generated_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
