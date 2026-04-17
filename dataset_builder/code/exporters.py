from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from utils import read_jsonl, write_json, write_jsonl


def _benchmark_artifact_counts(base_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = base_dir / f"{split}.jsonl"
        counts[split] = len(read_jsonl(path)) if path.exists() else 0
    counts["total"] = sum(counts[split] for split in ("train", "val", "test"))
    for protocol in ("random", "grouped", "domain_holdout"):
        protocol_dir = base_dir.parent / protocol
        if not protocol_dir.exists():
            continue
        for split in ("train", "val", "test"):
            path = protocol_dir / f"{split}.jsonl"
            key = f"{protocol}_{split}"
            counts[key] = len(read_jsonl(path)) if path.exists() else 0
    return counts


def _schema_fingerprint(schema: Any) -> Any:
    if isinstance(schema, dict):
        return schema.get("schema_fingerprint")
    return getattr(schema, "schema_fingerprint", None)


def write_split_outputs(base_dir: Path, payload: dict[str, list[dict[str, Any]]]) -> None:
    for split, rows in payload.items():
        write_jsonl(base_dir / f"{split}.jsonl", rows)


def write_named_outputs(base_dir: Path, payload: dict[str, list[dict[str, Any]]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in payload.items():
        write_jsonl(base_dir / f"{name}.jsonl", rows)


def write_benchmark_outputs(
    target_dir: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    protocol_views: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_jsonl(target_dir / f"{split}.jsonl", rows_by_split.get(split, []))
    if protocol_views:
        for protocol_name, payload in protocol_views.items():
            protocol_dir = target_dir.parent / protocol_name
            protocol_dir.mkdir(parents=True, exist_ok=True)
            for split in ("train", "val", "test"):
                write_jsonl(protocol_dir / f"{split}.jsonl", payload.get(split, []))
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_pipeline_outputs(
    *,
    cfg: Any,
    report: dict[str, Any],
    benchmark_rows_by_split: dict[str, list[dict[str, Any]]],
    benchmark_metadata: dict[str, Any],
    benchmark_protocol_views: dict[str, dict[str, list[dict[str, Any]]]] | None,
    benchmark_review_queue_rows: list[dict[str, Any]],
    run_registry: dict[str, Any],
    promoted_registry: dict[str, Any],
    quality_analysis_artifact: dict[str, Any],
    synthetic_accepted: list[dict[str, Any]],
    synthetic_rejected: list[dict[str, Any]],
    synthetic_audit: dict[str, Any],
    benchmark_v2_novelty: dict[str, Any],
    research_manifest: dict[str, Any],
    previous_accepted_path: Path,
    robust_training_eval: dict[str, Any],
    promotion_guard: dict[str, Any],
) -> dict[str, Any]:
    write_benchmark_outputs(
        cfg.benchmark_dir,
        benchmark_rows_by_split,
        benchmark_metadata,
        protocol_views=benchmark_protocol_views,
    )
    write_jsonl(cfg.benchmark_dir / "review_queue.jsonl", benchmark_review_queue_rows)
    write_json(cfg.reports_dir / "aspect_registry_run.json", run_registry)
    write_json(cfg.reports_dir / "aspect_registry_promoted.json", promoted_registry)
    write_json(cfg.reports_dir / "quality_analysis.json", quality_analysis_artifact)
    write_jsonl(
        cfg.reports_dir / "silver_pool.jsonl",
        [dict(item.get("row") or {}, decision=item.get("decision"), reason_codes=item.get("reason_codes", []), quality_score=item.get("quality_score"), usefulness_score=item.get("usefulness_score")) for item in quality_analysis_artifact.get("silver_rows", [])],
    )
    write_jsonl(
        cfg.reports_dir / "quality_review_queue.jsonl",
        [
            {
                **dict(item.get("row") or {}),
                "decision": item.get("decision"),
                "bucket": item.get("bucket"),
                "reason_codes": item.get("reason_codes", []),
                "recovery_eligible": item.get("recovery_eligible", False),
                "quality_score": item.get("quality_score"),
                "usefulness_score": item.get("usefulness_score"),
            }
            for item in quality_analysis_artifact.get("review_queue_rows", [])
        ],
    )
    from synthetic_generation import write_synthetic_outputs

    write_synthetic_outputs(
        output_dir=cfg.output_dir / "synthetic",
        accepted=synthetic_accepted,
        rejected=synthetic_rejected,
    )
    write_json(cfg.output_dir / "synthetic" / "audit.json", synthetic_audit)
    benchmark_artifact_counts = _benchmark_artifact_counts(cfg.benchmark_dir)
    benchmark_report_counts = dict(benchmark_metadata.get("split_counts", {}))
    benchmark_report_counts["total"] = int(benchmark_metadata.get("rows", 0))
    benchmark_artifact_counts_match = all(
        int(benchmark_artifact_counts.get(key, 0)) == int(value)
        for key, value in benchmark_report_counts.items()
    )
    report["benchmark_artifact_counts"] = benchmark_artifact_counts
    report["validation"]["benchmark_artifact_counts_match"] = benchmark_artifact_counts_match
    if not benchmark_artifact_counts_match:
        report["blocking_reasons"] = report.get("blocking_reasons", [])
        report["blocking_reasons"].append({
            "code": "BENCHMARK_ARTIFACT_COUNT_MISMATCH",
            "message": "Written benchmark JSONL counts do not match report metadata.",
        })
    if not bool(promotion_guard.get("blocked", False)):
        previous_accepted_path.parent.mkdir(parents=True, exist_ok=True)
        previous_accepted_path.write_text(
            json.dumps(
                {
                    "updated_at": report.get("generated_at"),
                    "worst_domain_f1": float((robust_training_eval.get("groupdro") or {}).get("worst_domain_f1", 0.0)),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    write_json(cfg.reports_dir / "build_report.json", report)
    write_json(cfg.reports_dir / "benchmark_v2_novelty_report.json", benchmark_v2_novelty)
    write_json(cfg.reports_dir / "data_quality_report.json", {
        "run_profile": report.get("run_profile"),
        "artifact_mode": report.get("artifact_mode"),
        "rows_in": report.get("row_counts", {}).get("input"),
        "rows_out": report.get("row_counts", {}).get("selected"),
        "text_column": report.get("text_column"),
        "schema_fingerprint": _schema_fingerprint(report.get("schema")),
        "candidate_aspects": report.get("candidate_aspects", []),
        "candidate_aspects_by_language": report.get("candidate_aspects_by_language", {}),
        "candidate_aspects_by_domain": report.get("candidate_aspects_by_domain", {}),
        "implicit_mode": report.get("implicit_mode"),
        "multilingual_mode": report.get("multilingual_mode"),
        "coreference_enabled": report.get("coreference_enabled"),
        "language_distribution": report.get("language_distribution", {}),
        "row_counts": report.get("row_counts", {}),
        "research": report.get("research", {}),
        "output_quality": report.get("output_quality", {}),
        "benchmark_summary": benchmark_metadata,
        "benchmark_v2_novelty": benchmark_v2_novelty,
        "aspect_registry_run": run_registry,
        "aspect_registry_promoted": promoted_registry,
        "sentiment_quality": report.get("sentiment_quality", {}),
        "synthetic_generation": synthetic_audit,
        "robust_training_eval": robust_training_eval,
        "promotion_guard": promotion_guard,
        "governance_signoff": report.get("governance_signoff", {}),
        "benchmark_artifact_counts": benchmark_artifact_counts,
        "benchmark_report_counts": benchmark_report_counts,
        "benchmark_artifact_counts_match": benchmark_artifact_counts_match,
        "strict_quality": report.get("strict_quality", {}),
        "strict_artifacts": report.get("strict_artifacts", {}),
        "train_salvage_stats": report.get("train_salvage_stats", {}),
        "train_quarantine_recoverable_rows": report.get("train_quarantine_recoverable_rows", 0),
        "train_quarantine_recovery_stats": report.get("train_quarantine_recovery_stats", {}),
        "train_topup_stats": report.get("train_topup_stats", {}),
        "train_reinference_stats": report.get("train_reinference_stats", {}),
        "train_review_dropped_soft_rows": report.get("train_review_dropped_soft_rows", 0),
        "train_review_dropped_hard_rows": report.get("train_review_dropped_hard_rows", 0),
        "silver_pool_rows": len(quality_analysis_artifact.get("silver_rows", [])),
        "hard_reject_rows": len(quality_analysis_artifact.get("hard_reject_rows", [])),
        "train_keep_rows": len(quality_analysis_artifact.get("train_keep_rows", [])),
        "decision_counts": quality_analysis_artifact.get("decision_counts", {}),
        "size_recovery_stage": report.get("size_recovery_stage", "none"),
        "size_recovery_shortfall_remaining": report.get("size_recovery_shortfall_remaining", 0),
        "topup_effectiveness": report.get("topup_effectiveness", {}),
        "train_topup_rejection_breakdown": report.get("train_topup_rejection_breakdown", {}),
        "train_target_stats": report.get("train_target_stats", {}),
        "train_sentiment_constraints": report.get("train_sentiment_constraints", {}),
        "train_domain_leakage_rows": report.get("train_domain_leakage_rows"),
        "train_domain_leakage_row_rate": report.get("train_domain_leakage_row_rate"),
        "train_domain_leakage_aspect_instances": report.get("train_domain_leakage_aspect_instances"),
        "eval_domain_leakage_rows": report.get("eval_domain_leakage_rows"),
        "eval_domain_leakage_row_rate": report.get("eval_domain_leakage_row_rate"),
        "eval_domain_leakage_aspect_instances": report.get("eval_domain_leakage_aspect_instances"),
        "train_negative_ratio": report.get("train_negative_ratio"),
        "train_positive_ratio": report.get("train_positive_ratio"),
        "grounded_prediction_rate": report.get("grounded_prediction_rate"),
        "ungrounded_non_general_count": report.get("ungrounded_non_general_count"),
        "gold_eval": report.get("gold_eval", {}),
        "domain_generalization": report.get("domain_generalization", {}),
        "unseen_domain_metrics": report.get("unseen_domain_metrics", {}),
        "domain_prior_boost_count": report.get("domain_prior_boost_count"),
        "domain_prior_penalty_count": report.get("domain_prior_penalty_count"),
        "novelty_identity": report.get("novelty_identity", {}),
        "promotion_eligibility": report.get("promotion_eligibility", {}),
        "blocking_reasons": report.get("blocking_reasons", []),
        "validation": report.get("validation", {}),
        "chunked_execution": report.get("chunked_execution", {}),
    })
    write_json(cfg.reports_dir / "research_manifest.json", research_manifest)
    return {
        "benchmark_artifact_counts": benchmark_artifact_counts,
        "benchmark_artifact_counts_match": benchmark_artifact_counts_match,
    }
