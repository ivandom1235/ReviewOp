from __future__ import annotations

from collections import Counter
from typing import Any


def _why_not_promoted_sidecar(quality_analysis_artifact: dict[str, Any]) -> dict[str, Any]:
    decision_counts = dict(quality_analysis_artifact.get("decision_counts", {}))
    review_queue_rows = list(quality_analysis_artifact.get("review_queue_rows", []))
    hard_reject_rows = list(quality_analysis_artifact.get("hard_reject_rows", []))
    train_keep_rows = list(quality_analysis_artifact.get("train_keep_rows", []))
    silver_rows = list(quality_analysis_artifact.get("silver_rows", []))

    reason_counter: Counter[str] = Counter()
    for row in review_queue_rows:
        for code in list((row or {}).get("reason_codes") or []):
            reason_counter[str(code)] += 1
    top_reasons = [{"reason": reason, "count": int(count)} for reason, count in reason_counter.most_common(10)]

    reason_groups = dict(quality_analysis_artifact.get("reason_group_counts", {}))
    implicit_rejection_reason_counts = dict(quality_analysis_artifact.get("implicit_rejection_reason_counts", reason_groups))

    return {
        "decision_counts": decision_counts,
        "review_queue_rows": len(review_queue_rows),
        "hard_reject_rows": len(hard_reject_rows),
        "train_keep_rows": len(train_keep_rows),
        "silver_rows": len(silver_rows),
        "top_reasons": top_reasons,
        "reason_group_counts": reason_groups,
        "implicit_rejection_reason_counts": implicit_rejection_reason_counts,
    }


def _release_summary(report: dict[str, Any], quality_summary: dict[str, Any]) -> dict[str, Any]:
    strict_artifacts = report.get("strict_artifacts", {}) or {}
    blocking_reasons = list(report.get("blocking_reasons", []) or [])
    validation = report.get("validation", {}) or {}
    benchmark_counts = report.get("benchmark_artifact_counts", {}) or {}
    topup_effectiveness = report.get("topup_effectiveness", {}) or {}

    output_quality = report.get("output_quality", {}) or {}
    reason_counts = dict(output_quality.get("implicit_rejection_reason_counts", output_quality.get("reason_group_counts", {})))

    return {
        "pipeline_version": str(report.get("pipeline_version") or ""),
        "generated_at": str(report.get("generated_at") or ""),
        "run_profile": str(report.get("run_profile") or ""),
        "artifact_mode": str(report.get("artifact_mode") or ""),
        "gold_rows": int(strict_artifacts.get("strict_train_rows", 0)),
        "silver_rows": int(quality_summary.get("silver_count", 0)),
        "train_keep_rows": int(quality_summary.get("train_keep_count", 0)),
        "hard_reject_rows": int(quality_summary.get("hard_reject_count", 0)),
        "recoverable_rows": int(quality_summary.get("recoverable_count", 0)),
        "blocked": bool(blocking_reasons),
        "blocking_codes": [str(item.get("code") or "") for item in blocking_reasons if str(item.get("code") or "").strip()],
        "benchmark_artifact_counts_match": bool(validation.get("benchmark_artifact_counts_match", True)),
        "benchmark_artifact_counts": benchmark_counts,
        "topup_effectiveness": topup_effectiveness,
        "size_recovery_stage": str(report.get("size_recovery_stage") or "none"),
        "size_recovery_shortfall_remaining": int(report.get("size_recovery_shortfall_remaining") or 0),
        "implicit_rejection_reason_counts": reason_counts,
    }

