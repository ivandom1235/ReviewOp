from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path
from typing import Any

try:
    from .contracts import BuilderConfig
    from .pipeline_runner import run_pipeline_sync
    from .utils import utc_now_iso, write_json
except ImportError:  # pragma: no cover
    from contracts import BuilderConfig
    from pipeline_runner import run_pipeline_sync
    from utils import utc_now_iso, write_json


QUALITY_GATES = {
    "fallback_only_rate_max": 0.22,
    "needs_review_rows_max": 1800,
    "generic_implicit_aspects_max": 0,
    "rejected_implicit_aspects_max": 0,
    "domain_leakage_row_rate_max": 0.06,
    "grounded_prediction_rate_min": 0.75,
    "train_general_dominance_rate_max": 0.2,
    "train_domain_leakage_row_rate_max": 0.0,
    "train_negative_ratio_min": 0.12,
    "train_positive_ratio_min": 0.12,
    "train_positive_ratio_max": 0.5,
    "train_neutral_ratio_max": 0.58,
    "train_target_size_compliant_required": 1,
    "unseen_non_general_coverage_min": 0.55,
    "unseen_implicit_not_ready_rate_max": 0.35,
    "unseen_domain_leakage_row_rate_max": 0.02,
    "gold_min_rows_for_promotion": 600,
    "gold_aspect_f1_min": 0.55,
    "gold_sentiment_f1_min": 0.55,
    "gold_span_overlap_f1_min": 0.4,
    "benchmark_implicit_purity_rate_min": 0.7,
    "benchmark_ontology_compatibility_rate_min": 0.9,
    "benchmark_duplicate_logical_row_rate_adjusted_max": 0.45,
    "worst_domain_f1_min": 0.4,
}
NOVELTY_GATES = {
    "required_ablations": ["explicit-only", "implicit-only", "no-grounding", "no-domain-conditioning", "llm-direct", "v6-full"],
    "full_model_must_outperform_count": 4,
}


def metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    quality = report.get("output_quality", {})
    gold = report.get("gold_eval", {})
    benchmark_gold = report.get("benchmark_gold_eval", {})
    has_benchmark_gold = bool(benchmark_gold.get("has_gold_interpretations", False))
    return {
        "fallback_only_rate": float(quality.get("fallback_only_rate", 1.0)),
        "needs_review_rows": int(quality.get("needs_review_rows", 10**9)),
        "generic_implicit_aspects": int(quality.get("generic_implicit_aspects", 10**9)),
        "rejected_implicit_aspects": int(quality.get("rejected_implicit_aspects", 10**9)),
        "domain_leakage_row_rate": float(quality.get("domain_leakage_row_rate", 1.0)),
        "grounded_prediction_rate": float(report.get("grounded_prediction_rate", 0.0)),
        "train_general_dominance_rate": float(report.get("train_general_dominance_rate", 1.0)),
        "train_domain_leakage_row_rate": float(report.get("train_domain_leakage_row_rate", 1.0)),
        "train_negative_ratio": float(report.get("train_negative_ratio", 0.0)),
        "train_positive_ratio": float(report.get("train_positive_ratio", 0.0)),
        "train_neutral_ratio": float(report.get("train_sentiment_constraints", {}).get("achieved", {}).get("neutral_ratio", 1.0)),
        "train_target_size_compliant": bool(report.get("train_target_stats", {}).get("size_within_target_range", False)),
        "unseen_non_general_coverage": float(report.get("unseen_domain_metrics", {}).get("unseen_non_general_coverage", 0.0)),
        "unseen_implicit_not_ready_rate": float(report.get("unseen_domain_metrics", {}).get("unseen_implicit_not_ready_rate", 1.0)),
        "unseen_domain_leakage_row_rate": float(report.get("unseen_domain_metrics", {}).get("unseen_domain_leakage_row_rate", 1.0)),
        "ungrounded_non_general_count": int(report.get("ungrounded_non_general_count", 10**9)),
        "has_gold_eval": bool(gold.get("has_gold_labels", False) or has_benchmark_gold),
        "has_benchmark_gold_eval": has_benchmark_gold,
        "gold_rows": int(gold.get("num_rows_with_gold", 0) or benchmark_gold.get("num_rows_with_gold_interpretations", 0) or 0),
        "gold_aspect_f1": float(gold.get("aspect_f1", 0.0)),
        "gold_sentiment_f1": float(gold.get("sentiment_f1", 0.0)),
        "gold_span_overlap_f1": float(gold.get("span_overlap_f1", 0.0)),
        "benchmark_gold_rows": int(benchmark_gold.get("num_rows_with_gold_interpretations", 0) or 0),
        "benchmark_average_gold_interpretations": float(benchmark_gold.get("average_gold_interpretations", 0.0)),
        "benchmark_multi_gold_label_rate": float(benchmark_gold.get("multi_gold_label_rate", 0.0)),
        "benchmark_grounded_evidence_rate": float(benchmark_gold.get("grounded_evidence_rate", 0.0)),
        "benchmark_duplicate_interpretation_rate": float(benchmark_gold.get("duplicate_interpretation_rate", 0.0)),
        "benchmark_duplicate_logical_row_rate_adjusted": float(report.get("benchmark_summary", {}).get("duplicate_logical_row_rate_adjusted", 1.0)),
        "benchmark_implicit_purity_rate": float(benchmark_gold.get("implicit_purity_rate", 0.0)),
        "benchmark_ontology_compatibility_rate": float(benchmark_gold.get("ontology_compatibility_rate", 0.0)),
        "worst_domain_f1": float(report.get("robust_training_eval", {}).get("groupdro", {}).get("worst_domain_f1", 0.0)),
        "promotion_guard_blocked": bool(report.get("promotion_guard", {}).get("blocked", False)),
    }


def core_score(metrics: dict[str, Any]) -> float:
    if metrics.get("has_gold_eval") and metrics.get("gold_rows", 0) > 0:
        if any(float(metrics.get(key, 0.0)) > 0.0 for key in ("gold_aspect_f1", "gold_sentiment_f1", "gold_span_overlap_f1")):
            return round(
                (
                    metrics.get("gold_aspect_f1", 0.0)
                    + metrics.get("gold_sentiment_f1", 0.0)
                    + metrics.get("gold_span_overlap_f1", 0.0)
                ) / 3.0,
                4,
            )
    if metrics.get("has_benchmark_gold_eval") and metrics.get("benchmark_gold_rows", 0) > 0:
        return round(
            (
                float(metrics.get("benchmark_grounded_evidence_rate", 0.0))
                + max(0.0, 1.0 - float(metrics.get("benchmark_duplicate_interpretation_rate", 1.0)))
                + float(metrics.get("benchmark_multi_gold_label_rate", 0.0))
            ) / 3.0,
            4,
        )

    fallback_component = max(0.0, 1.0 - float(metrics.get("fallback_only_rate", 1.0)))
    review_component = 1.0 / (1.0 + max(0, int(metrics.get("needs_review_rows", 0))))
    grounding_component = max(0.0, min(1.0, float(metrics.get("grounded_prediction_rate", 0.0))))
    return round((fallback_component + review_component + grounding_component) / 3.0, 4)


def meets_quality_gates(metrics: dict[str, Any], *, quality_gates: dict[str, Any] = QUALITY_GATES) -> bool:
    base_ok = (
        metrics["fallback_only_rate"] <= quality_gates["fallback_only_rate_max"]
        and metrics["needs_review_rows"] <= quality_gates["needs_review_rows_max"]
        and metrics["generic_implicit_aspects"] <= quality_gates["generic_implicit_aspects_max"]
        and metrics["rejected_implicit_aspects"] <= quality_gates["rejected_implicit_aspects_max"]
        and metrics["domain_leakage_row_rate"] <= quality_gates["domain_leakage_row_rate_max"]
        and metrics["grounded_prediction_rate"] >= quality_gates["grounded_prediction_rate_min"]
        and metrics["train_general_dominance_rate"] <= quality_gates["train_general_dominance_rate_max"]
        and metrics["train_domain_leakage_row_rate"] <= quality_gates["train_domain_leakage_row_rate_max"]
        and metrics["train_negative_ratio"] >= quality_gates["train_negative_ratio_min"]
        and metrics["train_positive_ratio"] >= quality_gates["train_positive_ratio_min"]
        and metrics["train_positive_ratio"] <= quality_gates["train_positive_ratio_max"]
        and metrics["train_neutral_ratio"] <= quality_gates["train_neutral_ratio_max"]
        and int(bool(metrics["train_target_size_compliant"])) >= int(quality_gates["train_target_size_compliant_required"])
        and metrics["unseen_non_general_coverage"] >= quality_gates["unseen_non_general_coverage_min"]
        and metrics["unseen_implicit_not_ready_rate"] <= quality_gates["unseen_implicit_not_ready_rate_max"]
        and metrics["unseen_domain_leakage_row_rate"] <= quality_gates["unseen_domain_leakage_row_rate_max"]
    )
    if not base_ok:
        return False
    if (not metrics["has_gold_eval"]) or metrics.get("gold_rows", 0) < int(quality_gates["gold_min_rows_for_promotion"]):
        return False
    return (
        metrics["gold_aspect_f1"] >= quality_gates["gold_aspect_f1_min"]
        and metrics["gold_sentiment_f1"] >= quality_gates["gold_sentiment_f1_min"]
        and metrics["gold_span_overlap_f1"] >= quality_gates["gold_span_overlap_f1_min"]
        and metrics["benchmark_implicit_purity_rate"] >= quality_gates["benchmark_implicit_purity_rate_min"]
        and metrics["benchmark_ontology_compatibility_rate"] >= quality_gates["benchmark_ontology_compatibility_rate_min"]
        and metrics["benchmark_duplicate_logical_row_rate_adjusted"] <= quality_gates["benchmark_duplicate_logical_row_rate_adjusted_max"]
        and metrics["worst_domain_f1"] >= quality_gates["worst_domain_f1_min"]
        and not metrics["promotion_guard_blocked"]
    )


def rank_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if candidate["meets_quality_gates"] else 1,
        candidate["metrics"]["fallback_only_rate"],
        candidate["metrics"]["needs_review_rows"],
        candidate["metrics"]["generic_implicit_aspects"],
        candidate["metrics"]["rejected_implicit_aspects"],
        candidate["metrics"]["domain_leakage_row_rate"],
        candidate["metrics"]["unseen_domain_leakage_row_rate"],
        candidate["metrics"]["train_domain_leakage_row_rate"],
        -candidate["metrics"]["unseen_non_general_coverage"],
        candidate["metrics"]["unseen_implicit_not_ready_rate"],
        -candidate["metrics"]["train_negative_ratio"],
        -candidate["metrics"]["train_positive_ratio"],
        candidate["metrics"]["train_neutral_ratio"],
        0 if candidate["metrics"]["train_target_size_compliant"] else 1,
        candidate["metrics"]["train_general_dominance_rate"],
        -candidate["metrics"]["grounded_prediction_rate"],
        -candidate["metrics"]["worst_domain_f1"],
        1 if candidate["metrics"]["promotion_guard_blocked"] else 0,
        0 if candidate["metrics"]["has_gold_eval"] else 1,
        -candidate["metrics"]["gold_aspect_f1"],
        -candidate["metrics"]["gold_sentiment_f1"],
        -candidate["metrics"]["gold_span_overlap_f1"],
        candidate["candidate_id"],
    )


def ablation_configs(cfg: BuilderConfig) -> list[tuple[str, BuilderConfig]]:
    return [
        ("v6-full", replace(cfg, enable_reasoned_recovery=True)),
        ("no-grounding", replace(cfg, enable_reasoned_recovery=True, enforce_grounding=False)),
        ("no-domain-conditioning", replace(cfg, enable_reasoned_recovery=True, use_domain_conditioning=False)),
        ("implicit-only", replace(cfg, enable_reasoned_recovery=True, use_domain_conditioning=True, enforce_grounding=True)),
        ("explicit-only", replace(cfg, implicit_min_tokens=999, enforce_grounding=False, enable_llm_fallback=False, enable_reasoned_recovery=False)),
        ("llm-direct", replace(cfg, implicit_mode="zeroshot", enable_llm_fallback=True, enable_reasoned_recovery=True, implicit_min_tokens=1)),
    ]


def candidate_grid(*, include_coref: bool, implicit_min_tokens_values: list[int], min_text_tokens_values: list[int]) -> list[dict[str, Any]]:
    implicit_modes = ["zeroshot", "hybrid"]
    confidence_thresholds = [0.55, 0.6]
    llm_fallback_thresholds = [0.55, 0.6, 0.65]
    use_coref_values = [False, True] if include_coref else [False]
    candidates: list[dict[str, Any]] = []
    for index, (implicit_mode, confidence_threshold, llm_fallback_threshold, use_coref, implicit_min_tokens, min_text_tokens) in enumerate(
        itertools.product(
            implicit_modes,
            confidence_thresholds,
            llm_fallback_thresholds,
            use_coref_values,
            implicit_min_tokens_values,
            min_text_tokens_values,
        ),
        start=1,
    ):
        candidates.append(
            {
                "candidate_id": f"cand_{index:03d}",
                "implicit_mode": implicit_mode,
                "confidence_threshold": confidence_threshold,
                "llm_fallback_threshold": llm_fallback_threshold,
                "use_coref": use_coref,
                "implicit_min_tokens": implicit_min_tokens,
                "min_text_tokens": min_text_tokens,
            }
        )
    return candidates


def build_sweep_summary(
    *,
    cfg: BuilderConfig,
    run_dir: Path,
    candidate_results: list[dict[str, Any]],
    include_coref: bool,
    ablation_summary: dict[str, Any] | None = None,
    quality_gates: dict[str, Any] = QUALITY_GATES,
) -> dict[str, Any]:
    ranked = sorted(candidate_results, key=rank_key)
    best = ranked[0] if ranked else None
    if ablation_summary is None:
        ablation_summary = run_ablation_matrix(cfg, run_dir)
    promoted_defaults = None
    if best and best["meets_quality_gates"] and ablation_summary["novelty_gate_passed"]:
        promoted_defaults = {
            "implicit_mode": best["implicit_mode"],
            "confidence_threshold": best["confidence_threshold"],
            "llm_fallback_threshold": best["llm_fallback_threshold"],
            "use_coref": best["use_coref"],
            "implicit_min_tokens": best["implicit_min_tokens"],
            "min_text_tokens": best["min_text_tokens"],
        }

    return {
        "generated_at": utc_now_iso(),
        "quality_gates": quality_gates,
        "novelty_gates": NOVELTY_GATES,
        "sweep_dimensions": {
            "implicit_mode": ["zeroshot", "hybrid"],
            "confidence_threshold": [0.55, 0.6],
            "llm_fallback_threshold": [0.55, 0.6, 0.65],
            "use_coref": [False, True] if include_coref else [False],
            "implicit_min_tokens": sorted({candidate["implicit_min_tokens"] for candidate in candidate_results}),
            "min_text_tokens": sorted({candidate["min_text_tokens"] for candidate in candidate_results}),
        },
        "candidate_count": len(candidate_results),
        "meets_any_quality_gates": any(candidate["meets_quality_gates"] for candidate in candidate_results),
        "best_candidate_id": best["candidate_id"] if best else None,
        "best_candidate": best,
        "promoted_defaults": promoted_defaults,
        "ablation_summary": ablation_summary,
        "ranked_candidates": ranked,
    }


def run_ablation_matrix(cfg: BuilderConfig, run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, ab_cfg in ablation_configs(cfg):
        out_dir = run_dir / "ablations" / name
        run_cfg = replace(ab_cfg, output_dir=out_dir, dry_run=False, preview_only=False)
        report = run_pipeline_sync(run_cfg)
        metrics = metrics_from_report(report)
        rows.append(
            {
                "name": name,
                "report_path": str(out_dir / "reports" / "build_report.json"),
                "metrics": metrics,
                "core_score": core_score(metrics),
            }
        )
    by_name = {row["name"]: row for row in rows}
    full = by_name.get("v6-full")
    outperform = 0
    if full:
        for name, row in by_name.items():
            if name == "v6-full":
                continue
            if full["core_score"] > row["core_score"]:
                outperform += 1
    summary = {
        "required_ablations": NOVELTY_GATES["required_ablations"],
        "ablation_rows": rows,
        "full_model_outperform_count": outperform,
        "novelty_gate_passed": outperform >= NOVELTY_GATES["full_model_must_outperform_count"],
    }
    write_json(run_dir / "ablation_summary.json", summary)
    return summary
