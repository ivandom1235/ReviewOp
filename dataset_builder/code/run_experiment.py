from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from build_dataset import run_pipeline
from contracts import BuilderConfig
from experiments import run_experiments
from research_stack import build_experiment_plan, benchmark_registry_payload, model_registry_payload
from utils import stable_id, utc_now_iso, write_json, compress_output_folder

try:
    from llm_utils import flush_llm_cache
except Exception:  # pragma: no cover
    def flush_llm_cache() -> None:
        return None

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
}
NOVELTY_GATES = {
    "required_ablations": ["explicit-only", "implicit-only", "no-grounding", "no-domain-conditioning", "llm-direct", "v5-full"],
    "full_model_must_outperform_count": 4,
}


def _parse_bounded_int_list(raw: str, *, minimum: int, maximum: int, fallback: list[int]) -> list[int]:
    values: list[int] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError:
            continue
        if minimum <= value <= maximum:
            values.append(value)
    unique_values = sorted(set(values))
    return unique_values or fallback


def _load_runtime_defaults() -> dict[str, Any]:
    defaults_path = Path(__file__).resolve().parent / "runtime_defaults.json"
    if not defaults_path.exists():
        return {}
    try:
        payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    defaults = payload.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def _metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    quality = report.get("output_quality", {})
    gold = report.get("gold_eval", {})
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
        "has_gold_eval": bool(gold.get("has_gold_labels", False)),
        "gold_rows": int(gold.get("num_rows_with_gold", 0) or 0),
        "gold_aspect_f1": float(gold.get("aspect_f1", 0.0)),
        "gold_sentiment_f1": float(gold.get("sentiment_f1", 0.0)),
        "gold_span_overlap_f1": float(gold.get("span_overlap_f1", 0.0)),
    }


def _core_score(metrics: dict[str, Any]) -> float:
    if metrics.get("has_gold_eval") and metrics.get("gold_rows", 0) > 0:
        return round(
            (
                metrics.get("gold_aspect_f1", 0.0)
                + metrics.get("gold_sentiment_f1", 0.0)
                + metrics.get("gold_span_overlap_f1", 0.0)
            ) / 3.0,
            4,
        )

    # Fallback proxy for novelty comparisons when gold labels are unavailable.
    fallback_component = max(0.0, 1.0 - float(metrics.get("fallback_only_rate", 1.0)))
    review_component = 1.0 / (1.0 + max(0, int(metrics.get("needs_review_rows", 0))))
    grounding_component = max(0.0, min(1.0, float(metrics.get("grounded_prediction_rate", 0.0))))
    return round(
        (
            fallback_component
            + review_component
            + grounding_component
        ) / 3.0,
        4,
    )


def _meets_quality_gates(metrics: dict[str, Any], *, quality_gates: dict[str, Any] = QUALITY_GATES) -> bool:
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
    )


def _rank_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
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
        0 if candidate["metrics"]["has_gold_eval"] else 1,
        -candidate["metrics"]["gold_aspect_f1"],
        -candidate["metrics"]["gold_sentiment_f1"],
        -candidate["metrics"]["gold_span_overlap_f1"],
        candidate["candidate_id"],
    )


def _run_ablation_matrix(cfg: BuilderConfig, run_dir: Path) -> dict[str, Any]:
    ablation_configs: list[tuple[str, BuilderConfig]] = [
        ("v5-full", replace(cfg, enable_reasoned_recovery=True)),
        ("no-grounding", replace(cfg, enable_reasoned_recovery=True, enforce_grounding=False)),
        ("no-domain-conditioning", replace(cfg, enable_reasoned_recovery=True, use_domain_conditioning=False)),
        ("implicit-only", replace(cfg, enable_reasoned_recovery=True, use_domain_conditioning=True, enforce_grounding=True)),
        ("explicit-only", replace(cfg, implicit_min_tokens=999, enforce_grounding=False, enable_llm_fallback=False, enable_reasoned_recovery=False)),
        ("llm-direct", replace(cfg, implicit_mode="zeroshot", enable_llm_fallback=True, enable_reasoned_recovery=True, implicit_min_tokens=1)),
    ]
    rows: list[dict[str, Any]] = []
    for name, ab_cfg in ablation_configs:
        out_dir = run_dir / "ablations" / name
        run_cfg = replace(ab_cfg, output_dir=out_dir, dry_run=False, preview_only=False)
        report = run_pipeline(run_cfg)
        metrics = _metrics_from_report(report)
        rows.append({
            "name": name,
            "report_path": str(out_dir / "reports" / "build_report.json"),
            "metrics": metrics,
            "core_score": _core_score(metrics),
        })
    by_name = {row["name"]: row for row in rows}
    full = by_name.get("v5-full")
    outperform = 0
    if full:
        for name, row in by_name.items():
            if name == "full":
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


def _bounded_v4_candidates(
    include_coref: bool,
    *,
    implicit_min_tokens_values: list[int],
    min_text_tokens_values: list[int],
) -> list[dict[str, Any]]:
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
        candidates.append({
            "candidate_id": f"cand_{index:03d}",
            "implicit_mode": implicit_mode,
            "confidence_threshold": confidence_threshold,
            "llm_fallback_threshold": llm_fallback_threshold,
            "use_coref": use_coref,
            "implicit_min_tokens": implicit_min_tokens,
            "min_text_tokens": min_text_tokens,
        })
    return candidates


def _execute_v4_sweep(
    cfg: BuilderConfig,
    run_dir: Path,
    *,
    include_coref: bool,
    implicit_min_tokens_values: list[int],
    min_text_tokens_values: list[int],
    quality_gates: dict[str, Any] = QUALITY_GATES,
) -> dict[str, Any]:
    candidates = _bounded_v4_candidates(
        include_coref=include_coref,
        implicit_min_tokens_values=implicit_min_tokens_values,
        min_text_tokens_values=min_text_tokens_values,
    )
    candidate_results: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_output_dir = run_dir / "candidates" / candidate["candidate_id"]
        candidate_cfg = replace(
            cfg,
            output_dir=candidate_output_dir,
            implicit_mode=candidate["implicit_mode"],
            confidence_threshold=candidate["confidence_threshold"],
            llm_fallback_threshold=candidate["llm_fallback_threshold"],
            use_coref=candidate["use_coref"],
            implicit_min_tokens=candidate["implicit_min_tokens"],
            min_text_tokens=candidate["min_text_tokens"],
            dry_run=False,
            preview_only=False,
        )
        report = run_pipeline(candidate_cfg)
        metrics = _metrics_from_report(report)
        candidate_results.append({
            **candidate,
            "output_dir": str(candidate_output_dir),
            "report_path": str(candidate_output_dir / "reports" / "build_report.json"),
            "metrics": metrics,
            "meets_quality_gates": _meets_quality_gates(metrics, quality_gates=quality_gates),
            "generated_at": report.get("generated_at"),
            "validation": report.get("validation", {}),
        })

    ranked = sorted(candidate_results, key=_rank_key)
    best = ranked[0] if ranked else None
    ablation_summary = _run_ablation_matrix(cfg, run_dir)
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

    summary = {
        "generated_at": utc_now_iso(),
        "quality_gates": quality_gates,
        "novelty_gates": NOVELTY_GATES,
        "sweep_dimensions": {
            "implicit_mode": ["zeroshot", "hybrid"],
            "confidence_threshold": [0.55, 0.6],
            "llm_fallback_threshold": [0.55, 0.6, 0.65],
            "use_coref": [False, True] if include_coref else [False],
            "implicit_min_tokens": implicit_min_tokens_values,
            "min_text_tokens": min_text_tokens_values,
        },
        "candidate_count": len(candidate_results),
        "meets_any_quality_gates": any(candidate["meets_quality_gates"] for candidate in candidate_results),
        "best_candidate_id": best["candidate_id"] if best else None,
        "best_candidate": best,
        "promoted_defaults": promoted_defaults,
        "ablation_summary": ablation_summary,
        "ranked_candidates": ranked,
    }
    write_json(run_dir / "v4_sweep_results.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    runtime_defaults = _load_runtime_defaults()
    parser = argparse.ArgumentParser(description="Dataset builder research experiment runner")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--run-profile", type=str, default="research", choices=["research", "debug"])
    parser.add_argument("--confidence-threshold", type=float, default=float(runtime_defaults.get("confidence_threshold", 0.6)))
    parser.add_argument("--max-aspects", type=int, default=20)
    parser.add_argument("--min-text-tokens", type=int, default=4)
    parser.add_argument("--implicit-min-tokens", type=int, default=8)
    parser.add_argument("--implicit-mode", type=str, default=str(runtime_defaults.get("implicit_mode", "zeroshot")), choices=["zeroshot", "supervised", "hybrid", "heuristic", "benchmark"])
    parser.add_argument("--multilingual-mode", type=str, default="shared_vocab")
    parser.add_argument("--use-coref", dest="use_coref", action="store_true")
    parser.add_argument("--no-use-coref", dest="use_coref", action="store_false")
    parser.set_defaults(use_coref=bool(runtime_defaults.get("use_coref", False)))
    parser.add_argument("--language-detection-mode", type=str, default="heuristic")
    parser.add_argument("--no-drop", action="store_true")
    parser.add_argument("--enable-llm-fallback", dest="enable_llm_fallback", action="store_true")
    parser.add_argument("--no-enable-llm-fallback", dest="enable_llm_fallback", action="store_false")
    parser.set_defaults(enable_llm_fallback=bool(runtime_defaults.get("enable_llm_fallback", True)))
    parser.add_argument("--llm-fallback-threshold", type=float, default=float(runtime_defaults.get("llm_fallback_threshold", 0.65)))
    parser.add_argument("--benchmark-key", type=str, default=None)
    parser.add_argument("--model-family", type=str, default="heuristic_latent")
    parser.add_argument("--augmentation-mode", type=str, default="none")
    parser.add_argument("--prompt-mode", type=str, default="constrained")
    parser.add_argument("--gold-annotations-path", type=Path, default=None)
    parser.add_argument("--evaluation-protocol", type=str, default="random", choices=["random", "loo", "source-free"])
    parser.add_argument("--domain-holdout", type=str, default=None)
    parser.add_argument("--no-enforce-grounding", dest="enforce_grounding", action="store_false")
    parser.add_argument("--no-domain-conditioning", dest="use_domain_conditioning", action="store_false")
    parser.add_argument("--no-strict-domain-conditioning", dest="strict_domain_conditioning", action="store_false")
    parser.add_argument("--domain-conditioning-mode", type=str, default="adaptive_soft", choices=["adaptive_soft", "strict_hard", "off"])
    parser.add_argument("--train-domain-conditioning-mode", type=str, default=None, choices=["adaptive_soft", "strict_hard", "off"])
    parser.add_argument("--eval-domain-conditioning-mode", type=str, default=None, choices=["adaptive_soft", "strict_hard", "off"])
    parser.set_defaults(enforce_grounding=True, use_domain_conditioning=True, strict_domain_conditioning=False)
    parser.add_argument("--domain-prior-boost", type=float, default=0.05)
    parser.add_argument("--domain-prior-penalty", type=float, default=0.08)
    parser.add_argument("--weak-domain-support-row-threshold", type=int, default=80)
    parser.add_argument("--unseen-non-general-coverage-min", type=float, default=0.55)
    parser.add_argument("--unseen-implicit-not-ready-rate-max", type=float, default=0.35)
    parser.add_argument("--unseen-domain-leakage-row-rate-max", type=float, default=0.02)
    parser.add_argument("--train-fallback-general-policy", type=str, default="cap", choices=["keep", "cap", "drop"])
    parser.add_argument("--enable-reasoned-recovery", dest="enable_reasoned_recovery", action="store_true")
    parser.add_argument("--no-enable-reasoned-recovery", dest="enable_reasoned_recovery", action="store_false")
    # Backward-compatible alias for older scripts.
    parser.add_argument("--no-enable-reasoned_recovery", dest="enable_reasoned_recovery", action="store_false")
    parser.set_defaults(enable_reasoned_recovery=bool(runtime_defaults.get("enable_reasoned_recovery", True)))
    parser.add_argument("--llm-provider", type=str, default=str(runtime_defaults.get("llm_provider", "runpod")), choices=["runpod", "openai", "anthropic", "ollama"])
    parser.add_argument("--llm-model-name", type=str, default=str(runtime_defaults.get("llm_model_name", "llama3-8b-instruct")))
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--train-fallback-general-cap-ratio", type=float, default=0.15)
    parser.add_argument("--train-review-filter-mode", type=str, default="reasoned_strict", choices=["keep", "drop_needs_review", "reasoned_strict"])
    parser.add_argument("--train-salvage-mode", type=str, default="recover_non_general", choices=["off", "recover_non_general"])
    parser.add_argument("--train-salvage-confidence-threshold", type=float, default=0.56)
    parser.add_argument("--train-salvage-accepted-support-types", type=str, default="exact,near_exact,gold")
    parser.add_argument(
        "--train-sentiment-balance-mode",
        type=str,
        default="cap_neutral_with_dual_floor",
        choices=["none", "cap_neutral", "cap_neutral_with_negative_floor", "cap_neutral_with_dual_floor"],
    )
    parser.add_argument("--train-neutral-cap-ratio", type=float, default=0.5)
    parser.add_argument("--train-min-negative-ratio", type=float, default=0.12)
    parser.add_argument("--train-min-positive-ratio", type=float, default=0.12)
    parser.add_argument("--train-max-positive-ratio", type=float, default=0.5)
    parser.add_argument("--train-neutral-max-ratio", type=float, default=0.58)
    parser.add_argument("--train-topup-recovery-mode", type=str, default="strict_topup", choices=["off", "strict_topup"])
    parser.add_argument("--train-topup-confidence-threshold", type=float, default=0.58)
    parser.add_argument("--train-topup-staged-recovery", dest="train_topup_staged_recovery", action="store_true")
    parser.add_argument("--no-train-topup-staged-recovery", dest="train_topup_staged_recovery", action="store_false")
    parser.set_defaults(train_topup_staged_recovery=True)
    parser.add_argument("--train-topup-stage-b-confidence-threshold", type=float, default=0.54)
    parser.add_argument("--train-topup-allow-weak-support-in-stage-c", dest="train_topup_allow_weak_support_in_stage_c", action="store_true")
    parser.add_argument("--no-train-topup-allow-weak-support-in-stage-c", dest="train_topup_allow_weak_support_in_stage_c", action="store_false")
    parser.set_defaults(train_topup_allow_weak_support_in_stage_c=True)
    parser.add_argument("--train-topup-stage-c-confidence-threshold", type=float, default=0.52)
    parser.add_argument("--train-topup-allowed-support-types", type=str, default="exact,near_exact,gold")
    parser.add_argument("--train-target-min-rows", type=int, default=2200)
    parser.add_argument("--train-target-max-rows", type=int, default=2500)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute-baseline", action="store_true")
    parser.add_argument("--execute-v4-sweep", action="store_true")
    parser.add_argument("--include-coref", action="store_true")
    parser.add_argument("--apply-best-defaults", action="store_true")
    parser.add_argument("--sweep-implicit-min-tokens", type=str, default="6,8")
    parser.add_argument("--sweep-min-text-tokens", type=str, default="3,4")
    parser.add_argument("--gold-min-rows-for-promotion", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    domain_conditioning_mode = str(args.domain_conditioning_mode or "").strip().lower()
    if not args.use_domain_conditioning:
        domain_conditioning_mode = "off"
    elif args.strict_domain_conditioning and domain_conditioning_mode == "adaptive_soft":
        domain_conditioning_mode = "strict_hard"
    train_domain_conditioning_mode = str(args.train_domain_conditioning_mode or "").strip().lower() or None
    eval_domain_conditioning_mode = str(args.eval_domain_conditioning_mode or "").strip().lower() or None
    if train_domain_conditioning_mode is None or eval_domain_conditioning_mode is None:
        if domain_conditioning_mode == "strict_hard":
            train_domain_conditioning_mode = train_domain_conditioning_mode or "strict_hard"
            eval_domain_conditioning_mode = eval_domain_conditioning_mode or "strict_hard"
        elif domain_conditioning_mode == "off":
            train_domain_conditioning_mode = train_domain_conditioning_mode or "off"
            eval_domain_conditioning_mode = eval_domain_conditioning_mode or "off"
        else:
            train_domain_conditioning_mode = train_domain_conditioning_mode or "strict_hard"
            eval_domain_conditioning_mode = eval_domain_conditioning_mode or "adaptive_soft"
    cfg = BuilderConfig(
        input_dir=args.input_dir or BuilderConfig().input_dir,
        output_dir=args.output_dir or BuilderConfig().output_dir,
        random_seed=args.seed,
        text_column_override=args.text_column,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_offset=args.chunk_offset,
        run_profile=args.run_profile,
        confidence_threshold=args.confidence_threshold,
        max_aspects=args.max_aspects,
        min_text_tokens=args.min_text_tokens,
        implicit_min_tokens=args.implicit_min_tokens,
        implicit_mode=args.implicit_mode,
        multilingual_mode=args.multilingual_mode,
        use_coref=args.use_coref,
        language_detection_mode=args.language_detection_mode,
        no_drop=args.no_drop,
        enable_llm_fallback=args.enable_llm_fallback,
        llm_fallback_threshold=args.llm_fallback_threshold,
        benchmark_key=args.benchmark_key,
        model_family=args.model_family,
        augmentation_mode=args.augmentation_mode,
        prompt_mode=args.prompt_mode,
        gold_annotations_path=args.gold_annotations_path,
        evaluation_protocol=args.evaluation_protocol,
        domain_holdout=args.domain_holdout,
        enforce_grounding=args.enforce_grounding,
        use_domain_conditioning=args.use_domain_conditioning,
        strict_domain_conditioning=args.strict_domain_conditioning,
        domain_conditioning_mode=domain_conditioning_mode,
        train_domain_conditioning_mode=str(train_domain_conditioning_mode),
        eval_domain_conditioning_mode=str(eval_domain_conditioning_mode),
        domain_prior_boost=args.domain_prior_boost,
        domain_prior_penalty=args.domain_prior_penalty,
        weak_domain_support_row_threshold=args.weak_domain_support_row_threshold,
        unseen_non_general_coverage_min=args.unseen_non_general_coverage_min,
        unseen_implicit_not_ready_rate_max=args.unseen_implicit_not_ready_rate_max,
        unseen_domain_leakage_row_rate_max=args.unseen_domain_leakage_row_rate_max,
        enable_reasoned_recovery=args.enable_reasoned_recovery,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model_name,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_max_retries=args.llm_max_retries,
        train_fallback_general_policy=args.train_fallback_general_policy,
        train_fallback_general_cap_ratio=args.train_fallback_general_cap_ratio,
        train_review_filter_mode=args.train_review_filter_mode,
        train_salvage_mode=args.train_salvage_mode,
        train_salvage_confidence_threshold=args.train_salvage_confidence_threshold,
        train_salvage_accepted_support_types=tuple(part.strip() for part in str(args.train_salvage_accepted_support_types).split(",") if part.strip()),
        train_sentiment_balance_mode=args.train_sentiment_balance_mode,
        train_neutral_cap_ratio=args.train_neutral_cap_ratio,
        train_min_negative_ratio=args.train_min_negative_ratio,
        train_min_positive_ratio=args.train_min_positive_ratio,
        train_max_positive_ratio=args.train_max_positive_ratio,
        train_neutral_max_ratio=args.train_neutral_max_ratio,
        train_topup_recovery_mode=args.train_topup_recovery_mode,
        train_topup_confidence_threshold=args.train_topup_confidence_threshold,
        train_topup_staged_recovery=args.train_topup_staged_recovery,
        train_topup_stage_b_confidence_threshold=args.train_topup_stage_b_confidence_threshold,
        train_topup_allow_weak_support_in_stage_c=args.train_topup_allow_weak_support_in_stage_c,
        train_topup_stage_c_confidence_threshold=args.train_topup_stage_c_confidence_threshold,
        train_topup_allowed_support_types=tuple(part.strip() for part in str(args.train_topup_allowed_support_types).split(",") if part.strip()),
        train_target_min_rows=args.train_target_min_rows,
        train_target_max_rows=args.train_target_max_rows,
    )

    run_id = stable_id(
        cfg.input_dir,
        cfg.output_dir,
        cfg.benchmark_key or "auto",
        cfg.model_family,
        cfg.random_seed,
        cfg.sample_size,
        cfg.chunk_size,
        cfg.chunk_offset,
        cfg.implicit_mode,
    )
    run_dir = cfg.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "benchmark_registry.json", benchmark_registry_payload())
    write_json(run_dir / "model_registry.json", model_registry_payload())
    write_json(run_dir / "experiment_plan.json", [asdict(item) for item in build_experiment_plan()])
    write_json(run_dir / "base_config.json", asdict(cfg))

    if args.plan_only:
        write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "status": "planned",
            "config": asdict(cfg),
        })
        flush_llm_cache()
        if not cfg.dry_run:
            zip_path = compress_output_folder(cfg.output_dir)
            if zip_path:
                print(f"Output compressed: {zip_path}")
        return 0

    if args.execute_baseline:
        baseline_cfg = replace(cfg)
        report = run_pipeline(baseline_cfg)
        write_json(run_dir / "baseline_report.json", report)
        write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "status": "completed",
            "config": asdict(cfg),
            "report": report,
        })
        flush_llm_cache()
        if not cfg.dry_run:
            zip_path = compress_output_folder(cfg.output_dir)
            if zip_path:
                print(f"Output compressed: {zip_path}")
        return 0

    if args.execute_v4_sweep:
        quality_gates = dict(QUALITY_GATES)
        quality_gates["gold_min_rows_for_promotion"] = max(1, int(args.gold_min_rows_for_promotion))
        quality_gates["unseen_non_general_coverage_min"] = float(args.unseen_non_general_coverage_min)
        quality_gates["unseen_implicit_not_ready_rate_max"] = float(args.unseen_implicit_not_ready_rate_max)
        quality_gates["unseen_domain_leakage_row_rate_max"] = float(args.unseen_domain_leakage_row_rate_max)
        implicit_min_tokens_values = _parse_bounded_int_list(
            args.sweep_implicit_min_tokens,
            minimum=4,
            maximum=16,
            fallback=[6, 8],
        )
        min_text_tokens_values = _parse_bounded_int_list(
            args.sweep_min_text_tokens,
            minimum=2,
            maximum=12,
            fallback=[3, 4],
        )
        sweep = _execute_v4_sweep(
            cfg,
            run_dir,
            include_coref=args.include_coref,
            implicit_min_tokens_values=implicit_min_tokens_values,
            min_text_tokens_values=min_text_tokens_values,
            quality_gates=quality_gates,
        )
        promoted_defaults = sweep.get("promoted_defaults")
        if args.apply_best_defaults and promoted_defaults:
            defaults_path = Path(__file__).resolve().parent / "runtime_defaults.json"
            write_json(defaults_path, {
                "generated_at": utc_now_iso(),
                "source_run_id": run_id,
                "quality_gates": quality_gates,
                "defaults": promoted_defaults,
            })
        write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "status": "completed_v4_sweep",
            "config": asdict(cfg),
            "quality_gates": quality_gates,
            "best_candidate_id": sweep.get("best_candidate_id"),
            "promoted_defaults": promoted_defaults if args.apply_best_defaults else None,
            "defaults_applied": bool(args.apply_best_defaults and promoted_defaults),
        })
        flush_llm_cache()
        if not cfg.dry_run:
            zip_path = compress_output_folder(cfg.output_dir)
            if zip_path:
                print(f"Output compressed: {zip_path}")
        return 0

    run_experiments(
        cfg,
        [{
            "model_family": cfg.model_family,
            "benchmark_key": cfg.benchmark_key,
            "implicit_mode": cfg.implicit_mode,
            "multilingual_mode": cfg.multilingual_mode,
            "use_coref": cfg.use_coref,
        }],
        run_dir,
    )
    write_json(run_dir / "manifest.json", {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "status": "configured",
        "config": asdict(cfg),
    })
    flush_llm_cache()
    if not cfg.dry_run:
        zip_path = compress_output_folder(cfg.output_dir)
        if zip_path:
            print(f"Output compressed: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
