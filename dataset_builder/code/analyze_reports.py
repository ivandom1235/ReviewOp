from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import utc_now_iso, write_json


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(current: float | int, previous: float | int | None) -> float | None:
    if previous is None:
        return None
    return round(float(current) - float(previous), 4)


def _score_from_verdict(*, usable_for_training: bool, research_ready: bool, publication_ready: bool) -> tuple[float, str]:
    if publication_ready:
        return 9.0, "Research-ready"
    if research_ready:
        return 8.0, "Training-ready"
    if usable_for_training:
        return 6.0, "Experimental"
    return 3.0, "Not usable"


def _build_scorecard(build: dict[str, Any], quality: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    row_counts = build.get("row_counts", {})
    output_quality = build.get("output_quality", {})
    strict_quality = build.get("strict_quality", output_quality.get("strict_quality", {}))
    validation = build.get("validation", {})
    target = build.get("train_target_stats", {})
    train_topup = build.get("train_topup_stats", {})
    train_sentiment_constraints = build.get("train_sentiment_constraints", {})
    explicit_metrics = build.get("explicit_metrics", {})
    prev_build = previous or {}
    prev_quality = prev_build.get("output_quality", {})
    prev_rows = prev_build.get("row_counts", {})

    train_export = int(row_counts.get("train_export", 0))
    train_target_min = int(target.get("target_min_rows", 0))
    train_target_max = int(target.get("target_max_rows", 0))
    size_shortfall = int(target.get("size_shortfall_rows", 0))
    train_leakage = float(build.get("train_domain_leakage_row_rate", 1.0))
    train_general = float(build.get("train_general_dominance_rate", 1.0))

    blocking_reasons = list(build.get("blocking_reasons") or [])
    if not blocking_reasons and bool(validation.get("train_target_blocking_failure")):
        blocking_reasons.append({"code": "TRAIN_SIZE_BELOW_TARGET", "message": "Train export below target minimum."})

    usable_for_training = (
        train_export > 0
        and train_leakage == 0.0
        and train_general == 0.0
    )
    research_ready = (
        usable_for_training
        and bool(target.get("size_within_target_range", False))
        and not bool(validation.get("train_target_blocking_failure"))
        and not bool(validation.get("sampled_run_blocked_or_debug"))
    )
    publication_ready = bool(research_ready and build.get("gold_eval", {}).get("has_gold_labels", False))
    score, status = _score_from_verdict(
        usable_for_training=usable_for_training,
        research_ready=research_ready,
        publication_ready=publication_ready,
    )

    improvements: list[str] = []
    if train_leakage == 0.0:
        improvements.append("Training leakage barrier is intact (`train_domain_leakage_row_rate=0.0`).")
    if train_general == 0.0:
        improvements.append("General aspect contamination is excluded from train export (`train_general_dominance_rate=0.0`).")
    if float(build.get("grounded_prediction_rate", 0.0)) >= 0.99:
        improvements.append("Grounding remains strong (`grounded_prediction_rate` is near 1.0).")
    if int(train_topup.get("topup_rows_added", 0)) > 0:
        improvements.append(f"Strict top-up recovered {int(train_topup.get('topup_rows_added', 0))} rows without relaxing safety constraints.")

    failures: list[str] = []
    if bool(validation.get("train_target_blocking_failure")):
        failures.append(
            f"Research run blocked on train size (`train_export={train_export}`, target range `{train_target_min}-{train_target_max}`)."
        )
    if float(output_quality.get("fallback_only_rate", 0.0)) > 0.25:
        failures.append(f"Fallback-only rate remains high at {float(output_quality.get('fallback_only_rate', 0.0)):.4f}.")
    if float(strict_quality.get("explicit_in_implicit_rate", 0.0)) > 0.0:
        failures.append("Strict implicit export has explicit contamination.")
    if int(strict_quality.get("boundary_false_positive_count", 0)) > 0:
        failures.append("Strict implicit export has boundary false positives.")
    if float(build.get("train_positive_ratio", 0.0)) > float(build.get("config", {}).get("train_max_positive_ratio", 0.5)):
        failures.append("Positive sentiment remains above configured maximum in train export.")

    root_causes: list[str] = []
    sampled = bool(build.get("config", {}).get("sample_size") is not None or build.get("config", {}).get("chunk_size") is not None)
    root_causes.append("Run used full corpus (no sampling/chunking)." if not sampled else "Run used sampled/chunked execution.")
    if size_shortfall > 0:
        root_causes.append(
            "Strict filters (review gating + leakage barrier + non-general enforcement + sentiment bounds) reduced train rows below target minimum."
        )
    if float(output_quality.get("fallback_only_rate", 0.0)) > 0.2:
        root_causes.append("Implicit generation quality remains bottlenecked by fallback-heavy rows in source corpus.")

    deltas = {
        "selected_delta": _delta(int(row_counts.get("selected", 0)), int(prev_rows.get("selected", 0)) if prev_rows else None),
        "train_export_delta": _delta(train_export, int(prev_rows.get("train_export", 0)) if prev_rows else None),
        "fallback_only_rate_delta": _delta(
            float(output_quality.get("fallback_only_rate", 0.0)),
            float(prev_quality.get("fallback_only_rate", 0.0)) if prev_quality else None,
        ),
        "domain_leakage_row_rate_delta": _delta(
            float(output_quality.get("domain_leakage_row_rate", 0.0)),
            float(prev_quality.get("domain_leakage_row_rate", 0.0)) if prev_quality else None,
        ),
    }

    return {
        "generated_at": utc_now_iso(),
        "dataset_version": str(build.get("output_version", "unknown")),
        "pipeline_version": str(build.get("pipeline_version", "unknown")),
        "run_profile": build.get("run_profile"),
        "overall_assessment": {
            "quality_score_out_of_10": score,
            "status": status,
            "usable_for_training": usable_for_training,
            "research_ready": research_ready,
            "publication_ready": publication_ready,
            "blocking_reasons": blocking_reasons,
        },
        "major_improvements": improvements,
        "critical_failures": failures,
        "root_cause_attribution": root_causes,
        "quality_deltas_vs_previous": deltas,
        "dataset_overview": {
            "size_and_coverage": {
                "input_rows": int(row_counts.get("input", 0)),
                "processed_rows": int(row_counts.get("preprocessed", 0)),
                "selected_rows": int(row_counts.get("selected", 0)),
                "train_rows": int(row_counts.get("train", 0)),
                "val_rows": int(row_counts.get("val", 0)),
                "test_rows": int(row_counts.get("test", 0)),
                "final_train_export_rows": train_export,
                "target_train_min_rows": train_target_min,
                "target_train_max_rows": train_target_max,
                "size_shortfall_rows": size_shortfall,
            },
            "domain_distribution": quality.get("output_quality", {}).get("top_implicit_aspects_by_domain", {}),
            "language_distribution": build.get("language_distribution", {}),
        },
        "label_and_annotation_quality": {
            "aspect_distribution": {
                "top_aspects": output_quality.get("top_implicit_aspects", []),
                "entropy": explicit_metrics.get("aspect_distribution_entropy"),
                "top_share": explicit_metrics.get("aspect_top_share"),
            },
            "sentiment_distribution": {
                "train_negative_ratio": float(build.get("train_negative_ratio", 0.0)),
                "train_positive_ratio": float(build.get("train_positive_ratio", 0.0)),
                "train_neutral_ratio": float(train_sentiment_constraints.get("achieved", {}).get("neutral_ratio", 0.0)),
            },
            "span_grounding": {
                "grounded_prediction_rate": float(build.get("grounded_prediction_rate", 0.0)),
                "span_support": output_quality.get("span_support", {}),
                "ungrounded_non_general_count": int(build.get("ungrounded_non_general_count", 0)),
            },
        },
        "core_quality_metrics": {
            "domain_leakage": {
                "corpus_domain_leakage_rows": int(output_quality.get("domain_leakage_rows", 0)),
                "corpus_domain_leakage_row_rate": float(output_quality.get("domain_leakage_row_rate", 0.0)),
                "train_domain_leakage_rows": int(build.get("train_domain_leakage_rows", 0)),
                "train_domain_leakage_row_rate": train_leakage,
                "eval_domain_leakage_row_rate": float(build.get("eval_domain_leakage_row_rate", 0.0)),
            },
            "fallback_usage": {
                "fallback_only_rows": int(output_quality.get("fallback_only_rows", 0)),
                "fallback_only_rate": float(output_quality.get("fallback_only_rate", 0.0)),
            },
            "needs_review": {
                "needs_review_rows": int(output_quality.get("needs_review_rows", 0)),
                "review_reason_counts": output_quality.get("review_reason_counts", {}),
            },
            "invalid_aspects": {
                "generic_implicit_aspects": int(output_quality.get("generic_implicit_aspects", 0)),
                "rejected_implicit_aspects": int(output_quality.get("rejected_implicit_aspects", 0)),
            },
            "strict_quality": strict_quality,
        },
        "training_set_integrity": {
            "post_filter_quality": {
                "train_general_dominance_rate": train_general,
                "train_domain_leakage_row_rate": train_leakage,
                "train_grounded_rate": float(build.get("grounded_prediction_rate", 0.0)),
            },
            "filtering_effects": {
                "train_review_filter": build.get("train_review_filter_applied", {}),
                "train_salvage_stats": build.get("train_salvage_stats", {}),
                "train_topup_stats": train_topup,
            },
            "sentiment_constraints": train_sentiment_constraints,
        },
        "pipeline_behavior": {
            "conditioning_strategy": {
                "train_domain_conditioning_mode": build.get("train_domain_conditioning_mode"),
                "eval_domain_conditioning_mode": build.get("eval_domain_conditioning_mode"),
            },
            "fallback_strategy": {
                "llm_fallback_enabled": bool(build.get("config", {}).get("enable_llm_fallback", False)),
                "llm_fallback_threshold": float(build.get("config", {}).get("llm_fallback_threshold", 0.0)),
            },
            "salvage_strategy": build.get("train_salvage_stats", {}),
            "topup_strategy": train_topup,
        },
        "failure_analysis": {
            "structural_failures": [item for item in failures if "blocked" in item or "leakage" in item.lower()],
            "signal_failures": [item for item in failures if item not in [f for f in failures if "blocked" in f or "leakage" in f.lower()]],
            "pipeline_misconfiguration": {
                "sample_size": build.get("config", {}).get("sample_size"),
                "chunk_size": build.get("config", {}).get("chunk_size"),
                "run_profile": build.get("run_profile"),
            },
        },
        "verdict": {
            "usable_for_training": usable_for_training,
            "research_ready": research_ready,
            "publication_ready": publication_ready,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    oa = report["overall_assessment"]
    overview = report["dataset_overview"]["size_and_coverage"]
    quality = report["core_quality_metrics"]
    integrity = report["training_set_integrity"]
    verdict = report["verdict"]
    lines: list[str] = []
    lines.append("# DATASET QUALITY REPORT (ABSA PIPELINE)")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(f"- Dataset Version: {report.get('dataset_version')}")
    lines.append(f"- Pipeline Version: {report.get('pipeline_version')}")
    lines.append(f"- Date: {report.get('generated_at')}")
    lines.append(f"- Quality Score: {oa.get('quality_score_out_of_10')} / 10")
    lines.append(f"- Status: {oa.get('status')}")
    lines.append("- Key Findings:")
    for item in report.get("major_improvements", [])[:3]:
        lines.append(f"  - {item}")
    for item in report.get("critical_failures", [])[:3]:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("## 2. Dataset Overview")
    lines.append(f"- Input rows: {overview.get('input_rows')}")
    lines.append(f"- Processed rows: {overview.get('processed_rows')}")
    lines.append(f"- Selected rows: {overview.get('selected_rows')}")
    lines.append(f"- Train / Val / Test: {overview.get('train_rows')} / {overview.get('val_rows')} / {overview.get('test_rows')}")
    lines.append(f"- Final train export size: {overview.get('final_train_export_rows')}")
    lines.append(f"- Target train size: {overview.get('target_train_min_rows')} - {overview.get('target_train_max_rows')}")
    lines.append(f"- Shortfall / Overshoot: {overview.get('size_shortfall_rows')}")
    lines.append("")
    lines.append("## 3. Core Quality Metrics (Critical)")
    lines.append(f"- Train domain leakage rate: {quality['domain_leakage']['train_domain_leakage_row_rate']}")
    lines.append(f"- Train general dominance rate: {integrity['post_filter_quality']['train_general_dominance_rate']}")
    lines.append(f"- Corpus fallback-only rate: {quality['fallback_usage']['fallback_only_rate']}")
    lines.append(f"- Needs review rows: {quality['needs_review']['needs_review_rows']}")
    lines.append("")
    lines.append("## 4. Training Set Integrity")
    lines.append(f"- Train leakage rows: {quality['domain_leakage']['train_domain_leakage_rows']}")
    lines.append(f"- Train top-up rows added: {integrity['filtering_effects']['train_topup_stats'].get('topup_rows_added', 0)}")
    lines.append(f"- Sentiment constraints achieved: {integrity['sentiment_constraints'].get('achieved', {})}")
    lines.append("")
    lines.append("## 5. Failure Analysis")
    for item in report.get("root_cause_attribution", []):
        lines.append(f"- {item}")
    if oa.get("blocking_reasons"):
        lines.append("- Blocking reasons:")
        for reason in oa["blocking_reasons"]:
            lines.append(f"  - {reason.get('code')}: {reason.get('message')}")
    lines.append("")
    lines.append("## 6. Final Verdict")
    lines.append(f"- usable_for_training: {verdict.get('usable_for_training')}")
    lines.append(f"- research_ready: {verdict.get('research_ready')}")
    lines.append(f"- publication_ready: {verdict.get('publication_ready')}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deep analysis scorecard from dataset_builder reports.")
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--data-quality-report", type=Path, required=True)
    parser.add_argument("--previous-build-report", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build = _load_json(args.build_report)
    quality = _load_json(args.data_quality_report)

    previous_path = args.previous_build_report or (args.build_report.parent / "build_report.previous.json")
    previous = _load_json(previous_path) if previous_path.exists() else None
    scorecard = _build_scorecard(build, quality, previous)

    out_json = args.out_json or (args.build_report.parent / "deep_analysis_report.json")
    out_md = args.out_md or (args.build_report.parent / "deep_analysis_report.md")
    write_json(out_json, scorecard)
    out_md.write_text(_render_markdown(scorecard), encoding="utf-8")

    previous_path.write_text(json.dumps(build, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Deep analysis written: {out_json}")
    print(f"Deep analysis written: {out_md}")
    print(f"Previous snapshot updated: {previous_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
