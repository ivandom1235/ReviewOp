from __future__ import annotations

from typing import Any


def build_blocking_reasons(
    *,
    report: dict[str, Any],
    run_profile: str,
    sampled_run: bool,
    train_topup_recovery_mode: str,
) -> list[dict[str, str]]:
    validation = report.get("validation", {})
    blocking_reasons: list[dict[str, str]] = list(report.get("blocking_reasons", []))

    if bool(validation.get("sampled_run_blocked_or_debug")) and run_profile == "debug":
        blocking_reasons.append({"code": "DEBUG_OR_SAMPLED_RUN", "message": "Run used debug profile and is not promotable."})
    if bool(validation.get("train_target_blocking_failure")):
        topup_shortfall = int(report.get("size_recovery_shortfall_remaining", 0))
        if str(train_topup_recovery_mode).strip().lower() == "strict_topup" and topup_shortfall > 0:
            blocking_reasons.append({
                "code": "TRAIN_SIZE_BELOW_TARGET_AFTER_STAGED_TOPUP",
                "message": "Train export remains below minimum after staged strict top-up. See train_topup_rejection_breakdown.",
            })
        else:
            blocking_reasons.append({"code": "TRAIN_SIZE_BELOW_TARGET", "message": "Train export is below configured minimum target size."})
    if not bool(validation.get("train_domain_leakage_ok")):
        blocking_reasons.append({"code": "TRAIN_DOMAIN_LEAKAGE", "message": "Train export contains cross-domain aspect leakage."})
    if not bool(validation.get("train_general_excluded")):
        blocking_reasons.append({"code": "TRAIN_GENERAL_CONTAMINATION", "message": "Train export still includes general-only fallback rows."})
    if not bool(validation.get("no_generic_aspects")) or not bool(validation.get("no_rejected_aspects")):
        blocking_reasons.append({"code": "INVALID_ASPECT_LABELS", "message": "Generic or rejected aspect labels detected."})
    if not bool(validation.get("train_positive_ratio_within_max")):
        blocking_reasons.append({"code": "TRAIN_POSITIVE_RATIO_TOO_HIGH", "message": "Positive sentiment exceeds configured maximum ratio."})
    if not bool(validation.get("train_neutral_ratio_within_max")):
        blocking_reasons.append({"code": "TRAIN_NEUTRAL_RATIO_TOO_HIGH", "message": "Neutral sentiment exceeds configured maximum ratio."})
    if not bool(validation.get("strict_explicit_contamination_ok")):
        blocking_reasons.append({"code": "STRICT_EXPLICIT_CONTAMINATION", "message": "Strict implicit set contains explicit span contamination."})
    if not bool(validation.get("strict_boundary_fp_ok")):
        blocking_reasons.append({"code": "STRICT_BOUNDARY_FALSE_POSITIVES", "message": "Strict implicit set still includes boundary false positives."})
    if not bool(validation.get("strict_h2_h3_ok")):
        blocking_reasons.append({"code": "STRICT_HARDNESS_TOO_LOW", "message": "Strict implicit set does not meet H2/H3 minimum ratio."})
    if not bool(validation.get("strict_multi_aspect_ok")):
        blocking_reasons.append({"code": "STRICT_MULTI_ASPECT_TOO_LOW", "message": "Strict implicit set does not meet multi-aspect minimum ratio."})
    if not bool(validation.get("strict_challenge_ok")):
        blocking_reasons.append({"code": "STRICT_CHALLENGE_TOO_LOW", "message": "Strict challenge metric is below configured floor."})
    if not bool(validation.get("grouped_split_leakage_ok")):
        blocking_reasons.append({"code": "GROUPED_SPLIT_LEAKAGE", "message": "Grouped split leakage detected across benchmark splits."})
    if not bool(validation.get("benchmark_val_non_empty")):
        blocking_reasons.append({"code": "BENCHMARK_VAL_EMPTY", "message": "Benchmark validation split is empty."})
    if not bool(validation.get("benchmark_grounded_evidence_ok")):
        blocking_reasons.append({"code": "BENCHMARK_EVIDENCE_NOT_GROUNDED", "message": "Benchmark evidence grounding rate is below required threshold."})
    if not bool(validation.get("benchmark_duplicate_rate_ok")):
        blocking_reasons.append({"code": "BENCHMARK_DUPLICATE_INTERPRETATIONS", "message": "Benchmark still contains duplicate interpretations."})
    if not bool(validation.get("benchmark_thermal_share_ok")):
        blocking_reasons.append({"code": "BENCHMARK_THERMAL_OVERCONCENTRATION", "message": "Thermal aspect share remains over concentrated."})
    if not bool(validation.get("benchmark_domain_coverage_ok")):
        blocking_reasons.append({"code": "BENCHMARK_DOMAIN_COVERAGE", "message": "Benchmark is missing a core V1 domain family present in the source pool."})
    if not bool(validation.get("benchmark_family_floor_ok")):
        blocking_reasons.append({"code": "BENCHMARK_FAMILY_FLOOR", "message": "Debug benchmark family floor could not restore a missing core family."})
    if not bool(validation.get("benchmark_implicit_purity_ok")):
        blocking_reasons.append({"code": "BENCHMARK_IMPLICIT_PURITY", "message": "Benchmark implicit purity rate is below threshold."})
    if not bool(validation.get("benchmark_ontology_compatibility_ok")):
        blocking_reasons.append({"code": "BENCHMARK_ONTOLOGY_COMPATIBILITY", "message": "Benchmark ontology compatibility is below threshold."})
    if not bool(validation.get("sentiment_mismatch_rate_ok")):
        blocking_reasons.append({"code": "SENTIMENT_MISMATCH_RATE", "message": "Sentiment mismatch rate exceeds allowed maximum."})
    if not bool(validation.get("promotion_guard_ok")):
        blocking_reasons.append({"code": "WORST_DOMAIN_REGRESSION", "message": "Worst-domain F1 regressed above allowed threshold."})
    return blocking_reasons


def finalize_report(
    report: dict[str, Any],
    *,
    run_profile: str,
    sampled_run: bool,
    train_topup_recovery_mode: str,
) -> dict[str, Any]:
    finalized = dict(report)
    finalized["blocking_reasons"] = build_blocking_reasons(
        report=finalized,
        run_profile=run_profile,
        sampled_run=sampled_run,
        train_topup_recovery_mode=train_topup_recovery_mode,
    )
    return finalized
