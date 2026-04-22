from __future__ import annotations

from typing import Any

try:
    from extraction.aspect_registry import canonicalize_domain_aspect
except ImportError:  # pragma: no cover
    from ..extraction.aspect_registry import canonicalize_domain_aspect


TIER_1_SUPPORT_TYPES = {"exact", "near_exact", "gold"}
TIER_2_SUPPORT_TYPES = {"symptom_based", "paraphrastic"}
TIER_3_SUPPORT_TYPES = {"domain_consistent_weak", "vector_semantic", "llm_reasoning", "discovered"}

_QUALITY_REJECT_REASONS = {
    "domain_leakage",
    "explicit_contamination",
    "invalid_aspect",
    "fallback_general",
    "mapping_fail",
    "tier3_support_primary_disallowed",
}

_QUALITY_BORDERLINE_REASONS = {
    "domain_soft_mismatch",
    "low_confidence",
    "weak_support",
    "general_only",
    "unsupported_support_type",
    "low_mapping_confidence",
    "weak_evidence",
}


def _support_tier(support_type: str) -> int:
    support = str(support_type or "").strip().lower()
    if support in TIER_1_SUPPORT_TYPES:
        return 1
    if support in TIER_2_SUPPORT_TYPES:
        return 2
    if support in TIER_3_SUPPORT_TYPES:
        return 3
    return 99


def _max_mapping_confidence(row: dict[str, Any], spans: list[dict[str, Any]]) -> float:
    candidates: list[float] = []
    row_level = row.get("implicit", {}).get("canonical_mapping_confidence")
    if row_level is not None:
        try:
            candidates.append(float(row_level))
        except (TypeError, ValueError):
            pass
    for span in spans:
        value = span.get("mapping_confidence")
        if value is not None:
            try:
                candidates.append(float(value))
            except (TypeError, ValueError):
                continue
    return max(candidates) if candidates else 1.0


def quality_reason_codes(
    row: dict[str, Any],
    *,
    min_confidence: float,
    accepted_support_types: tuple[str, ...],
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
    policy_mode: str = "primary",
) -> list[str]:
    implicit = row.get("implicit", {}) or {}
    explicit = row.get("explicit", {}) or {}
    review_reason = str(implicit.get("review_reason") or "").strip()
    reasons: list[str] = []
    if review_reason and review_reason not in {"domain_leakage", "domain_soft_mismatch"}:
        reasons.append(review_reason)

    aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
    spans = list(implicit.get("spans") or [])
    if not aspects:
        reasons.append("general_only")
    if not spans:
        reasons.append("no_spans")
    weak_evidence = spans and all(
        _support_tier(str(span.get("support_type") or "")) > 1
        and not str(span.get("evidence_text") or span.get("clause") or "").strip()
        for span in spans
    )
    if weak_evidence:
        reasons.append("weak_evidence")

    accepted = {str(value).strip() for value in accepted_support_types if str(value).strip()}
    if spans and any(str(span.get("support_type") or "") not in accepted for span in spans):
        reasons.append("unsupported_support_type")

    aspect_conf = implicit.get("aspect_confidence", {}) or {}
    confidences = [float(value) for value in aspect_conf.values() if value is not None]
    if not confidences:
        confidences = [float(span.get("confidence", 0.0)) for span in spans if span.get("confidence") is not None]
    max_conf = max(confidences) if confidences else 0.0

    tiers = [_support_tier(str(span.get("support_type") or "")) for span in spans]
    has_tier2 = any(tier == 2 for tier in tiers)
    has_tier3 = any(tier == 3 for tier in tiers)
    if policy_mode == "primary" and has_tier3:
        reasons.append("tier3_support_primary_disallowed")
    if has_tier2 and max_conf < max(float(min_confidence) + 0.08, 0.68):
        reasons.append("low_confidence")
    elif confidences and max_conf < float(min_confidence):
        reasons.append("low_confidence")
    if has_tier3 and all(tier == 3 for tier in tiers):
        reasons.append("weak_support")

    mapping_conf = _max_mapping_confidence(row, spans)
    if has_tier2 and mapping_conf < 0.9:
        reasons.append("low_mapping_confidence")
    elif not has_tier2 and mapping_conf < 0.85:
        reasons.append("low_mapping_confidence")

    if review_reason == "implicit_not_ready" or not bool(implicit.get("implicit_ready", True)):
        reasons.append("implicit_not_ready")
    if review_reason == "fallback_general":
        reasons.append("fallback_general")
    if review_reason == "boundary_false_positive":
        reasons.append("boundary_false_positive")

    if candidate_aspects_by_domain is not None and not _row_domain_valid_for_train(row=row, candidate_aspects_by_domain=candidate_aspects_by_domain):
        if _row_domain_soft_mismatch(
            row,
            candidate_aspects_by_domain=candidate_aspects_by_domain,
            accepted_support_types=accepted or {"exact", "near_exact", "gold"},
            min_confidence=min_confidence,
        ):
            reasons.append("domain_soft_mismatch")
        else:
            reasons.append("domain_leakage")

    explicit_aspects = {
        str(aspect).strip().lower()
        for aspect in list(explicit.get("aspects") or [])
        if str(aspect).strip()
    }
    if explicit_aspects and any(str(aspect).strip().lower() in explicit_aspects for aspect in aspects):
        reasons.append("explicit_contamination")

    if any(not _is_valid_latent_aspect(aspect=aspect, domain=str(row.get("domain") or "")) for aspect in aspects):
        reasons.append("invalid_aspect")
    if any(_mapping_failed(aspect=aspect, domain=str(row.get("domain") or ""), spans=spans) for aspect in aspects):
        reasons.append("mapping_fail")

    return list(dict.fromkeys(reasons))


def quality_row_bucket(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return None
    if any(code in _QUALITY_REJECT_REASONS for code in reason_codes):
        return "rejected"
    if any(code in _QUALITY_BORDERLINE_REASONS for code in reason_codes):
        return "borderline"
    return "rejected"


def quality_decision_record(
    row: dict[str, Any],
    *,
    min_confidence: float,
    recovery_confidence_threshold: float,
    accepted_support_types: tuple[str, ...],
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
    allow_weak_support_in_recovery: bool = False,
    policy_mode: str = "primary",
) -> dict[str, Any]:
    reason_codes = quality_reason_codes(
        row,
        min_confidence=min_confidence,
        accepted_support_types=accepted_support_types,
        candidate_aspects_by_domain=candidate_aspects_by_domain,
        policy_mode=policy_mode,
    )
    bucket = quality_row_bucket(reason_codes)
    if bucket == "borderline":
        decision = "silver"
    elif bucket == "rejected":
        decision = "hard_reject"
    else:
        decision = "train_keep"
    eligible, eligible_reasons = _quality_recovery_eligible(
        row,
        min_confidence=min_confidence,
        recovery_confidence_threshold=recovery_confidence_threshold,
        accepted_support_types=accepted_support_types,
        candidate_aspects_by_domain=candidate_aspects_by_domain,
        allow_weak_support=allow_weak_support_in_recovery,
        allow_domain_soft_mismatch=True,
    )
    aspect_conf = row.get("implicit", {}).get("aspect_confidence", {}) or {}
    confidences = [float(value) for value in aspect_conf.values() if value is not None]
    if not confidences:
        confidences = [
            float(span.get("confidence", 0.0))
            for span in list(row.get("implicit", {}).get("spans") or [])
            if span.get("confidence") is not None
        ]
    quality_score = round(max(confidences) if confidences else 0.0, 4)
    usefulness_score = 0.0
    implicit = row.get("implicit", {}) or {}
    if len([aspect for aspect in implicit.get("aspects", []) if str(aspect) != "general"]) > 1:
        usefulness_score += 0.18
    if str(implicit.get("review_reason") or "") in {"weak_support", "low_confidence", "domain_soft_mismatch", "domain_mismatch"}:
        usefulness_score += 0.16
    if bool(row.get("abstain_acceptable", False)):
        usefulness_score += 0.2
    if bool(row.get("novel_acceptable", False)):
        usefulness_score += 0.2
    if str(row.get("domain") or "").strip().lower() in {"electronics", "restaurant", "telecom"}:
        usefulness_score += 0.08
    return {
        "row": dict(row),
        "decision": decision,
        "bucket": bucket or decision,
        "reason_codes": reason_codes,
        "recovery_eligible": bool(eligible),
        "recovery_reason_codes": eligible_reasons,
        "quality_score": quality_score,
        "usefulness_score": round(min(1.0, usefulness_score), 4),
    }


def quality_analysis_artifact(
    train_rows: list[dict[str, Any]],
    final_train_rows: list[dict[str, Any]],
    *,
    min_confidence: float,
    recovery_confidence_threshold: float,
    accepted_support_types: tuple[str, ...],
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
    allow_weak_support_in_recovery: bool = False,
) -> dict[str, Any]:
    final_ids = {str(row.get("id") or "") for row in final_train_rows}
    train_keep_rows: list[dict[str, Any]] = []
    train_keep_records: list[dict[str, Any]] = []
    silver_rows: list[dict[str, Any]] = []
    hard_reject_rows: list[dict[str, Any]] = []
    recoverable_rows: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    review_queue_rows: list[dict[str, Any]] = []

    for row in train_rows:
        record = quality_decision_record(
            row,
            min_confidence=min_confidence,
            recovery_confidence_threshold=recovery_confidence_threshold,
            accepted_support_types=accepted_support_types,
            candidate_aspects_by_domain=candidate_aspects_by_domain,
            allow_weak_support_in_recovery=allow_weak_support_in_recovery,
        )
        row_id = str(row.get("id") or "")
        if row_id in final_ids:
            final_record = dict(record)
            final_record["source_decision"] = final_record.get("decision")
            final_record["source_bucket"] = final_record.get("bucket")
            final_record["decision"] = "train_keep"
            final_record["bucket"] = "train_keep"
            train_keep_rows.append(dict(row))
            train_keep_records.append(final_record)
            decision_records.append(final_record)
            decision_counts["train_keep"] = decision_counts.get("train_keep", 0) + 1
            continue
        decision_counts[str(record["decision"])] = decision_counts.get(str(record["decision"]), 0) + 1
        for code in record["reason_codes"]:
            reason_counts[code] = reason_counts.get(code, 0) + 1
        review_queue_rows.append(record)
        decision_records.append(record)
        if record["decision"] == "silver":
            silver_rows.append(record)
            if record["recovery_eligible"]:
                recoverable_rows.append(record)
        else:
            hard_reject_rows.append(record)

    return {
        "train_rows": len(train_rows),
        "final_train_rows": len(final_train_rows),
        "excluded_rows": len(train_rows) - len(final_train_rows),
        "train_keep_count": len(train_keep_rows),
        "silver_count": len(silver_rows),
        "hard_reject_count": len(hard_reject_rows),
        "borderline_count": len(silver_rows),
        "recoverable_count": len(recoverable_rows),
        "rejected_count": len(hard_reject_rows),
        "reason_group_counts": dict(reason_counts),
        "implicit_rejection_reason_counts": dict(reason_counts),
        "decision_counts": dict(decision_counts),
        "decision_records": decision_records,
        "silver_rows": silver_rows,
        "borderline_rows": silver_rows,
        "train_keep_rows": train_keep_rows,
        "train_keep_records": train_keep_records,
        "recoverable_rows": recoverable_rows,
        "hard_reject_rows": hard_reject_rows,
        "rejected_rows": hard_reject_rows,
        "review_queue_rows": review_queue_rows,
        "summary": {
            "train_rows": len(train_rows),
            "final_train_rows": len(final_train_rows),
            "excluded_rows": len(train_rows) - len(final_train_rows),
            "train_keep_count": len(train_keep_rows),
            "silver_count": len(silver_rows),
            "hard_reject_count": len(hard_reject_rows),
            "borderline_count": len(silver_rows),
            "recoverable_count": len(recoverable_rows),
            "rejected_count": len(hard_reject_rows),
            "reason_group_counts": dict(reason_counts),
            "decision_counts": dict(decision_counts),
            "decision_record_count": len(decision_records),
        },
    }


def _row_domain_valid_for_train(*, row: dict[str, Any], candidate_aspects_by_domain: dict[str, list[str]]) -> bool:
    domain = str(row.get("domain") or "").strip().lower()
    if not domain:
        return True
    return bool(candidate_aspects_by_domain.get(domain))


def _row_domain_soft_mismatch(
    row: dict[str, Any],
    *,
    candidate_aspects_by_domain: dict[str, list[str]],
    accepted_support_types: set[str],
    min_confidence: float,
) -> bool:
    return _row_domain_valid_for_train(row=row, candidate_aspects_by_domain=candidate_aspects_by_domain)


def _quality_recovery_eligible(
    row: dict[str, Any],
    *,
    min_confidence: float,
    recovery_confidence_threshold: float,
    accepted_support_types: tuple[str, ...],
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
    allow_weak_support: bool = False,
    allow_domain_soft_mismatch: bool = True,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    implicit = row.get("implicit", {}) or {}
    spans = list(implicit.get("spans") or [])
    if not spans:
        return False, ["no_spans"]
    accepted = {str(value).strip() for value in accepted_support_types if str(value).strip()}
    if any(str(span.get("support_type") or "").strip() not in accepted for span in spans):
        return False, ["unsupported_support_type"]
    max_conf = max((float(span.get("confidence", 0.0) or 0.0) for span in spans), default=0.0)
    if max_conf < float(recovery_confidence_threshold):
        reasons.append("low_confidence")
    tiers = [_support_tier(str(span.get("support_type") or "")) for span in spans]
    has_tier3 = any(tier == 3 for tier in tiers)
    if has_tier3 and not allow_weak_support:
        reasons.append("weak_support_disallowed")
    mapping_conf = _max_mapping_confidence(row, spans)
    if has_tier3 and mapping_conf < 0.9:
        reasons.append("low_mapping_confidence")
    elif not has_tier3 and mapping_conf < 0.85:
        reasons.append("low_mapping_confidence")
    return (len(reasons) == 0), reasons


def _is_valid_latent_aspect(*, aspect: str, domain: str) -> bool:
    return bool(canonicalize_domain_aspect(domain=domain, aspect_label=aspect, surface_rationale_tag=aspect))


def _mapping_failed(*, aspect: str, domain: str, spans: list[dict[str, Any]]) -> bool:
    surface = ""
    for span in spans:
        latent = str(span.get("latent_label") or span.get("aspect") or "").strip().lower()
        if latent == str(aspect).strip().lower():
            surface = str(span.get("aspect") or span.get("surface_rationale_tag") or span.get("evidence_text") or "")
            break
    return canonicalize_domain_aspect(domain=domain, aspect_label=aspect, surface_rationale_tag=surface) is None
