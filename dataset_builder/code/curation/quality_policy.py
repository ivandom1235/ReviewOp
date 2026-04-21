from __future__ import annotations

from typing import Any


_QUALITY_REJECT_REASONS = {
    "domain_leakage",
    "explicit_contamination",
    "invalid_aspect",
    "fallback_general",
}

_QUALITY_BORDERLINE_REASONS = {
    "domain_soft_mismatch",
    "low_confidence",
    "weak_support",
    "general_only",
}


def quality_reason_codes(
    row: dict[str, Any],
    *,
    min_confidence: float,
    accepted_support_types: tuple[str, ...],
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
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
    if spans and any(str(span.get("support_type") or "") not in accepted_support_types for span in spans):
        reasons.append("unsupported_support_type")

    aspect_conf = implicit.get("aspect_confidence", {}) or {}
    confidences = [float(value) for value in aspect_conf.values() if value is not None]
    if not confidences:
        confidences = [float(span.get("confidence", 0.0)) for span in spans if span.get("confidence") is not None]
    if confidences and max(confidences) < float(min_confidence):
        reasons.append("low_confidence")

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
            accepted_support_types={str(value).strip() for value in accepted_support_types if str(value).strip()} or {"exact", "near_exact", "gold"},
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

    if any(not _is_valid_latent_aspect(aspect) for aspect in aspects):
        reasons.append("invalid_aspect")

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
) -> dict[str, Any]:
    reason_codes = quality_reason_codes(
        row,
        min_confidence=min_confidence,
        accepted_support_types=accepted_support_types,
        candidate_aspects_by_domain=candidate_aspects_by_domain,
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
    if str(implicit.get("review_reason") or "") in {"weak_support", "low_confidence", "domain_soft_mismatch"}:
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
    return False, []


def _is_valid_latent_aspect(aspect: str) -> bool:
    return bool(str(aspect).strip())

