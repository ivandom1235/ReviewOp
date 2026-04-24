from __future__ import annotations

from .span_validator import validate_span


def score_evidence_quality(review_text: str, evidence_text: str, evidence_span: list[int]) -> float:
    validation = validate_span(review_text, evidence_text, evidence_span)
    if validation.valid:
        return 1.0
    if validation.repair_severity == 1:
        return 0.5
    return 0.0


def span_alignment_score(review_text: str, evidence_text: str, evidence_span: list[int]) -> float:
    return score_evidence_quality(review_text, evidence_text, evidence_span)
