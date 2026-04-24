from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpanValidation:
    valid: bool
    reason_codes: list[str]
    repair_severity: int = 0


def validate_span(review_text: str, evidence_text: str, evidence_span: list[int] | tuple[int, int] | None) -> SpanValidation:
    if not isinstance(evidence_span, (list, tuple)) or len(evidence_span) != 2:
        return SpanValidation(False, ["invalid_shape"], 2)
    try:
        start = int(evidence_span[0])
        end = int(evidence_span[1])
    except (TypeError, ValueError):
        return SpanValidation(False, ["invalid_offset"], 2)
    if start < 0 or end <= start or end > len(review_text):
        return SpanValidation(False, ["out_of_range"], 2)
    extracted = review_text[start:end]
    if evidence_text and extracted.lower() != evidence_text.lower():
        return SpanValidation(False, ["text_mismatch"], 1)
    if not extracted.strip():
        return SpanValidation(False, ["empty_span"], 2)
    return SpanValidation(True, [], 0)
