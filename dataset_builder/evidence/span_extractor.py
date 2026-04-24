from __future__ import annotations


def extract_span_from_sentence(review_text: str, evidence_text: str) -> list[int]:
    start = str(review_text).lower().find(str(evidence_text).lower())
    if start < 0:
        return [-1, -1]
    return [start, start + len(evidence_text)]


def extract_span_for_aspect(review_text: str, aspect: str) -> list[int]:
    start = str(review_text).lower().find(str(aspect).lower())
    if start < 0:
        return [-1, -1]
    return [start, start + len(aspect)]
