from __future__ import annotations


def mark_abstain_acceptable(ambiguity_score: float, evidence_quality: float) -> bool:
    return ambiguity_score >= 0.5 or evidence_quality < 0.5
