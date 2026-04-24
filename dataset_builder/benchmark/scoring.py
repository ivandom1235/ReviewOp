from __future__ import annotations


def score_row(evidence_quality: float, novelty_bonus: float = 0.0, ambiguity_bonus: float = 0.0, repair_penalty: float = 0.0) -> float:
    return max(0.0, min(1.0, evidence_quality + novelty_bonus + ambiguity_bonus - repair_penalty))


def evidence_penalty(repair_severity: int) -> float:
    return min(0.5, max(0, repair_severity) * 0.2)
