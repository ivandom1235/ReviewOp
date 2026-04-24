from __future__ import annotations


def assign_hardness_tier(ambiguity_score: float, novelty_status: str) -> str:
    if novelty_status != "known":
        return "H3"
    if ambiguity_score >= 0.75:
        return "H2"
    if ambiguity_score > 0:
        return "H1"
    return "H0"
