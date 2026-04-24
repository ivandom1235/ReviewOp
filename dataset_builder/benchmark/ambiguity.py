from __future__ import annotations

from ..schemas.interpretation import Interpretation


def compute_ambiguity_score(items: list[Interpretation]) -> float:
    return min(1.0, max(0, len({item.aspect_canonical for item in items}) - 1) * 0.25)


def count_gold_interpretations(items: list[Interpretation]) -> int:
    return len(items)
