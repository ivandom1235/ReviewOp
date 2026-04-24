from __future__ import annotations

from ..schemas.interpretation import Interpretation


def compute_ambiguity_score(items: list[Interpretation]) -> float:
    if not items:
        return 0.0
    canonicals = {item.aspect_canonical for item in items if item.aspect_canonical}
    source_types = {item.source_type for item in items if item.source_type}
    sentiments = {item.sentiment for item in items if item.sentiment and item.sentiment != "unknown"}
    score = 0.0
    score += min(0.35, max(0, len(canonicals) - 1) * 0.18)
    score += 0.2 if len(source_types) > 1 else 0.0
    score += 0.2 if len(sentiments) > 1 else 0.0
    score += 0.15 if any(item.aspect_canonical == "unknown" for item in items) else 0.0
    score += 0.1 if any(item.canonical_confidence < 0.5 for item in items) else 0.0
    return min(1.0, score)


def count_gold_interpretations(items: list[Interpretation]) -> int:
    return len(items)
