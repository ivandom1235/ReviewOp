from __future__ import annotations

from dataclasses import dataclass

from .text_features import evidence_support


@dataclass(frozen=True)
class OpenWorldName:
    aspect: str
    confidence: float
    reason: str


DEFAULT_ALIAS_MAP = {
    "food_quality": ["food", "meal", "dish", "taste", "flavor", "pizza", "sushi", "pasta", "sandwich"],
    "service_quality": ["service", "staff", "waiter", "waitress", "server", "attentive", "rude", "friendly"],
    "ambience": ["atmosphere", "ambience", "vibe", "decor", "setting", "place", "quiet", "loud", "noisy"],
    "value": ["price", "prices", "cost", "cheap", "worth", "deal", "low"],
    "availability": ["available", "reservation", "stock", "sold out"],
    "cleanliness": ["clean", "dirty", "smell", "hygiene"],
    "battery_life": ["battery", "charge", "charging", "charger", "lasted", "drain"],
    "connectivity": ["network", "wifi", "signal", "connection", "calls dropped"],
    "reliability": ["crash", "failed", "failure", "broke", "disconnect"],
    "service_speed": ["wait", "waiting", "minutes", "slow service", "delay"],
}


def infer_named_open_world_aspect(
    text: str,
    *,
    alias_map: dict[str, list[str]] | None = None,
    min_confidence: float = 0.35,
) -> OpenWorldName | None:
    aliases = alias_map or DEFAULT_ALIAS_MAP
    scores: list[tuple[str, float]] = []
    for aspect, terms in aliases.items():
        vals = [evidence_support(text, term, matched_terms=None) for term in terms]
        score = max(vals) if vals else 0.0
        if score > 0:
            scores.append((aspect, float(score)))
    if not scores:
        return None
    aspect, score = max(scores, key=lambda x: x[1])
    if score < min_confidence:
        return None
    return OpenWorldName(aspect=aspect, confidence=score, reason="alias_named_open_world")
