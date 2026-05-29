from __future__ import annotations

from dataclasses import dataclass
import re


GENERIC_PARENTS = {
    "food": "food_quality",
    "meal": "food_quality",
    "dish": "food_quality",
    "taste": "food_quality",
    "pasta": "food_quality",
    "atmosphere": "ambience",
    "ambience": "ambience",
    "noisy": "ambience",
    "place": "ambience",
    "staff": "service_quality",
    "waiter": "service_quality",
    "server": "service_quality",
    "wait": "service_speed",
    "waiting": "service_speed",
    "minutes": "service_speed",
    "price": "value",
    "worth": "value",
    "cost": "value",
}


@dataclass(frozen=True)
class OpenWorldName:
    label: str
    confidence: float
    source: str


def propose_open_world_label(text: str) -> OpenWorldName:
    low = (text or "").lower()
    hits: list[str] = []
    for cue, label in GENERIC_PARENTS.items():
        if re.search(rf"\b{re.escape(cue)}\b", low):
            hits.append(label)
    if hits:
        label = max(set(hits), key=hits.count)
        return OpenWorldName(
            label=label,
            confidence=min(1.0, 0.55 + 0.10 * hits.count(label)),
            source="lexical_parent",
        )
    return OpenWorldName(label="quality", confidence=0.35, source="generic_fallback")
