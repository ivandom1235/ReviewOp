from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .io_utils import read_json

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9_\s-]")
_TIME_TOKEN_RE = re.compile(r"^\d{1,2}(:\d{2})?$")


_KEEP_CANONICAL = {
    "service",
    "quality",
    "food_quality",
    "value",
    "ambience",
    "performance",
    "price",
    "usability",
    "display",
    "battery_life",
    "keyboard",
    "power",
    "portability",
    "trackpad",
    "aesthetics",
    "service_speed",
    "customer_support",
    "delivery",
    "connectivity",
    "call_reliability",
}


def _collapse_phrase_fragment(label: str) -> str:
    # Keep stable high-signal canonical labels untouched.
    if label in _KEEP_CANONICAL:
        return label

    toks = [t for t in label.split("_") if t]
    joined = " ".join(toks)

    # Storage-related phrase fragments -> coarse storage family.
    if any(t in toks for t in ("hard", "drive", "disk", "dvd", "storage")):
        return "storage"

    # Food/drink phrase fragments -> food quality.
    if any(t in toks for t in ("pizza", "sandwich", "meal", "dinner", "lunch", "cuisine", "beer", "wine", "tofu", "potato", "pasta", "dumplings", "bagels", "dessert", "daiquiries")):
        return "food_quality"

    # Venue/location phrase fragments -> ambience.
    if any(t in toks for t in ("place", "spot", "restaurant", "atmosphere", "ambience", "crowded", "romantic")):
        return "ambience"

    # Deal/warranty/price phrase fragments -> value.
    if any(t in toks for t in ("deal", "value", "price", "warranty", "cost", "worth")):
        return "value"

    # Failure/problem phrase fragments -> reliability.
    if any(t in toks for t in ("problem", "problems", "malfunction", "crash", "failure", "failed")):
        return "reliability"

    # Very long singleton-style phrases are often noisy label fragments.
    if len(toks) >= 3 and any(t in joined for t in ("great", "excellent", "amazing", "perfect")):
        return "quality"

    # Overly specific phrase labels should not remain as stable classes.
    if len(toks) >= 4:
        return "quality"

    # Two/three-token sentiment-led phrase labels are often low-signal fragments.
    sentiment_prefix = {"very", "really", "great", "excellent", "amazing", "wonderful", "good", "bad"}
    if toks and toks[0] in sentiment_prefix and len(toks) >= 2:
        return "quality"

    return label


def normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    label = label.replace("-", "_").replace("/", "_")
    label = _NON_WORD_RE.sub(" ", label)
    label = _WS_RE.sub("_", label).strip("_")
    aliases = {
        "battery": "battery_life",
        "battery_backup": "battery_life",
        "network": "connectivity",
        "call_quality": "call_reliability",
        "customer_service": "customer_support",
        "support": "customer_support",
        "delivery_time": "delivery_speed",
        "shipping_time": "delivery_speed",
        "wifi": "connectivity",
    }
    label = aliases.get(label, label or "unknown")
    label = _collapse_phrase_fragment(label)
    toks = [t for t in label.split("_") if t]
    if not toks:
        return "unknown"

    # Remove obviously noisy class labels.
    noisy_words = {
        "name",
        "reservation",
        "minutes",
        "minute",
        "hours",
        "hour",
        "bartender",
        "cute",
    }
    if any(t in noisy_words for t in toks):
        return "unknown"
    if any(t.startswith("#") for t in toks):
        return "unknown"
    if any(_TIME_TOKEN_RE.match(t) for t in toks):
        return "unknown"
    if any(any(ch.isdigit() for ch in t) for t in toks):
        return "unknown"

    return label or "unknown"


@dataclass
class LabelNormalizer:
    alias_to_canonical: dict[str, str] = field(default_factory=dict)
    equivalence_classes: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> "LabelNormalizer":
        artifact_dir = Path(artifact_dir)
        alias_to_canonical: dict[str, str] = {}
        equivalence_classes: dict[str, set[str]] = {}

        for name in ["label_equivalence.json", "equivalence_map.json", "aspect_equivalence.json"]:
            data = read_json(artifact_dir / name, default=None)
            if not data:
                continue
            if isinstance(data, dict):
                # supports {canonical: [aliases]} or {alias: canonical}
                for k, v in data.items():
                    nk = normalize_label(k)
                    if isinstance(v, list):
                        members = {nk, *[normalize_label(x) for x in v]}
                        equivalence_classes[nk] = members
                        for m in members:
                            alias_to_canonical[m] = nk
                    elif isinstance(v, str):
                        alias_to_canonical[normalize_label(k)] = normalize_label(v)

        return cls(alias_to_canonical=alias_to_canonical, equivalence_classes=equivalence_classes)

    def normalize(self, label: str) -> str:
        n = normalize_label(label)
        return self.alias_to_canonical.get(n, n)

    def normalize_set(self, labels: Iterable[str]) -> set[str]:
        return {self.normalize(x) for x in labels if x}

    def relaxed_match(self, a: str, b: str) -> bool:
        na, nb = self.normalize(a), self.normalize(b)
        if na == nb:
            return True
        for members in self.equivalence_classes.values():
            if na in members and nb in members:
                return True
        return False

    def canonicalize_prediction_set(self, labels: Iterable[str]) -> set[str]:
        return self.normalize_set(labels)
