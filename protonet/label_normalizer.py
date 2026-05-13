from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .io_utils import read_json

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9_\s-]")


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
    return aliases.get(label, label or "unknown")


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
