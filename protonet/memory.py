from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ECProtoNetV2Config
from .encoder import TextEncoder, cosine_matrix
from .label_normalizer import LabelNormalizer


@dataclass
class MemoryEntry:
    aspect: str
    status: str
    trigger_patterns: list[str]
    quality: float = 1.0
    support_count: int = 1
    raw: dict = field(default_factory=dict)


@dataclass
class MemorySupportIndex:
    entries: list[MemoryEntry]
    trigger_texts: list[str]
    trigger_aspects: list[str]
    trigger_weights: np.ndarray
    trigger_matrix: np.ndarray

    def support_for(self, query_vector: np.ndarray, aspect: str, threshold: float) -> float:
        if self.trigger_matrix.size == 0:
            return 0.0
        aspect = aspect or ""
        indices = [i for i, a in enumerate(self.trigger_aspects) if a == aspect]
        if not indices:
            return 0.0
        mat = self.trigger_matrix[indices]
        sims = cosine_matrix(query_vector.reshape(1, -1), mat)[0]
        weights = self.trigger_weights[indices]
        best = float(np.max(sims * weights)) if sims.size else 0.0
        return best if best >= threshold else 0.0


def _extract_entries(raw_items: list[dict], normalizer: LabelNormalizer, config: ECProtoNetV2Config) -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []
    for item in raw_items:
        status = str(item.get("status") or item.get("validation_status") or "detected")
        if config.memory_min_status == "promoted" and status != "promoted":
            continue
        aspect = item.get("suggested_aspect") or item.get("aspect_raw") or item.get("aspect") or item.get("canonical")
        if not aspect:
            continue
        patterns = item.get("trigger_patterns") or item.get("evidence_texts") or []
        rep = item.get("representative_trigger")
        if rep:
            patterns = [rep, *patterns]
        patterns = [str(p) for p in patterns if p]
        if not patterns:
            continue
        quality = float(item.get("quality", item.get("consistency", 1.0)) or 1.0)
        support_count = int(item.get("support_count", max(1, len(patterns))) or 1)
        entries.append(
            MemoryEntry(
                aspect=normalizer.normalize(str(aspect)),
                status=status,
                trigger_patterns=patterns,
                quality=max(0.05, min(1.0, quality)),
                support_count=support_count,
                raw=item,
            )
        )
    return entries


def build_memory_index(raw_items: list[dict], normalizer: LabelNormalizer, encoder: TextEncoder, config: ECProtoNetV2Config) -> MemorySupportIndex:
    entries = _extract_entries(raw_items, normalizer, config)
    trigger_texts: list[str] = []
    trigger_aspects: list[str] = []
    weights: list[float] = []
    for e in entries:
        for p in e.trigger_patterns:
            trigger_texts.append(p)
            trigger_aspects.append(e.aspect)
            weights.append(e.quality)
    matrix = encoder.encode(trigger_texts) if trigger_texts else np.zeros((0, 0), dtype=np.float32)
    return MemorySupportIndex(
        entries=entries,
        trigger_texts=trigger_texts,
        trigger_aspects=trigger_aspects,
        trigger_weights=np.array(weights, dtype=np.float32),
        trigger_matrix=matrix,
    )
