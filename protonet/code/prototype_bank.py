from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

try:
    from .config import ProtonetConfig
    from .progress import task_bar
except ImportError:
    from config import ProtonetConfig
    from progress import task_bar


@dataclass
class PrototypeBank:
    labels: List[str]
    prototypes: torch.Tensor
    counts: Dict[str, int]
    mean_confidence: Dict[str, float]

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "prototypes": self.prototypes.detach().cpu(),
            "counts": self.counts,
            "mean_confidence": self.mean_confidence,
        }


def build_global_prototype_bank(model, episodes: List[Dict[str, Any]], cfg: ProtonetConfig) -> PrototypeBank:
    model.eval()
    unique_examples: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        for item in list(episode.get("support_set", [])) + list(episode.get("query_set", [])):
            key = str(item.get("example_id") or item.get("parent_review_id"))
            if key not in unique_examples:
                unique_examples[key] = dict(item)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in unique_examples.values():
        label = str(item.get("joint_label") or f"{item.get('aspect')}__{item.get('sentiment')}")
        grouped.setdefault(label, []).append(item)

    labels = sorted(grouped)
    prototype_rows: List[torch.Tensor] = []
    counts: Dict[str, int] = {}
    mean_confidence: Dict[str, float] = {}
    global_mean = None
    with torch.no_grad():
        embedding_cache: Dict[str, torch.Tensor] = {}
        with task_bar(total=len(labels), desc="prototype-bank", enabled=cfg.progress_enabled) as bar:
            for label in labels:
                items = grouped[label]
                embeddings = model.encode_items(items)
                embedding_cache[label] = embeddings
                raw_confidences = [float(item.get("confidence", 1.0)) for item in items]
                weights = torch.tensor(
                    raw_confidences,
                    dtype=torch.float32,
                    device=cfg.device,
                )
                weights = weights / weights.sum().clamp(min=1e-6)
                prototype_rows.append((embeddings * weights.unsqueeze(-1)).sum(dim=0))
                counts[label] = len(grouped[label])
                mean_confidence[label] = round(sum(raw_confidences) / max(1, len(raw_confidences)), 4)
                bar.update(1)
        raw_prototypes = torch.stack(prototype_rows, dim=0)
        global_mean = raw_prototypes.mean(dim=0)
        smoothed = torch.stack(
            [
                torch.nn.functional.normalize(
                    (1.0 - cfg.prototype_smoothing) * proto + cfg.prototype_smoothing * global_mean,
                    p=2,
                    dim=-1,
                )
                for proto in raw_prototypes
            ],
            dim=0,
        )
    return PrototypeBank(labels=labels, prototypes=smoothed, counts=counts, mean_confidence=mean_confidence)
