from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .config import ProtonetConfig
    from .encoder import HybridTextEncoder, format_input_text
    from .projection_head import ProjectionHead
except ImportError:
    from config import ProtonetConfig
    from encoder import HybridTextEncoder, format_input_text
    from projection_head import ProjectionHead


@dataclass
class EpisodeForwardOutput:
    logits: torch.Tensor
    targets: torch.Tensor
    ordered_labels: List[str]
    predictions: List[str]
    probabilities: torch.Tensor


def _item_joint_label(item: Dict[str, Any], separator: str) -> str:
    if item.get("joint_label"):
        return str(item["joint_label"])
    aspect = str(item.get("aspect") or item.get("implicit_aspect") or "unknown").strip()
    sentiment = str(item.get("sentiment") or "neutral").strip().lower() or "neutral"
    return f"{aspect}{separator}{sentiment}"


class ProtoNetModel(nn.Module):
    def __init__(self, cfg: ProtonetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = HybridTextEncoder(cfg)
        if cfg.production_require_transformer and self.encoder.backend != "transformer":
            raise RuntimeError("Transformer encoder is required for this run, but no transformer backend was loaded.")
        self.projection = ProjectionHead(self.encoder.hidden_size, cfg.projection_dim, cfg.dropout)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(cfg.temperature_init), dtype=torch.float32)))
        self.to(cfg.device)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.05, max=10.0)

    def encode_items(self, items: List[Dict[str, Any]]) -> torch.Tensor:
        texts = [
            format_input_text(
                str(item.get("review_text", "")),
                str(item.get("evidence_sentence") or item.get("review_text", "")),
                str(item.get("domain", "unknown")),
            )
            for item in items
        ]
        embeddings = self.encoder(texts)
        projection_dtype = next(self.projection.parameters()).dtype
        embeddings = embeddings.to(device=self.cfg.device, dtype=projection_dtype)
        return self.projection(embeddings)

    def episode_forward(self, episode: Dict[str, Any]) -> EpisodeForwardOutput:
        support = list(episode.get("support_set", []))
        query = list(episode.get("query_set", []))
        support_embeddings = self.encode_items(support)
        query_embeddings = self.encode_items(query)

        support_labels = [_item_joint_label(item, self.cfg.joint_label_separator) for item in support]
        query_labels = [_item_joint_label(item, self.cfg.joint_label_separator) for item in query]
        ordered_labels = sorted(dict.fromkeys(support_labels))

        prototype_rows = []
        global_mean = support_embeddings.mean(dim=0)
        for label in ordered_labels:
            mask = torch.tensor([lab == label for lab in support_labels], device=self.cfg.device, dtype=torch.bool)
            label_embeddings = support_embeddings[mask]
            label_items = [item for item, lab in zip(support, support_labels) if lab == label]
            weights = torch.tensor(
                [float(item.get("confidence", 1.0)) for item in label_items],
                dtype=torch.float32,
                device=self.cfg.device,
            )
            weights = weights / weights.sum().clamp(min=1e-6)
            prototype = (label_embeddings * weights.unsqueeze(-1)).sum(dim=0)
            prototype = F.normalize(
                (1.0 - self.cfg.prototype_smoothing) * prototype + self.cfg.prototype_smoothing * global_mean,
                p=2,
                dim=-1,
            )
            prototype_rows.append(prototype)
        prototypes = torch.stack(prototype_rows, dim=0)

        logits = -torch.cdist(query_embeddings, prototypes, p=2).pow(2) / self.temperature
        label_to_index = {label: idx for idx, label in enumerate(ordered_labels)}
        targets = torch.tensor([label_to_index[label] for label in query_labels], dtype=torch.long, device=self.cfg.device)
        probabilities = torch.softmax(logits, dim=-1)
        prediction_indices = probabilities.argmax(dim=-1).tolist()
        predictions = [ordered_labels[idx] for idx in prediction_indices]
        return EpisodeForwardOutput(
            logits=logits,
            targets=targets,
            ordered_labels=ordered_labels,
            predictions=predictions,
            probabilities=probabilities.detach().cpu(),
        )

    def export_description(self) -> Dict[str, object]:
        return {
            "encoder": self.encoder.export_info(),
            "projection_dim": self.cfg.projection_dim,
            "temperature": float(self.temperature.detach().cpu().item()),
        }
