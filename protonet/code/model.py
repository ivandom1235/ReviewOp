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
    support_embeddings: torch.Tensor
    query_embeddings: torch.Tensor


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

        # Vectorized prototype calculation
        num_support = support_embeddings.size(0)
        num_labels = len(ordered_labels)
        
        # Create a one-hot mask for each label: (num_labels, num_support)
        label_to_index = {label: idx for idx, label in enumerate(ordered_labels)}
        target_indices = torch.tensor([label_to_index[lab] for lab in support_labels], device=self.cfg.device)
        one_hot = F.one_hot(target_indices, num_classes=num_labels).float().T # (num_labels, num_support)
        
        # Apply confidence weights
        weights = torch.tensor(
            [float(item.get("confidence", 1.0)) for item in support],
            dtype=torch.float32,
            device=self.cfg.device,
        )
        weighted_one_hot = one_hot * weights.unsqueeze(0) # (num_labels, num_support)
        
        # Normalize weights per label
        weight_sums = weighted_one_hot.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = weighted_one_hot / weight_sums # (num_labels, num_support)
        
        # Compute prototypes: (num_labels, hidden)
        prototypes = torch.matmul(normalized_weights, support_embeddings)
        
        # Apply smoothing and global mean
        global_mean = support_embeddings.mean(dim=0)
        prototypes = (1.0 - self.cfg.prototype_smoothing) * prototypes + self.cfg.prototype_smoothing * global_mean
        prototypes = F.normalize(prototypes, p=2, dim=-1)

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
            support_embeddings=support_embeddings,
            query_embeddings=query_embeddings,
        )

    def export_description(self) -> Dict[str, object]:
        return {
            "encoder": self.encoder.export_info(),
            "projection_dim": self.cfg.projection_dim,
            "temperature": float(self.temperature.detach().cpu().item()),
        }
