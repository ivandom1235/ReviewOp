from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(output_dim, min(input_dim, output_dim * 2))
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.layers(inputs)
        return F.normalize(projected, p=2, dim=-1)
