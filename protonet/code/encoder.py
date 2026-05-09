from __future__ import annotations

from typing import Sequence
import torch
from sentence_transformers import SentenceTransformer


class ECEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a list of texts into embeddings."""
        if not texts:
            return torch.empty(0)
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
