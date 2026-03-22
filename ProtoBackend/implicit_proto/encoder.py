from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: sentence_transformers. "
        "Use the project environment: "
        "`backend\\venv\\Scripts\\python.exe ProtoBackend\\proto_cli.py ...` "
        "or install dependencies from `backend/requirements.txt`."
    ) from exc


class PrototypeEncoder:
    """Sentence encoder wrapper for prototype-based classification."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.device = resolved_device
        self.local_files_only = local_files_only
        try:
            self.model = SentenceTransformer(
                model_name,
                device=resolved_device,
                local_files_only=local_files_only,
            )
        except TypeError:
            self.model = SentenceTransformer(model_name, device=resolved_device)

    def encode(self, sentences: Iterable[str], batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray:
        sentence_list: List[str] = [s.strip() for s in sentences if s and s.strip()]
        if not sentence_list:
            raise ValueError("No sentences provided for encoding")

        embeddings = self.model.encode(
            sentence_list,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def save(self, output_dir: str | Path) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(out_dir))
        return out_dir
