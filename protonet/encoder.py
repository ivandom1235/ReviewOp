from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class TextEncoder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashingTextEncoder(TextEncoder):
    """Dependency-light deterministic encoder.

    This is not the final accuracy encoder. It exists so the new pipeline can be
    tested without downloading models. For best results, use SentenceTransformersEncoder.
    """

    def __init__(self, dim: int = 2048, normalize: bool = True):
        self.dim = int(dim)
        self.normalize = bool(normalize)
        self._cache = {}

    def _token_hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % self.dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        return idx, sign

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        for tok in tokens:
            idx, sign = self._token_hash(tok)
            vec[idx] += sign
            # char ngrams help symptom phrases a bit
            if len(tok) >= 5:
                for i in range(len(tok) - 2):
                    idx2, sign2 = self._token_hash(tok[i : i + 3])
                    vec[idx2] += 0.25 * sign2
        if self.normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        key = tuple(texts)
        if key in self._cache:
            return self._cache[key]
        res = np.vstack([self._encode_one(t) for t in texts]).astype(np.float32)
        self._cache[key] = res
        return res


class SentenceTransformersEncoder(TextEncoder):
    def __init__(self, model_name: str, normalize: bool = True, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.batch_size = batch_size
        self._cache = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        key = tuple(texts)
        if key in self._cache:
            return self._cache[key]
        arr = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        res = arr.astype(np.float32)
        self._cache[key] = res
        return res


def build_encoder(kind: str, model_name: str, normalize: bool, hashing_dim: int, batch_size: int) -> TextEncoder:
    kind = (kind or "hashing").lower()
    if kind in {"hashing", "hash", "bow"}:
        return HashingTextEncoder(dim=hashing_dim, normalize=normalize)
    if kind in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformersEncoder(model_name=model_name, normalize=normalize, batch_size=batch_size)
    raise ValueError(f"Unknown encoder kind: {kind}")


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.matmul(a.astype(np.float32), b.astype(np.float32).T)


def safe_minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if math.isclose(mn, mx):
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)
