from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from .encoder import PrototypeEncoder
from .prototype_builder import PrototypeArtifacts, PrototypeBuilder


@dataclass(frozen=True)
class Prediction:
    aspect: str
    score: float


class ImplicitAspectDetector:
    """Prototype detector with max-centroid aggregation and optional per-label thresholds."""

    def __init__(self, encoder: PrototypeEncoder, artifacts: PrototypeArtifacts) -> None:
        self.encoder = encoder
        self.labels = artifacts.labels
        self.prototypes = artifacts.prototypes.astype(np.float32)
        self.centroid_labels = list(artifacts.centroid_labels or artifacts.labels)
        if len(self.centroid_labels) != self.prototypes.shape[0]:
            raise ValueError("Centroid labels and prototype vectors are misaligned")

    @classmethod
    def from_artifacts(
        cls,
        prototypes_path: str | Path,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | None = None,
    ) -> "ImplicitAspectDetector":
        encoder = PrototypeEncoder(model_name=model_name, device=device)
        artifacts = PrototypeBuilder.load(prototypes_path)
        return cls(encoder=encoder, artifacts=artifacts)

    def score_sentence(self, sentence: str) -> Dict[str, float]:
        emb = self.encoder.encode([sentence], batch_size=1, normalize_embeddings=True)[0]
        sims = self.prototypes @ emb
        best_scores: Dict[str, float] = {}
        for label, score in zip(self.centroid_labels, sims.tolist()):
            current = best_scores.get(label)
            if current is None or score > current:
                best_scores[label] = float(score)
        for label in self.labels:
            best_scores.setdefault(label, float("-inf"))
        return best_scores

    def predict_aspects(
        self,
        sentence: str,
        top_k: int = 3,
        threshold: float = 0.6,
        return_top1_if_empty: bool = False,
        label_thresholds: Mapping[str, float] | None = None,
    ) -> List[Prediction]:
        scored = self.score_sentence(sentence)
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        kept: List[Prediction] = []
        for aspect, score in ranked[: max(1, top_k)]:
            effective_threshold = float(label_thresholds.get(aspect, threshold)) if label_thresholds else float(threshold)
            if score >= effective_threshold:
                kept.append(Prediction(aspect=aspect, score=float(score)))

        if not kept and return_top1_if_empty and ranked:
            top_aspect, top_score = ranked[0]
            kept = [Prediction(aspect=top_aspect, score=float(top_score))]

        return kept

    def predict_aspect_dicts(
        self,
        sentence: str,
        top_k: int = 3,
        threshold: float = 0.6,
        return_top1_if_empty: bool = False,
        label_thresholds: Mapping[str, float] | None = None,
    ) -> List[Dict[str, float | str]]:
        return [
            {"aspect": p.aspect, "score": round(float(p.score), 6)}
            for p in self.predict_aspects(
                sentence=sentence,
                top_k=top_k,
                threshold=threshold,
                return_top1_if_empty=return_top1_if_empty,
                label_thresholds=label_thresholds,
            )
        ]
