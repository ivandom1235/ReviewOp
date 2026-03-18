from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from .dataset import SentenceDataset
from .encoder import PrototypeEncoder


@dataclass
class PrototypeArtifacts:
    labels: List[str]
    prototypes: np.ndarray
    centroid_labels: List[str]
    centroid_sizes: List[int]
    label_counts: Dict[str, int]
    config: Dict[str, float | int | bool]


class PrototypeBuilder:
    """Builds class prototypes from normalized sentence embeddings."""

    def __init__(
        self,
        encoder: PrototypeEncoder,
        *,
        dedupe_sentences: bool = True,
        shrinkage_alpha: float = 4.0,
        min_examples_per_centroid: int = 6,
        max_centroids_per_label: int = 2,
        random_state: int = 42,
        low_support_single_centroid_threshold: int = 10,
        low_support_shrinkage_boost: float = 2.0,
    ) -> None:
        self.encoder = encoder
        self.dedupe_sentences = dedupe_sentences
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.min_examples_per_centroid = int(min_examples_per_centroid)
        self.max_centroids_per_label = int(max_centroids_per_label)
        self.random_state = int(random_state)
        self.low_support_single_centroid_threshold = int(low_support_single_centroid_threshold)
        self.low_support_shrinkage_boost = float(low_support_shrinkage_boost)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _make_centroids(self, embeddings: np.ndarray) -> tuple[np.ndarray, List[int]]:
        num_examples = int(embeddings.shape[0])
        if num_examples == 0:
            raise ValueError("Cannot build centroids from zero embeddings")
        if num_examples <= self.low_support_single_centroid_threshold:
            return embeddings.mean(axis=0, keepdims=True), [num_examples]
        if num_examples < self.min_examples_per_centroid or self.max_centroids_per_label <= 1:
            return embeddings.mean(axis=0, keepdims=True), [num_examples]

        max_supported = min(self.max_centroids_per_label, num_examples // self.min_examples_per_centroid)
        if max_supported <= 1:
            return embeddings.mean(axis=0, keepdims=True), [num_examples]

        model = KMeans(n_clusters=max_supported, random_state=self.random_state, n_init=10)
        assignments = model.fit_predict(embeddings)

        centroids: List[np.ndarray] = []
        sizes: List[int] = []
        for cluster_idx in range(max_supported):
            cluster_points = embeddings[assignments == cluster_idx]
            if len(cluster_points) == 0:
                continue
            centroids.append(cluster_points.mean(axis=0))
            sizes.append(int(len(cluster_points)))
        if not centroids:
            return embeddings.mean(axis=0, keepdims=True), [num_examples]
        return np.vstack(centroids), sizes

    def build(self, dataset: SentenceDataset, batch_size: int = 32) -> PrototypeArtifacts:
        grouped = dataset.group_by_aspect(dedupe_sentences=self.dedupe_sentences)
        labels = sorted(grouped.keys())
        if not labels:
            raise ValueError("No aspect labels found for prototype creation")

        all_sentences: List[str] = []
        all_sentence_counts: Dict[str, int] = {}
        per_label_sentences: Dict[str, List[str]] = {}
        for label in labels:
            sentences = grouped[label]
            per_label_sentences[label] = sentences
            all_sentences.extend(sentences)
            all_sentence_counts[label] = len(sentences)

        all_embeddings = self.encoder.encode(all_sentences, batch_size=batch_size, normalize_embeddings=True)
        global_mean = self._normalize(all_embeddings.mean(axis=0))

        proto_rows: List[np.ndarray] = []
        centroid_labels: List[str] = []
        centroid_sizes: List[int] = []

        cursor = 0
        for label in labels:
            label_sentences = per_label_sentences[label]
            label_count = len(label_sentences)
            label_embeddings = all_embeddings[cursor : cursor + label_count]
            cursor += label_count

            raw_centroids, sizes = self._make_centroids(label_embeddings)
            effective_alpha = self.shrinkage_alpha
            if label_count <= self.low_support_single_centroid_threshold:
                effective_alpha = self.shrinkage_alpha * max(1.0, self.low_support_shrinkage_boost)
            shrink = label_count / (label_count + effective_alpha) if effective_alpha > 0 else 1.0
            for centroid, size in zip(raw_centroids, sizes):
                adjusted = self._normalize((shrink * centroid) + ((1.0 - shrink) * global_mean))
                proto_rows.append(adjusted.astype(np.float32))
                centroid_labels.append(label)
                centroid_sizes.append(int(size))

        matrix = np.vstack(proto_rows).astype(np.float32)
        return PrototypeArtifacts(
            labels=labels,
            prototypes=matrix,
            centroid_labels=centroid_labels,
            centroid_sizes=centroid_sizes,
            label_counts=all_sentence_counts,
            config={
                "dedupe_sentences": self.dedupe_sentences,
                "shrinkage_alpha": self.shrinkage_alpha,
                "min_examples_per_centroid": self.min_examples_per_centroid,
                "max_centroids_per_label": self.max_centroids_per_label,
                "random_state": self.random_state,
                "low_support_single_centroid_threshold": self.low_support_single_centroid_threshold,
                "low_support_shrinkage_boost": self.low_support_shrinkage_boost,
            },
        )

    @staticmethod
    def save(artifacts: PrototypeArtifacts, output_dir: str | Path, model_name: str) -> Dict[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        proto_path = out_dir / "prototypes.npz"
        np.savez_compressed(
            proto_path,
            labels=np.array(artifacts.labels),
            prototypes=artifacts.prototypes,
            centroid_labels=np.array(artifacts.centroid_labels),
            centroid_sizes=np.array(artifacts.centroid_sizes, dtype=np.int32),
            label_counts=np.array([artifacts.label_counts[label] for label in artifacts.labels], dtype=np.int32),
        )

        label_map = {label: idx for idx, label in enumerate(artifacts.labels)}
        label_map_path = out_dir / "label_map.json"
        label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

        summary = {
            "model_name": model_name,
            "num_aspects": len(artifacts.labels),
            "num_centroids": int(artifacts.prototypes.shape[0]),
            "embedding_dim": int(artifacts.prototypes.shape[1]),
            "label_counts": artifacts.label_counts,
            "builder_config": artifacts.config,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = out_dir / "train_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "prototypes": proto_path,
            "label_map": label_map_path,
            "summary": summary_path,
        }

    @staticmethod
    def load(prototypes_path: str | Path) -> PrototypeArtifacts:
        data = np.load(Path(prototypes_path), allow_pickle=False)
        labels = [str(x) for x in data["labels"].tolist()]
        prototypes = np.asarray(data["prototypes"], dtype=np.float32)
        centroid_labels_raw = data["centroid_labels"].tolist() if "centroid_labels" in data.files else labels
        centroid_sizes_raw = data["centroid_sizes"].tolist() if "centroid_sizes" in data.files else [1] * len(centroid_labels_raw)
        label_counts_array = data["label_counts"].tolist() if "label_counts" in data.files else [1] * len(labels)
        label_counts = {label: int(count) for label, count in zip(labels, label_counts_array)}
        return PrototypeArtifacts(
            labels=labels,
            prototypes=prototypes,
            centroid_labels=[str(x) for x in centroid_labels_raw],
            centroid_sizes=[int(x) for x in centroid_sizes_raw],
            label_counts=label_counts,
            config={},
        )
