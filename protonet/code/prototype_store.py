from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .schema import ReviewExample

_DEFAULT_DESCRIPTIONS_PATH = Path("protonet/configs/generic_aspect_descriptions.json")


class PrototypeStore:
    def __init__(self):
        self.prototypes: dict[str, torch.Tensor] = {}
        self.aspect_counts: dict[str, int] = {}
        self.prototype_sources: dict[str, str] = {}
        self.description_embeddings: dict[str, torch.Tensor] = {}
        self.trigger_embeddings: dict[str, list[torch.Tensor]] = {}

    def build_from_examples(
        self,
        examples: list[ReviewExample],
        encoder: "ECEncoder",
        use_evidence: bool = True,
    ) -> "PrototypeStore":
        aspect_embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)

        for ex in examples:
            for interp in ex.gold_interpretations:
                text = interp.evidence_text if use_evidence else ex.text
                if not text:
                    text = ex.text
                embedding = encoder.encode([text])[0]
                aspect_embeddings[interp.aspect].append(embedding)

        for aspect, embeddings in aspect_embeddings.items():
            stacked = torch.stack(embeddings)
            self.prototypes[aspect] = torch.mean(stacked, dim=0)
            self.aspect_counts[aspect] = len(embeddings)
            self.prototype_sources.setdefault(aspect, "train_evidence")

        return self

    def build_from_descriptions(
        self,
        encoder: "ECEncoder",
        descriptions_path: str | Path = _DEFAULT_DESCRIPTIONS_PATH,
    ) -> "PrototypeStore":
        """
        Add description-based prototypes from the generic aspect families config.
        Skips aspects already covered by train_evidence prototypes so training
        examples always take precedence when both are present.
        """
        path = Path(descriptions_path)
        if not path.exists():
            return self

        try:
            bank: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return self

        for aspect, info in bank.items():
            description = info.get("description", "").strip()
            if not description:
                continue
            
            embedding = encoder.encode([description])[0]
            self.description_embeddings[aspect] = embedding
            
            triggers = info.get("behavior_triggers", []) + info.get("aliases", []) + [aspect]
            if triggers:
                self.trigger_embeddings[aspect] = [encoder.encode([t])[0] for t in triggers if t]

            if aspect in self.prototypes:
                source = self.prototype_sources.get(aspect, "")
                if "generic_description" not in source:
                    self.prototype_sources[aspect] = source + "+generic_description"
                continue
            
            self.prototypes[aspect] = embedding
            self.aspect_counts[aspect] = 0
            self.prototype_sources[aspect] = "generic_description"

        return self

    def build_from_memory(
        self,
        encoder: "ECEncoder",
        memory_summary: dict[str, Any],
    ) -> "PrototypeStore":
        """
        Add AspectMemory cluster prototypes from an aspect_memory_summary dict.
        Uses representative trigger patterns as prototype text.
        """
        top_clusters = memory_summary.get("top_clusters", [])

        for cluster in top_clusters:
            aspect = cluster.get("suggested_aspect") or cluster.get("aspect_raw")
            if not aspect:
                continue

            trigger = cluster.get("representative_trigger", "")
            patterns = cluster.get("trigger_patterns", [])
            texts = [t for t in ([trigger] + patterns) if t]
            if not texts:
                continue

            embeddings = [encoder.encode([t])[0] for t in texts[:5]]
            if not embeddings:
                continue

            stacked = torch.stack(embeddings)
            proto = torch.mean(stacked, dim=0)

            if aspect in self.prototypes:
                source = self.prototype_sources.get(aspect, "")
                if "aspect_memory" not in source:
                    self.prototype_sources[aspect] = source + "+aspect_memory"
            else:
                self.prototypes[aspect] = proto
                self.aspect_counts[aspect] = len(texts)
                self.prototype_sources[aspect] = "aspect_memory"

        return self

    def merge_prototype_sources(self) -> dict[str, int]:
        """Return a summary of prototype counts per source type."""
        counts: dict[str, int] = defaultdict(int)
        for source in self.prototype_sources.values():
            for part in source.split("+"):
                counts[part] += 1
        return dict(counts)

    def get_similarity(self, query_embedding: torch.Tensor) -> dict[str, float]:
        if not self.prototypes:
            return {}
        q_norm = query_embedding / (query_embedding.norm(p=2) + 1e-8)
        similarities = {}
        for aspect, proto in self.prototypes.items():
            p_norm = proto / (proto.norm(p=2) + 1e-8)
            similarities[aspect] = torch.dot(q_norm, p_norm).item()
        return similarities

    def get_all_aspects(self) -> list[str]:
        return sorted(list(self.prototypes.keys()))

    def summary(self) -> dict[str, Any]:
        source_counts = self.merge_prototype_sources()
        return {
            "prototype_sources": source_counts,
            "total_prototypes": len(self.prototypes),
            "domain_specific_required": False,
            "generic_description_enabled": source_counts.get("generic_description", 0) > 0,
            "memory_enabled": source_counts.get("aspect_memory", 0) > 0,
        }
