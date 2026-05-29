from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

_DEFAULT_CONFIG_PATH = Path("dataset_builder/config/generic_aspect_families.json")


class GenericParentInferencer:
    def __init__(
        self,
        families: dict[str, Any],
        threshold_auto: float = 0.78,
        threshold_review: float = 0.60,
    ):
        self._families = families
        self.threshold_auto = threshold_auto
        self.threshold_review = threshold_review
        self._model = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = _DEFAULT_CONFIG_PATH,
        threshold_auto: float = 0.78,
        threshold_review: float = 0.60,
    ) -> "GenericParentInferencer":
        path = Path(config_path)
        families: dict[str, Any] = {}
        if path.exists():
            try:
                families = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                families = {}
        return cls(families, threshold_auto=threshold_auto, threshold_review=threshold_review)

    @property
    def _descriptions(self) -> dict[str, str]:
        return {
            family: info.get("description", "")
            for family, info in self._families.items()
            if info.get("description")
        }

    @property
    def model(self):
        if self._model is None and SentenceTransformer is not None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def infer_parent(
        self, trigger_pattern: str, evidence_examples: list[str]
    ) -> dict[str, Any]:
        descriptions = self._descriptions
        if not descriptions:
            return {
                "generic_parent": None,
                "score": 0.0,
                "top_candidates": [],
                "decision": "none",
                "reason": "no generic families loaded",
            }

        if self.model is None or util is None:
            return {
                "generic_parent": None,
                "score": 0.0,
                "top_candidates": [],
                "decision": "none",
                "reason": "sentence-transformers not available",
            }

        query = f"{trigger_pattern}. " + " ".join(evidence_examples[:2])
        query_emb = self.model.encode(query, convert_to_tensor=True)

        results = []
        for parent, description in descriptions.items():
            parent_emb = self.model.encode(description, convert_to_tensor=True)
            score = float(util.cos_sim(query_emb, parent_emb).item())
            results.append({"parent": parent, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[0]

        decision = "none"
        if top["score"] >= self.threshold_auto:
            decision = "auto_accept"
        elif top["score"] >= self.threshold_review:
            decision = "review_queue"

        return {
            "generic_parent": top["parent"] if decision != "none" else None,
            "score": top["score"],
            "top_candidates": results[:3],
            "decision": decision,
            "reason": f"Semantic similarity score {top['score']:.2f} against {top['parent']} description",
        }
