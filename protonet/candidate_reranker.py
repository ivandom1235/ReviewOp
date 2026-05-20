from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

GENERIC_LABELS = {"quality", "value", "performance", "usability"}

@dataclass
class CandidateReranker:
    model: Pipeline
    feature_names: list[str]

    def score(self, candidate, *, source_type: str) -> float:
        x = np.array([candidate_features(candidate, source_type, self.feature_names)], dtype=float)
        return float(self.model.predict_proba(x)[0, 1])

def candidate_features(c, source_type: str, names: list[str]) -> list[float]:
    source = str(getattr(c, "candidate_source", "") or "")
    reason = str(getattr(c, "decision_reason", "") or "")
    aspect = str(getattr(c, "aspect", "") or "")
    values = {
        "proto_score": float(getattr(c, "proto_score", 0.0) or 0.0),
        "final_score": float(getattr(c, "final_score", 0.0) or 0.0),
        "known_confidence": float(getattr(c, "known_confidence", 0.0) or 0.0),
        "lexical": float(getattr(c, "lexical_evidence_support", 0.0) or 0.0),
        "semantic": float(getattr(c, "semantic_evidence_support", 0.0) or 0.0),
        "description": float(getattr(c, "description_support", 0.0) or 0.0),
        "memory": float(getattr(c, "memory_support", 0.0) or 0.0),
        "margin": float(getattr(c, "margin_to_next", 0.0) or 0.0),
        "support_log": np.log1p(float(getattr(c, "support_count", 0) or 0)),
        "rank_inv": 1.0 / max(1.0, float(getattr(c, "rank", 1) or 1)),
        "unknown": float(getattr(c, "unknown_score", 0.0) or 0.0),
        "is_classifier": 1.0 if source == "classifier" else 0.0,
        "is_hybrid": 1.0 if source == "hybrid" else 0.0,
        "is_generic": 1.0 if aspect in GENERIC_LABELS else 0.0,
        "is_implicit": 1.0 if source_type in {"implicit", "synthetic", "counterfactual", "silver"} else 0.0,
    }
    return [values[n] for n in names]

def train_candidate_reranker(rows, labels) -> CandidateReranker:
    feature_names = [
        "proto_score", "final_score", "known_confidence", "lexical", "semantic",
        "description", "memory", "margin", "support_log", "rank_inv",
        "unknown", "is_classifier", "is_hybrid", "is_generic", "is_implicit",
    ]
    X = np.array([candidate_features(c, source_type, feature_names) for c, source_type in rows], dtype=float)
    y = np.array(labels, dtype=int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=500, solver="liblinear", random_state=42)),
    ])
    model.fit(X, y)
    return CandidateReranker(model=model, feature_names=feature_names)
