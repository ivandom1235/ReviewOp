from __future__ import annotations
from typing import Any, Optional
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

class GenericParentInferencer:
    # Abstract domain-neutral descriptions as per specification
    GENERIC_PARENTS = {
        "durability": "lasting over time; not breaking, tearing, fraying, cracking, or wearing out",
        "reliability": "consistent operation; not failing, disconnecting, crashing, timing out, or resetting",
        "usability": "ease of use, setup, navigation, clarity, confusion, or user effort",
        "service": "human support, responsiveness, waiting, staff behavior, or issue resolution",
        "value": "cost, worth, affordability, fairness of price, or cost-benefit",
        "comfort": "physical comfort, fit, noise, temperature, seating, or wearing experience",
        "performance": "speed, responsiveness, lag, throughput, or task execution",
        "availability": "access, stock, availability, missing items, unavailable service, or inability to obtain something",
        "cleanliness": "clean, dirty, hygiene, smell, mess, stains, or sanitation",
        "appearance": "visual look, design, color, shape, styling, or aesthetics"
    }

    def __init__(self, threshold_auto: float = 0.78, threshold_review: float = 0.60):
        self.threshold_auto = threshold_auto
        self.threshold_review = threshold_review
        self._model = None

    @property
    def model(self):
        if self._model is None and SentenceTransformer is not None:
            # Reusing the same small model for efficiency
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def infer_parent(self, trigger_pattern: str, evidence_examples: list[str]) -> dict[str, Any]:
        if self.model is None or util is None:
            return {
                "generic_parent": None, 
                "score": 0.0, 
                "top_candidates": [], 
                "decision": "none", 
                "reason": "sentence-transformers not available"
            }
            
        # Combine trigger and representative evidence for context-rich inference
        query = f"{trigger_pattern}. " + " ".join(evidence_examples[:2])
        query_emb = self.model.encode(query, convert_to_tensor=True)
        
        results = []
        for parent, description in self.GENERIC_PARENTS.items():
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
            "reason": f"Semantic similarity score {top['score']:.2f} against {top['parent']} description"
        }
