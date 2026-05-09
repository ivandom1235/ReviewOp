from __future__ import annotations

from .schema import ReviewExample
from .config import ECConfig


class ContradictionModule:
    def __init__(self, config: ECConfig):
        self.config = config

    def compute_penalty(self, query: ReviewExample, aspect: str, evidence_support: float) -> float:
        """
        Compute contradiction penalty.
        """
        penalty = 0.0
        
        # 1. Prototype/Evidence mismatch
        # If evidence support is very low (no keyword matches) but we are scoring this aspect,
        # it's a high risk of being a generic embedding match.
        if evidence_support < 0.1:
            penalty += 0.4
            
        # 2. Domain contradiction (if metadata contains domain predictions)
        # Placeholder for future domain-aspect compatibility matrix
        
        # 3. Sentiment contradiction (if metadata contains sentiment)
        # If the aspect name contains negative words but sentiment is positive
        neg_words = {"bad", "poor", "awful", "terrible", "broke", "broken", "slow", "fail"}
        aspect_lower = aspect.lower()
        has_neg = any(w in aspect_lower for w in neg_words)
        
        row_sentiment = query.metadata.get("sentiment_predicted", "unknown")
        if has_neg and row_sentiment == "positive":
            penalty += 0.3
            
        return max(0.0, min(1.0, penalty))
