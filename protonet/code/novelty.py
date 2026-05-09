from __future__ import annotations

import torch
from .schema import ReviewExample
from .config import ECConfig


class NoveltyModule:
    def __init__(self, config: ECConfig):
        self.config = config

    def compute_risk(self, similarities: list[float], memory_support: float = 0.0) -> float:
        """
        Compute novelty risk score.
        High risk means the example is likely novel or out-of-distribution.
        """
        if not similarities:
            return 1.0
            
        max_sim = max(similarities)
        
        # 1. Distance-based risk
        risk = 1.0 - max_sim
        
        # 2. Entropy penalty (ambiguity between top classes increases risk)
        if len(similarities) > 1:
            sorted_sims = sorted(similarities, reverse=True)
            margin = sorted_sims[0] - sorted_sims[1]
            if margin < 0.1: # Threshold for "similar" high scores
                risk += 0.15
        
        # 3. Memory discount (if it matches memory, it's not novel)
        risk = risk * (1.0 - memory_support)
        
        return max(0.0, min(1.0, risk))
