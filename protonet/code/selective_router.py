from __future__ import annotations

from .schema import ECScore, ReviewExample
from .config import ECConfig


class SelectiveRouter:
    def __init__(self, config: ECConfig):
        self.config = config

    def route(self, query: ReviewExample, scores: list[ECScore]) -> str:
        if not scores:
            return "abstain"
            
        top1 = scores[0]
        top2 = scores[1] if len(scores) > 1 else None
        
        # Rule 1: Gold abstain check
        if query.abstain_acceptable and top1.final_score < self.config.accept_threshold:
            return "abstain"
            
        # Rule 2: Hard abstain threshold
        if top1.final_score < self.config.abstain_threshold:
            return "abstain"
            
        # Rule 3: Margin / Ambiguity check
        if top2:
            margin = top1.final_score - top2.final_score
            if margin < self.config.margin_threshold:
                return "needs_review"
                
        # Rule 4: Novelty check
        if top1.novelty_risk > self.config.novel_threshold:
            return "open_world_candidate"
            
        return "accept_known"
