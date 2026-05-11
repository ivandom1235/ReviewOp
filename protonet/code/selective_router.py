from __future__ import annotations

from .schema import ECScore, ReviewExample
from .config import ECConfig


class SelectiveRouter:
    def __init__(self, config: ECConfig):
        self.config = config

    def route_candidates(self, query: ReviewExample, scores: list[ECScore]) -> list[ECScore]:
        """
        Route each candidate independently based on its signals.
        Returns a new list of scores with updated decision fields.
        """
        if not scores:
            return []

        from dataclasses import replace
        results = []
        
        for i, current in enumerate(scores):
            next_score = scores[i+1] if i + 1 < len(scores) else None
            
            decision = "accept_known"
            
            # Rule 1: Abstain (Very low score)
            if current.final_score < self.config.abstain_threshold:
                decision = "abstain"
            
            # Rule 2: Novelty check
            elif current.novelty_risk > self.config.novel_threshold:
                decision = "open_world_candidate"
                
            # Rule 3: Acceptance threshold
            elif current.final_score < self.config.accept_threshold:
                decision = "needs_review"
                
            # Rule 4: Margin / Ambiguity check (with next candidate)
            elif next_score:
                margin = current.final_score - next_score.final_score
                if margin < self.config.margin_threshold:
                    decision = "needs_review"
            
            results.append(replace(current, decision=decision))
            
        return results

    def route(self, query: ReviewExample, scores: list[ECScore]) -> str:
        """Legacy row-level routing for backward compatibility. Returns the top decision."""
        routed = self.route_candidates(query, scores)
        return routed[0].decision if routed else "abstain"
