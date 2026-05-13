from __future__ import annotations
import json
from typing import Any
from ..schemas.interpretation import Interpretation
from ..llm.provider_factory import get_llm_client
from dataclasses import replace

class SentimentClassifier:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.client = None
        if cfg and cfg.llm_provider != "none":
            try:
                self.client = get_llm_client(cfg)
            except Exception:
                self.client = None

    def classify_sentiment_heuristic(self, text: str) -> str:
        """Simple heuristic for sentiment analysis based on keyword matching."""
        pos = {"great", "good", "amazing", "excellent", "fast", "long", "happy", "love", "perfect"}
        neg = {"bad", "poor", "terrible", "slow", "short", "dim", "lag", "broken", "worst", "hate"}
        
        text_lower = text.lower()
        pos_count = sum(1 for word in pos if word in text_lower)
        neg_count = sum(1 for word in neg if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    def classify_batch(self, row_text: str, interpretations: list[Interpretation]) -> list[Interpretation]:
        """Analyze sentiment for multiple interpretations in one call if possible."""
        if not interpretations:
            return []
            
        if not self.client:
            return [replace(i, sentiment=self.classify_sentiment_heuristic(row_text)) for i in interpretations]

        from .prompts import build_batch_sentiment_prompt
        try:
            aspect_texts = [i.aspect_raw for i in interpretations]
            system_prompt, user_prompt = build_batch_sentiment_prompt(row_text, aspect_texts)
            response_text = self.client.generate(user_prompt, system_prompt=system_prompt).strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            results = json.loads(response_text)
            # Create a lookup map for sentiment
            sentiment_map = {item["aspect"].lower(): item["sentiment"].lower() for item in results if "aspect" in item and "sentiment" in item}
            
            final_interps = []
            for i in interpretations:
                sent = sentiment_map.get(i.aspect_raw.lower(), "neutral")
                if sent not in ["positive", "negative", "neutral"]:
                    sent = self.classify_sentiment_heuristic(row_text)
                final_interps.append(replace(i, sentiment=sent))
            return final_interps
        except Exception:
            # Fallback to heuristic for all if batch fails
            return [replace(i, sentiment=self.classify_sentiment_heuristic(row_text)) for i in interpretations]

    def classify(self, row_text: str, interpretation: Interpretation) -> Interpretation:
        """Analyze sentiment for a specific interpretation and return updated object."""
        return self.classify_batch(row_text, [interpretation])[0]

def analyze_sentiment(row_text: str, interpretation: Interpretation, cfg: Any = None) -> str:
    """Legacy function for single calls (uses heuristic if no cfg)."""
    classifier = SentimentClassifier(cfg)
    return classifier.classify(row_text, interpretation).sentiment
