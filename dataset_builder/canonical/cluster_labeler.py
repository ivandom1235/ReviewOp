from __future__ import annotations
from collections import Counter
import re
from typing import List

class ClusterLabeler:
    """
    Generates high-quality canonical labels for evidence clusters.
    """
    BEHAVIOR_CUES = {
        "broke", "broken", "fraying", "loose", "dropped", "dropping", "crashed", "waited",
        "slow", "fast", "cold", "hot", "tiny", "small", "expensive", "cheap", "stale",
        "late", "delayed", "disconnected", "logged", "logout", "noisy", "buffering", "weak",
        "soggy", "undercooked", "responsive", "friendly", "terrible", "great", "poor", "tiny",
        "frayed", "opened", "opened", "fray", "frays", "cut", "cutting", "died", "died",
    }
    SENTIMENT_ONLY = {"good", "bad", "great", "poor", "nice", "awful", "excellent", "terrible", "best", "worst", "love", "hate", "amazing", "horrible"}
    BROAD_LABELS = {"unknown", "general", "misc", "thing", "item", "product", "place", "good", "bad", "food", "service", "quality", "battery", "screen", "computer", "restaurant"}

    def label_cluster(self, trigger_patterns: List[str], base_aspect: str) -> str:
        """Old API for backward compatibility."""
        label, _ = self.label_cluster_v2(trigger_patterns, base_aspect)
        return label

    def label_cluster_v2(self, trigger_patterns: List[str], base_aspect: str) -> tuple[str, str]:
        """
        Returns (canonical_label, representative_phrase).
        """
        cleaned = [p for p in (self._clean(p) for p in trigger_patterns) if p]
        base_clean = self._clean(base_aspect)
        if not cleaned:
            return (base_clean or "unknown"), (base_aspect or "unknown")

        counts = Counter(cleaned)
        candidates = list(dict.fromkeys(cleaned))
        if base_clean:
            candidates.append(base_clean)

        best_label = base_clean or cleaned[0]
        best_score = float("-inf")
        for candidate in candidates:
            score = float(counts.get(candidate, 0)) * 2.0
            tokens = candidate.split("_")
            token_count = len(tokens)
            score += 0.10 if candidate == base_clean else 0.0
            if self._is_generic(candidate):
                score -= 1.0
            if self._is_sentiment_only(tokens):
                score -= 1.0
            if self._looks_entity_only(tokens):
                score -= 0.8
            if self._is_broad_label(candidate):
                score -= 0.8
            if any(t in self.BEHAVIOR_CUES for t in tokens):
                score += 0.5
            if counts.get(candidate, 0) > 1:
                score += 0.5
            if 2 <= token_count <= 3 and not self._is_generic(candidate) and not self._is_sentiment_only(tokens):
                score += 0.3
            if score > best_score:
                best_score = score
                best_label = candidate

        # Find best original phrase matching the best label
        best_phrase = trigger_patterns[0] if trigger_patterns else base_aspect
        for p in trigger_patterns:
            if self._clean(p) == best_label:
                best_phrase = p
                break

        # If best_label is too long/noisy, simplify for canonical version
        canonical = best_label
        if len(canonical.split("_")) > 2:
             # Try to find a shorter sub-pattern or use base_aspect
             if base_clean and base_clean in canonical:
                 canonical = base_clean

        return canonical, best_phrase

    def _clean(self, text: str) -> str:
        # Remove common stop words and punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        stop_words = {"the", "a", "an", "is", "was", "were", "are", "but", "and", "or", "to", "in", "it"}
        words = [w for w in text.split() if w not in stop_words]
        return "_".join(words[:4]) # Limit to 4 words for a label

    def _is_generic(self, candidate: str) -> bool:
        return candidate in {
            "unknown", "general", "misc", "thing", "item", "product", "place", "good", "bad",
            "food", "service", "quality", "battery", "screen", "computer", "restaurant",
        }

    def _is_sentiment_only(self, tokens: List[str]) -> bool:
        if not tokens:
            return True
        return all(token in self.SENTIMENT_ONLY for token in tokens)

    def _looks_entity_only(self, tokens: List[str]) -> bool:
        if not tokens:
            return True
        return len(tokens) == 1 and len(tokens[0]) <= 4

    def _is_broad_label(self, candidate: str) -> bool:
        cleaned = candidate.strip().lower()
        return cleaned in self.BROAD_LABELS
