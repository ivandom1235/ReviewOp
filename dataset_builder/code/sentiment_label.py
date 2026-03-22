from __future__ import annotations

import re
from typing import Dict

POS = {"great", "good", "excellent", "amazing", "fast", "love", "clean", "polite", "helpful", "smooth", "fresh"}
NEG = {"bad", "poor", "terrible", "awful", "slow", "hate", "dirty", "rude", "lag", "late", "delay", "bland", "broken"}


def normalize_sentiment(value: str) -> str:
    v = str(value or "").strip().lower()
    mapping = {
        "positive": "positive", "pos": "positive",
        "neutral": "neutral", "neu": "neutral", "mixed": "neutral",
        "negative": "negative", "neg": "negative",
    }
    return mapping.get(v, "neutral")


def infer_aspect_sentiment(evidence_sentence: str, raw_sentiment: str, rating: int | None, multi_aspect: bool) -> Dict[str, str | float | bool]:
    tokens = [t for t in re.split(r"\W+", evidence_sentence.lower()) if t]
    pos = sum(1 for t in tokens if t in POS)
    neg = sum(1 for t in tokens if t in NEG)

    if pos > neg:
        sentiment = "positive"
        confidence = 0.9
    elif neg > pos:
        sentiment = "negative"
        confidence = 0.9
    else:
        explicit = normalize_sentiment(raw_sentiment)
        if multi_aspect:
            # In multi-aspect settings, avoid review-level leakage.
            sentiment = "neutral" if explicit == "neutral" else explicit
            confidence = 0.55 if sentiment == "neutral" else 0.65
        else:
            if explicit != "neutral":
                sentiment = explicit
                confidence = 0.7
            elif rating is not None:
                sentiment = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
                confidence = 0.6
            else:
                sentiment = "neutral"
                confidence = 0.5

    ambiguity = pos > 0 and neg > 0
    unresolved = sentiment == "neutral" and pos == 0 and neg == 0
    return {
        "sentiment": sentiment,
        "ambiguous": ambiguity,
        "unresolved": unresolved,
        "sentiment_confidence": confidence,
    }
