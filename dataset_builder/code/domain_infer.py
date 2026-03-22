"""Domain inference from filename, columns, and text tokens."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List

from mappings import DOMAIN_HINTS
from utils import normalize_text


def infer_domain(file_path: str, columns: List[str], texts: Iterable[str], aspects: Iterable[str] | None = None) -> str:
    scores = Counter()
    file_hint = Path(file_path).stem.lower()
    col_blob = " ".join(c.lower() for c in columns)
    aspect_blob = " ".join(normalize_text(x).lower() for x in (aspects or []))

    for domain, words in DOMAIN_HINTS.items():
        for w in words:
            if w in file_hint:
                scores[domain] += 3
            if w in col_blob:
                scores[domain] += 1
            if w in aspect_blob:
                scores[domain] += 2

    text_tokens = " ".join(normalize_text(t).lower() for _, t in zip(range(300), texts))
    for domain, words in DOMAIN_HINTS.items():
        for w in words:
            if w in text_tokens:
                scores[domain] += 1

    # A file-level metadata hint is useful, but should not override stronger text or filename cues.
    if any(k in file_hint for k in ["laptop", "phone", "tablet", "computer", "electronics"]):
        scores["electronics"] += 2
    if any(k in file_hint for k in ["restaurant", "food", "dining", "meal"]):
        scores["restaurant"] += 2
    if any(k in file_hint for k in ["hotel", "room", "stay", "travel"]):
        scores["hotel"] += 2
    if any(k in file_hint for k in ["telecom", "network", "signal", "call"]):
        scores["telecom"] += 2

    if not scores:
        return "generic"
    top_domain, top_score = scores.most_common(1)[0]
    return top_domain if top_score > 0 else "generic"
