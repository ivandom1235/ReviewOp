from __future__ import annotations

from datetime import datetime
from typing import Optional

from models.tables import Prediction


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except Exception:
            return None


def normalize_text(value: str) -> str:
    """Basic normalization + plural reduction + attribute stripping."""
    val = " ".join((value or "").lower().replace("_", " ").replace("-", " ").split())
    if not val:
        return ""
    
    # Strip common redundant attributes for cleaner graphs
    if val.endswith(" quality") and len(val) > 8:
        val = val[:-8].strip()
    
    # Simple heuristic-based lemmatization for common aspect plurals
    tokens = val.split()
    results = []
    for t in tokens:
        if t.endswith("ies") and len(t) > 4:
            results.append(t[:-3] + "y")
        elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            results.append(t[:-1])
        else:
            results.append(t)
    return " ".join(results)


def aspect_key(value: str) -> str:
    return normalize_text(value).replace(" ", "_")


def aspect_label(aspect: str) -> str:
    return " ".join(part.capitalize() for part in aspect.replace("-", " ").replace("_", " ").split()) or aspect


def canonical_aspect(prediction: Prediction) -> str:
    # Priority: canonical > normalized > cluster > raw
    val = (
        getattr(prediction, "aspect_canonical", None)
        or getattr(prediction, "aspect_normalized", None)
        or getattr(prediction, "aspect_cluster", None)
        or getattr(prediction, "aspect_raw", None)
        or "unknown"
    )
    return aspect_key(str(val))


def infer_origin(aspect: str, snippet: str | None) -> str:
    aspect_terms = set(normalize_text(aspect).split())
    snippet_terms = set(normalize_text(snippet or "").split())
    if aspect_terms and aspect_terms.issubset(snippet_terms):
        return "explicit"
    return "implicit"


def prediction_origin(prediction: Prediction, snippet: str | None) -> str:
    source = str(getattr(prediction, "source", "") or "").strip().lower()
    if source in {"explicit", "implicit"}:
        return source
    return infer_origin(canonical_aspect(prediction), snippet)
