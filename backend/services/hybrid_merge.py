# proto/backend/services/hybrid_merge.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple


PredictionLike = Dict[str, Any]


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _best_key(row: PredictionLike) -> Tuple[str, str]:
    from services.analytics_common import normalize_text
    aspect = normalize_text(row.get("aspect_cluster") or row.get("aspect_raw") or row.get("aspect") or "")
    sentiment = _norm(row.get("sentiment") or "neutral")
    return aspect, sentiment


def merge_predictions(
    explicit_predictions: List[PredictionLike],
    implicit_predictions: List[PredictionLike],
) -> List[PredictionLike]:
    merged: Dict[Tuple[str, str], PredictionLike] = {}

    def upsert(row: PredictionLike) -> None:
        key = _best_key(row)
        if not key[0]:
            return

        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(row)
            return

        # Prefer explicit if confidence is similar, otherwise keep higher confidence
        existing_conf = float(existing.get("confidence", 0.0))
        new_conf = float(row.get("confidence", 0.0))

        existing_source = existing.get("source", "")
        new_source = row.get("source", "")

        if new_source == "explicit" and existing_source != "explicit":
            if new_conf >= existing_conf - 0.05:
                merged[key] = dict(row)
            return

        if new_conf > existing_conf:
            merged[key] = dict(row)

    for row in explicit_predictions:
        row = dict(row)
        row["source"] = row.get("source") or "explicit"
        upsert(row)

    for row in implicit_predictions:
        row = dict(row)
        row["source"] = row.get("source") or "implicit"
        upsert(row)

    return list(merged.values())
