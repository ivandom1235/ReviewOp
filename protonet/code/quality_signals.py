from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


HARDNESS_MULTIPLIERS: Dict[str, float] = {
    "H0": 1.00,
    "H1": 1.04,
    "H2": 1.10,
    "H3": 1.18,
}


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def example_quality_weight(item: Dict[str, Any]) -> float:
    confidence = _bounded(float(item.get("confidence", 1.0)), 0.40, 1.0)
    hardness_tier = str(item.get("hardness_tier") or "H0").strip().upper()
    hardness = HARDNESS_MULTIPLIERS.get(hardness_tier, 1.0)
    grounded = 0.82 if bool(item.get("evidence_fallback_used", False)) else 1.0
    ambiguity = _bounded(1.0 - 0.12 * float(item.get("benchmark_ambiguity_score", 0.0)), 0.78, 1.0)
    abstain_safe = 0.96 if bool(item.get("abstain_acceptable", False)) else 1.0
    weight = confidence * hardness * grounded * ambiguity * abstain_safe
    return _bounded(weight, 0.25, 1.50)


def prediction_error_buckets(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    buckets = Counter(
        {
            "correct_and_confident": 0,
            "correct_but_abstained": 0,
            "wrong_but_confident": 0,
            "wrong_and_abstained": 0,
        }
    )
    for row in rows:
        correct = bool(row.get("correct", False))
        abstained = bool(row.get("abstained", False))
        if correct and abstained:
            buckets["correct_but_abstained"] += 1
        elif correct:
            buckets["correct_and_confident"] += 1
        elif abstained:
            buckets["wrong_and_abstained"] += 1
        else:
            buckets["wrong_but_confident"] += 1
    return dict(buckets)


def top_aspect_confusions(rows: List[Dict[str, Any]], *, separator: str = "__", limit: int = 5) -> List[Dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if bool(row.get("correct", False)):
            continue
        true_label = str(row.get("true_label") or "")
        pred_label = str(row.get("pred_label") or "")
        true_aspect = true_label.split(separator, 1)[0] if separator in true_label else true_label
        pred_aspect = pred_label.split(separator, 1)[0] if separator in pred_label else pred_label
        counter[(true_aspect, pred_aspect)] += 1

    top: List[Dict[str, Any]] = []
    for (true_aspect, pred_aspect), count in counter.most_common(limit):
        top.append(
            {
                "true_aspect": true_aspect,
                "pred_aspect": pred_aspect,
                "count": int(count),
            }
        )
    return top
