from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable


def aspect_metrics(pred_rows: Iterable[dict[str, Any]]) -> Dict[str, Any]:
    rows = list(pred_rows)
    aspect_counts: Counter[str] = Counter()
    sentiments: Counter[str] = Counter()
    for row in rows:
        implicit = row.get("implicit", {})
        for aspect in implicit.get("aspects", []):
            aspect_counts[aspect] += 1
            sentiments[str(implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral")))] += 1
    return {
        "num_rows": len(rows),
        "aspect_counts": dict(aspect_counts),
        "sentiment_counts": dict(sentiments),
    }


def _span_signature(span: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(span.get("latent_aspect") or span.get("aspect") or ""),
        str(span.get("surface_aspect") or span.get("matched_surface") or ""),
        int(span.get("start_char", -1) or -1),
        int(span.get("end_char", -1) or -1),
        str(span.get("sentiment") or "neutral"),
    )


def span_f1(pred_rows: Iterable[dict[str, Any]], gold_rows: Iterable[dict[str, Any]]) -> Dict[str, Any]:
    pred = list(pred_rows)
    gold = list(gold_rows)
    pred_spans = set()
    gold_spans = set()
    for row in pred:
        for span in row.get("implicit", {}).get("spans", []):
            pred_spans.add((row.get("id"), *_span_signature(span)))
    for row in gold:
        for span in row.get("implicit", {}).get("spans", []):
            gold_spans.add((row.get("id"), *_span_signature(span)))
    true_positive = len(pred_spans & gold_spans)
    precision = true_positive / len(pred_spans) if pred_spans else 0.0
    recall = true_positive / len(gold_spans) if gold_spans else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "predicted": len(pred_spans),
        "gold": len(gold_spans),
    }


def benchmark_scorecard(
    pred_rows: Iterable[dict[str, Any]],
    gold_rows: Iterable[dict[str, Any]],
    *,
    benchmark_family: str,
    model_family: str,
) -> Dict[str, Any]:
    scorecard = {
        "benchmark_family": benchmark_family,
        "model_family": model_family,
        "span_f1": span_f1(pred_rows, gold_rows),
    }
    scorecard["aspect_metrics"] = aspect_metrics(pred_rows)
    return scorecard
