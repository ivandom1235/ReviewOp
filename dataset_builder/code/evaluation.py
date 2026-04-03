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
    pred = list(pred_rows)
    gold = list(gold_rows)
    scorecard = {
        "benchmark_family": benchmark_family,
        "model_family": model_family,
        "span_f1": span_f1(pred, gold),
    }
    scorecard["aspect_metrics"] = aspect_metrics(pred)
    scorecard["language_breakdown"] = {}
    languages = sorted({str(row.get("language", "unknown")) for row in pred})
    for language in languages:
        pred_subset = [row for row in pred if str(row.get("language", "unknown")) == language]
        gold_subset = [row for row in gold if str(row.get("language", "unknown")) == language]
        if pred_subset or gold_subset:
            scorecard["language_breakdown"][language] = {
                "span_f1": span_f1(pred_subset, gold_subset),
                "aspect_metrics": aspect_metrics(pred_subset),
            }
    return scorecard


def _norm_aspect(value: Any) -> str:
    return str(value or "").strip().lower()


def _label_signature(label: dict[str, Any]) -> tuple[str, str]:
    return (_norm_aspect(label.get("aspect")), str(label.get("sentiment") or "neutral").lower())


def gold_eval(rows: Iterable[dict[str, Any]]) -> Dict[str, Any]:
    records = list(rows)
    eligible = [row for row in records if isinstance(row.get("gold_labels"), list) and row.get("gold_labels")]
    if not eligible:
        return {
            "has_gold_labels": False,
            "num_rows_with_gold": 0,
            "aspect_f1": 0.0,
            "sentiment_f1": 0.0,
            "span_overlap_f1": 0.0,
            "by_domain": {},
        }

    def compute(subset: list[dict[str, Any]]) -> dict[str, Any]:
        gold_aspects: set[tuple[Any, str]] = set()
        pred_aspects: set[tuple[Any, str]] = set()
        gold_sentiments: set[tuple[Any, str, str]] = set()
        pred_sentiments: set[tuple[Any, str, str]] = set()
        gold_spans: set[tuple[Any, str, int, int]] = set()
        pred_spans: set[tuple[Any, str, int, int]] = set()
        for row in subset:
            row_id = row.get("id")
            for label in row.get("gold_labels", []):
                if not isinstance(label, dict):
                    continue
                aspect, sentiment = _label_signature(label)
                if aspect:
                    gold_aspects.add((row_id, aspect))
                    gold_sentiments.add((row_id, aspect, sentiment))
                    start = int(label.get("start", -1) or -1)
                    end = int(label.get("end", -1) or -1)
                    if start >= 0 and end >= 0:
                        gold_spans.add((row_id, aspect, start, end))
            implicit = row.get("implicit", {})
            for aspect in implicit.get("aspects", []):
                aspect_key = _norm_aspect(aspect)
                if aspect_key and aspect_key != "general":
                    pred_aspects.add((row_id, aspect_key))
                    sentiment = str(implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral"))).lower()
                    pred_sentiments.add((row_id, aspect_key, sentiment))
            for span in implicit.get("spans", []):
                if not isinstance(span, dict):
                    continue
                aspect_key = _norm_aspect(span.get("latent_aspect") or span.get("aspect"))
                start = int(span.get("start_char", -1) or -1)
                end = int(span.get("end_char", -1) or -1)
                if aspect_key and aspect_key != "general" and start >= 0 and end >= 0:
                    pred_spans.add((row_id, aspect_key, start, end))

        def f1(pred: set, gold: set) -> float:
            tp = len(pred & gold)
            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(gold) if gold else 0.0
            return round((2 * precision * recall / (precision + recall)) if precision + recall else 0.0, 4)

        return {
            "num_rows": len(subset),
            "aspect_f1": f1(pred_aspects, gold_aspects),
            "sentiment_f1": f1(pred_sentiments, gold_sentiments),
            "span_overlap_f1": f1(pred_spans, gold_spans),
        }

    overall = compute(eligible)
    by_domain: dict[str, Any] = {}
    for domain in sorted({str(row.get("domain", "unknown")) for row in eligible}):
        domain_rows = [row for row in eligible if str(row.get("domain", "unknown")) == domain]
        by_domain[domain] = compute(domain_rows)
    return {
        "has_gold_labels": True,
        "num_rows_with_gold": len(eligible),
        "aspect_f1": overall["aspect_f1"],
        "sentiment_f1": overall["sentiment_f1"],
        "span_overlap_f1": overall["span_overlap_f1"],
        "by_domain": by_domain,
    }
