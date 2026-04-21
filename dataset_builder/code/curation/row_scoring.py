from __future__ import annotations

from collections import Counter
from typing import Any


def train_sentiment_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"negative": 0, "positive": 0, "neutral": 0}
    for row in rows:
        sentiment = str(row.get("implicit", {}).get("dominant_sentiment") or row.get("sentiment") or "neutral").strip().lower()
        if sentiment not in counts:
            sentiment = "neutral"
        counts[sentiment] += 1
    return counts


def aspect_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for aspect in row.get("implicit", {}).get("aspects", []):
            if str(aspect) != "general":
                counts[str(aspect)] += 1
    return counts


def promotion_scoring(
    row: dict[str, Any],
    *,
    train_rows: list[dict[str, Any]],
    core_benchmark_domains: set[str],
) -> dict[str, float]:
    implicit = row.get("implicit", {}) or {}
    aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
    review_reason = str(implicit.get("review_reason") or "").strip().lower()
    domain_family = str(row.get("domain") or "").strip().lower()
    sentiment = str(implicit.get("dominant_sentiment") or row.get("sentiment") or "unknown").strip().lower()
    hardness = str(implicit.get("hardness_tier") or "").strip().upper()
    domain_family_counts = Counter(str(existing.get("domain") or "").strip().lower() for existing in train_rows)
    sentiment_counts = train_sentiment_counts(train_rows)
    aspect_hist = aspect_counts(train_rows)
    aspect_conf = implicit.get("aspect_confidence", {}) or {}
    confidences = [float(value) for value in aspect_conf.values() if value is not None]
    if not confidences:
        confidences = [float(span.get("confidence", 0.0)) for span in list(implicit.get("spans") or []) if span.get("confidence") is not None]
    quality_score = round(max(confidences) if confidences else 0.0, 4)

    usefulness_score = 0.0
    if len(aspects) > 1:
        usefulness_score += 0.18
    if review_reason in {"weak_support", "low_confidence", "domain_soft_mismatch"}:
        usefulness_score += 0.16
    if bool(row.get("abstain_acceptable", False)):
        usefulness_score += 0.2
    if bool(row.get("novel_acceptable", False)):
        usefulness_score += 0.2
    if domain_family in {"electronics", "restaurant", "telecom"}:
        usefulness_score += 0.08
    if hardness in {"H2", "H3"}:
        usefulness_score += 0.08 if hardness == "H2" else 0.12
    if aspects:
        rare_aspect = min((aspect_hist.get(aspect, 0) for aspect in aspects), default=0)
        if rare_aspect <= max(1, len(train_rows) // 12):
            usefulness_score += 0.12
    sentiment_total = max(1, sum(sentiment_counts.values()))
    sentiment_share = sentiment_counts.get(sentiment, 0) / sentiment_total
    if sentiment in {"positive", "negative"} and sentiment_share <= 0.18:
        usefulness_score += 0.12
    if sentiment == "neutral" and sentiment_share >= 0.58:
        usefulness_score -= 0.08
    if domain_family in core_benchmark_domains and domain_family_counts.get(domain_family, 0) <= max(1, len(train_rows) // 10):
        usefulness_score += 0.1
    if not aspects:
        usefulness_score -= 0.08

    redundancy_score = 0.0
    if len(aspects) <= 1:
        redundancy_score += 0.06
    if aspects and any(aspect_hist.get(aspect, 0) > 0 for aspect in aspects):
        redundancy_score += 0.08
    if review_reason in {"fallback_general", "implicit_not_ready"}:
        redundancy_score += 0.12
    if sentiment == "neutral" and sentiment_share >= 0.58:
        redundancy_score += 0.08
    if domain_family_counts.get(domain_family, 0) > max(1, len(train_rows) // 8):
        redundancy_score += 0.08
    redundancy_score = round(min(1.0, max(0.0, redundancy_score)), 4)
    usefulness_score = round(min(1.0, max(0.0, usefulness_score)), 4)
    return {
        "quality_score": quality_score,
        "usefulness_score": usefulness_score,
        "redundancy_score": redundancy_score,
        "priority_score": round(max(0.0, min(1.0, (0.45 * quality_score) + (0.4 * usefulness_score) - (0.2 * redundancy_score))), 4),
    }
