from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from math import sqrt
from typing import Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session, selectinload

from models.tables import Prediction, Review
from services.analytics_common import aspect_key, canonical_aspect, normalize_text, parse_dt, prediction_origin


def _pct(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return (numer / denom) * 100.0


def _wilson_ci_95(p_hat: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.96
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    half = (z * sqrt((p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _first_snippet(prediction: Prediction) -> str | None:
    span = min(
        prediction.evidence_spans,
        key=lambda item: (item.start_char, item.end_char),
        default=None,
    )
    return span.snippet if span else None


def _aspect_filter_candidates(aspect: str) -> set[str]:
    normalized = normalize_text(aspect)
    if not normalized:
        return set()

    candidates = {normalized, aspect_key(normalized), normalized.replace(" ", "-"), str(aspect or "").strip()}
    tokens = normalized.split()
    if tokens:
        last = tokens[-1]
        if last.endswith("y") and len(last) > 1:
            plural_last = f"{last[:-1]}ies"
        elif last.endswith("s"):
            plural_last = last
        else:
            plural_last = f"{last}s"
        plural = " ".join([*tokens[:-1], plural_last])
        candidates.update({plural, plural.replace(" ", "_"), plural.replace(" ", "-")})
    return {candidate for candidate in candidates if candidate}


def _prediction_rows(
    db: Session,
    *,
    dt_from: str | None = None,
    dt_to: str | None = None,
    domain: str | None = None,
    aspect: str | None = None,
    sentiment: str | None = None,
    limit: int | None = None,
) -> list[tuple[Prediction, Review]]:
    f = parse_dt(dt_from)
    t = parse_dt(dt_to)
    q = (
        db.query(Prediction, Review)
        .options(selectinload(Prediction.evidence_spans))
        .join(Review, Review.id == Prediction.review_id)
    )
    if domain:
        q = q.filter(Review.domain == domain)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)
    if sentiment:
        q = q.filter(Prediction.sentiment == sentiment)
    if aspect:
        candidates = _aspect_filter_candidates(aspect)
        if candidates:
            q = q.filter(
                or_(
                    Prediction.aspect_canonical.in_(candidates),
                    Prediction.aspect_normalized.in_(candidates),
                    Prediction.aspect_cluster.in_(candidates),
                    Prediction.aspect_raw.in_(candidates),
                )
            )
    q = q.order_by(Review.created_at.desc(), Prediction.id.desc())

    rows = q.all()
    if aspect:
        target = aspect_key(aspect)
        rows = [(pred, review) for pred, review in rows if canonical_aspect(pred) == target]
    if limit is not None:
        rows = rows[: max(1, int(limit))]
    return rows


def _bucket_for(created_at: datetime | None, interval: str) -> str:
    value = created_at or datetime.utcnow()
    if interval == "week":
        year, week, _ = value.isocalendar()
        return f"{year}-W{week:02d}"
    return value.strftime("%Y-%m-%d")


def _distribution(rows: list[tuple[Prediction, Review]]) -> dict[str, Counter]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for pred, _ in rows:
        grouped[canonical_aspect(pred)][str(pred.sentiment or "neutral")] += 1
    return grouped


def aspect_leaderboard(db: Session, limit: int = 25, domain: Optional[str] = None) -> list[dict]:
    now = datetime.utcnow()
    current_start = now - timedelta(days=7)
    previous_start = now - timedelta(days=14)

    current_rows = _prediction_rows(db, dt_from=current_start.isoformat(), dt_to=now.isoformat(), domain=domain)
    previous_rows = _prediction_rows(db, dt_from=previous_start.isoformat(), dt_to=current_start.isoformat(), domain=domain)

    current_dist = _distribution(current_rows)
    previous_freq = Counter(canonical_aspect(pred) for pred, _ in previous_rows)

    sample_size_q = db.query(func.count(Review.id)).filter(Review.created_at >= current_start, Review.created_at < now)
    if domain:
        sample_size_q = sample_size_q.filter(Review.domain == domain)
    sample_size = int(sample_size_q.scalar() or 0)

    origin_counts: dict[str, Counter] = defaultdict(Counter)
    for pred, _ in current_rows:
        aspect = canonical_aspect(pred)
        origin_counts[aspect][prediction_origin(pred, _first_snippet(pred))] += 1

    out: list[dict] = []
    aspects = sorted(
        current_dist.keys(),
        key=lambda aspect: (-(sum(current_dist[aspect].values())), aspect),
    )[: max(1, int(limit))]
    for aspect in aspects:
        counts = current_dist[aspect]
        current_freq = int(sum(counts.values()))
        previous_count = int(previous_freq.get(aspect, 0))
        positive = int(counts.get("positive", 0))
        neutral = int(counts.get("neutral", 0))
        negative = int(counts.get("negative", 0))
        total = max(0, positive + neutral + negative)
        pos_pct = _pct(positive, total)
        neu_pct = _pct(neutral, total)
        neg_pct = _pct(negative, total)
        net_sentiment = pos_pct - neg_pct

        if previous_count <= 0:
            change_vs_previous_period = 100.0 if current_freq else 0.0
        else:
            change_vs_previous_period = ((current_freq - previous_count) / previous_count) * 100.0

        p_hat = (negative / total) if total > 0 else 0.0
        ci_lo, ci_hi = _wilson_ci_95(p_hat, total)
        implicit = int(origin_counts[aspect].get("implicit", 0))
        explicit = int(origin_counts[aspect].get("explicit", 0))

        out.append(
            {
                "aspect": aspect,
                "frequency": current_freq,
                "sample_size": sample_size,
                "mentions_per_100_reviews": round(_pct(current_freq, sample_size), 4),
                "positive_pct": round(pos_pct, 2),
                "neutral_pct": round(neu_pct, 2),
                "negative_pct": round(neg_pct, 2),
                "net_sentiment": round(net_sentiment, 2),
                "change_vs_previous_period": round(change_vs_previous_period, 2),
                "change_7d_vs_prev_7d": round(change_vs_previous_period, 2),
                "negative_ci_low": round(ci_lo * 100.0, 2),
                "negative_ci_high": round(ci_hi * 100.0, 2),
                "implicit_pct": round(_pct(implicit, implicit + explicit), 2),
            }
        )

    out.sort(key=lambda row: (-row["frequency"], -row["change_7d_vs_prev_7d"], row["aspect"]))
    return out[: max(1, int(limit))]


def top_aspects(db: Session, limit: int, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    rows = _prediction_rows(db, dt_from=dt_from, dt_to=dt_to, domain=domain)
    counts = Counter(canonical_aspect(pred) for pred, _ in rows)
    return [
        {"aspect": aspect, "count": int(count)}
        for aspect, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(1, int(limit))]
    ]


def aspect_sentiment_distribution(db: Session, limit: int, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    rows = _prediction_rows(db, dt_from=dt_from, dt_to=dt_to, domain=domain)
    grouped = _distribution(rows)
    out = []
    for aspect, counts in sorted(grouped.items(), key=lambda item: (-(sum(item[1].values())), item[0]))[: max(1, int(limit))]:
        out.append(
            {
                "aspect": aspect,
                "positive": int(counts.get("positive", 0)),
                "neutral": int(counts.get("neutral", 0)),
                "negative": int(counts.get("negative", 0)),
            }
        )
    return out


def trends(db: Session, interval: str, aspect: Optional[str], dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    rows = _prediction_rows(db, dt_from=dt_from, dt_to=dt_to, domain=domain, aspect=aspect)
    buckets: dict[str, Counter] = defaultdict(Counter)
    for pred, review in rows:
        buckets[_bucket_for(review.created_at, interval)][str(pred.sentiment or "neutral")] += 1

    out = []
    for bucket in sorted(buckets):
        counts = buckets[bucket]
        mentions = int(sum(counts.values()))
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neg_pct = (neg / mentions) if mentions > 0 else 0.0
        pos_pct = (pos / mentions) if mentions > 0 else 0.0
        out.append(
            {
                "bucket": bucket,
                "mentions": mentions,
                "negative_pct": round(neg_pct, 4),
                "sentiment_score": round(pos_pct - neg_pct, 4),
            }
        )
    return out


def evidence_drilldown(db: Session, aspect: Optional[str] = None, sentiment: Optional[str] = None, limit: int = 50, domain: Optional[str] = None) -> list[dict]:
    rows = _prediction_rows(db, domain=domain, aspect=aspect, sentiment=sentiment, limit=max(1, min(limit, 200)))
    out = []
    for pred, review in rows:
        span = min(pred.evidence_spans, key=lambda item: (item.start_char, item.end_char), default=None)
        snippet = span.snippet if span else None
        out.append(
            {
                "review_id": review.id,
                "review_text": review.text,
                "aspect": canonical_aspect(pred),
                "sentiment": pred.sentiment,
                "origin": prediction_origin(pred, snippet),
                "evidence": snippet,
                "evidence_start": span.start_char if span else None,
                "evidence_end": span.end_char if span else None,
                "created_at": review.created_at.isoformat() if review.created_at else None,
            }
        )
    return out


def aspect_trends(db: Session, interval: str = "day", domain: Optional[str] = None, limit: int = 200) -> list[dict]:
    out = []
    top = top_aspects(db, 12, None, None, domain)
    for row in top:
        aspect = row["aspect"]
        points = trends(db, interval, aspect, None, None, domain)
        for point in points:
            out.append({"bucket": point["bucket"], "aspect": aspect, "mentions": point["mentions"], "negative_pct": point["negative_pct"]})
    return out[: max(1, limit)]


def emerging_aspects(db: Session, interval: str = "day", lookback_buckets: int = 7, domain: Optional[str] = None) -> list[dict]:
    tr = aspect_trends(db, interval=interval, domain=domain, limit=5000)
    grouped = defaultdict(list)
    for item in tr:
        grouped[item["aspect"]].append(item)
    out = []
    for aspect, points in grouped.items():
        points_sorted = sorted(points, key=lambda item: item["bucket"])
        if len(points_sorted) < 2:
            continue
        recent = points_sorted[-1]["mentions"]
        baseline_points = points_sorted[-(lookback_buckets + 1):-1]
        if not baseline_points:
            continue
        baseline = sum(p["mentions"] for p in baseline_points) / len(baseline_points)
        if recent >= 3 and recent > baseline * 1.5:
            out.append({"aspect": aspect, "recent_mentions": recent, "baseline_mentions": round(baseline, 2)})
    out.sort(key=lambda x: (-(x["recent_mentions"] - x["baseline_mentions"]), x["aspect"]))
    return out


def aspect_detail(db: Session, aspect: str, interval: str = "day", domain: Optional[str] = None) -> dict:
    canonical = aspect_key(aspect)
    rows = _prediction_rows(db, domain=domain, aspect=canonical)
    if not rows:
        return {"aspect": canonical, "frequency": 0, "positive": 0, "neutral": 0, "negative": 0, "explicit_count": 0, "implicit_count": 0, "connected_aspects": [], "trend": [], "examples": []}

    counts = Counter(str(pred.sentiment or "neutral") for pred, _ in rows)
    origin_counts = Counter(prediction_origin(pred, _first_snippet(pred)) for pred, _ in rows)

    review_ids = {review.id for _, review in rows}
    peer_counts: Counter = Counter()
    if review_ids:
        peer_rows = (
            db.query(Prediction)
            .options(selectinload(Prediction.evidence_spans))
            .filter(Prediction.review_id.in_(review_ids))
            .all()
        )
        for pred in peer_rows:
            peer = canonical_aspect(pred)
            if peer != canonical:
                peer_counts[peer] += 1

    connected = [
        {"aspect": peer, "weight": int(weight)}
        for peer, weight in sorted(peer_counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    ]
    trend = [
        {"bucket": point["bucket"], "aspect": canonical, "mentions": point["mentions"], "negative_pct": point["negative_pct"]}
        for point in trends(db, interval, canonical, None, None, domain)
    ]
    return {
        "aspect": canonical,
        "frequency": int(sum(counts.values())),
        "positive": int(counts.get("positive", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "negative": int(counts.get("negative", 0)),
        "explicit_count": int(origin_counts.get("explicit", 0)),
        "implicit_count": int(origin_counts.get("implicit", 0)),
        "connected_aspects": connected,
        "trend": trend,
        "examples": evidence_drilldown(db, aspect=canonical, limit=8, domain=domain),
    }
