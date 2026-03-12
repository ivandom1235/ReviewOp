# proto/backend/services/analytics.py
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from math import sqrt
from statistics import pstdev
from typing import Optional, List, Tuple

from sqlalchemy import func, case, text
from sqlalchemy.orm import Session

from models.tables import Review, Prediction
from services.graph_builders import _infer_origin


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # Accept YYYY-MM-DD or ISO
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None


def overview(db: Session, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]) -> dict:
    f = _parse_dt(dt_from)
    t = _parse_dt(dt_to)

    q_reviews = db.query(func.count(Review.id))
    q_preds = db.query(func.count(Prediction.id))
    q_unique = db.query(func.count(func.distinct(Prediction.aspect_raw)))
    q_avg_conf = db.query(func.avg(Prediction.confidence))

    if domain:
        q_reviews = q_reviews.filter(Review.domain == domain)

    if f:
        q_reviews = q_reviews.filter(Review.created_at >= f)
        q_preds = q_preds.join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= f)
        q_unique = q_unique.join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= f)
        q_avg_conf = q_avg_conf.join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= f)

    if t:
        q_reviews = q_reviews.filter(Review.created_at <= t)
        q_preds = q_preds.join(Review, Review.id == Prediction.review_id).filter(Review.created_at <= t)
        q_unique = q_unique.join(Review, Review.id == Prediction.review_id).filter(Review.created_at <= t)
        q_avg_conf = q_avg_conf.join(Review, Review.id == Prediction.review_id).filter(Review.created_at <= t)

    total_reviews = int(q_reviews.scalar() or 0)
    total_mentions = int(q_preds.scalar() or 0)
    unique_aspects = int(q_unique.scalar() or 0)
    avg_conf = float(q_avg_conf.scalar() or 0.0)

    # sentiment counts
    q_sent = db.query(
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("pos"),
        func.sum(case((Prediction.sentiment == "neutral", 1), else_=0)).label("neu"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg"),
    ).join(Review, Review.id == Prediction.review_id)

    if domain:
        q_sent = q_sent.filter(Review.domain == domain)
    if f:
        q_sent = q_sent.filter(Review.created_at >= f)
    if t:
        q_sent = q_sent.filter(Review.created_at <= t)

    row = q_sent.first()
    sentiment_counts = {
        "positive": int(row.pos or 0),
        "neutral": int(row.neu or 0),
        "negative": int(row.neg or 0),
    }

    return {
        "total_reviews": total_reviews,
        "total_aspect_mentions": total_mentions,
        "unique_aspects_raw": unique_aspects,
        "avg_confidence": round(avg_conf, 4),
        "sentiment_counts": sentiment_counts,
    }


def top_aspects(db: Session, limit: int, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    f = _parse_dt(dt_from)
    t = _parse_dt(dt_to)

    q = db.query(
        Prediction.aspect_raw.label("aspect"),
        func.count(Prediction.id).label("count"),
    ).join(Review, Review.id == Prediction.review_id)

    if domain:
        q = q.filter(Review.domain == domain)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)

    q = q.group_by(Prediction.aspect_raw).order_by(text("count DESC")).limit(limit)
    return [{"aspect": r.aspect, "count": int(r.count)} for r in q.all()]


def aspect_sentiment_distribution(db: Session, limit: int, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    f = _parse_dt(dt_from)
    t = _parse_dt(dt_to)

    q = db.query(
        Prediction.aspect_raw.label("aspect"),
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("positive"),
        func.sum(case((Prediction.sentiment == "neutral", 1), else_=0)).label("neutral"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("negative"),
        func.count(Prediction.id).label("total"),
    ).join(Review, Review.id == Prediction.review_id)

    if domain:
        q = q.filter(Review.domain == domain)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)

    q = q.group_by(Prediction.aspect_raw).order_by(text("total DESC")).limit(limit)

    out = []
    for r in q.all():
        out.append(
            {
                "aspect": r.aspect,
                "positive": int(r.positive or 0),
                "neutral": int(r.neutral or 0),
                "negative": int(r.negative or 0),
            }
        )
    return out


def trends(db: Session, interval: str, aspect: Optional[str], dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    """
    interval: day|week
    sentiment_score = positive_pct - negative_pct
    """
    f = _parse_dt(dt_from)
    t = _parse_dt(dt_to)

    # MySQL bucket formatting
    if interval == "week":
        bucket_expr = func.date_format(Review.created_at, "%x-W%v")  # ISO week
    else:
        bucket_expr = func.date_format(Review.created_at, "%Y-%m-%d")

    q = db.query(
        bucket_expr.label("bucket"),
        func.count(Prediction.id).label("mentions"),
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("pos"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg"),
    ).join(Review, Review.id == Prediction.review_id)

    if domain:
        q = q.filter(Review.domain == domain)
    if aspect:
        q = q.filter(Prediction.aspect_raw == aspect)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)

    q = q.group_by(bucket_expr).order_by(bucket_expr.asc())

    out = []
    for r in q.all():
        mentions = int(r.mentions or 0)
        pos = int(r.pos or 0)
        neg = int(r.neg or 0)
        neg_pct = (neg / mentions) if mentions > 0 else 0.0
        pos_pct = (pos / mentions) if mentions > 0 else 0.0
        score = pos_pct - neg_pct
        out.append(
            {
                "bucket": str(r.bucket),
                "mentions": mentions,
                "negative_pct": round(neg_pct, 4),
                "sentiment_score": round(score, 4),
            }
        )
    return out


def dashboard_kpis(db: Session, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]) -> dict:
    base = overview(db, dt_from, dt_to, domain)
    total_mentions = int(base["total_aspect_mentions"] or 0)
    negative_count = int(base["sentiment_counts"]["negative"] or 0)
    negative_sentiment_pct = round((negative_count / total_mentions) * 100, 2) if total_mentions else 0.0

    rows = aspect_sentiment_distribution(db, 100, dt_from, dt_to, domain)
    most_negative_aspect = None
    if rows:
        ranked = sorted(rows, key=lambda r: (-(r["negative"] / max((r["positive"] + r["neutral"] + r["negative"]), 1)), -r["negative"], r["aspect"]))
        most_negative_aspect = ranked[0]["aspect"]

    emerging = emerging_aspects(db, interval="day", lookback_buckets=7, domain=domain)
    return {
        "total_reviews": base["total_reviews"],
        "total_aspects": base["total_aspect_mentions"],
        "most_negative_aspect": most_negative_aspect,
        "negative_sentiment_pct": negative_sentiment_pct,
        "emerging_issues_count": len(emerging),
    }


def _period_bounds(days: int = 30) -> tuple[datetime, datetime, datetime, datetime]:
    now = datetime.utcnow()
    current_start = now - timedelta(days=days)
    previous_end = current_start
    previous_start = previous_end - timedelta(days=days)
    return previous_start, previous_end, current_start, now


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + (z * z / n)
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def aspect_leaderboard(db: Session, limit: int = 25, domain: Optional[str] = None) -> list[dict]:
    prev_start, prev_end, cur_start, cur_end = _period_bounds(30)

    current_q = db.query(
        Prediction.aspect_raw.label("aspect"),
        func.count(Prediction.id).label("frequency"),
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("positive"),
        func.sum(case((Prediction.sentiment == "neutral", 1), else_=0)).label("neutral"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("negative"),
    ).join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= cur_start, Review.created_at <= cur_end)

    prev_q = db.query(
        Prediction.aspect_raw.label("aspect"),
        func.count(Prediction.id).label("frequency"),
    ).join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= prev_start, Review.created_at < prev_end)

    if domain:
        current_q = current_q.filter(Review.domain == domain)
        prev_q = prev_q.filter(Review.domain == domain)

    current_rows = current_q.group_by(Prediction.aspect_raw).all()
    prev_rows = prev_q.group_by(Prediction.aspect_raw).all()
    prev_counts = {r.aspect: int(r.frequency or 0) for r in prev_rows}

    implicit_counts = defaultdict(int)
    explicit_counts = defaultdict(int)
    sample_preds = db.query(Prediction).join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= cur_start, Review.created_at <= cur_end)
    if domain:
        sample_preds = sample_preds.filter(Review.domain == domain)
    for pred in sample_preds.limit(5000):
        span = pred.evidence_spans[0].snippet if pred.evidence_spans else None
        origin = _infer_origin(pred.aspect_raw, span)
        if origin == "implicit":
            implicit_counts[pred.aspect_raw] += 1
        else:
            explicit_counts[pred.aspect_raw] += 1

    out = []
    q_reviews = db.query(func.count(Review.id))
    if domain:
        q_reviews = q_reviews.filter(Review.domain == domain)
    q_reviews = q_reviews.filter(Review.created_at >= cur_start, Review.created_at <= cur_end)
    current_reviews = int(q_reviews.scalar() or 0)

    p7_start = cur_end - timedelta(days=7)
    p14_start = p7_start - timedelta(days=7)
    week_current = db.query(Prediction.aspect_raw.label("aspect"), func.count(Prediction.id).label("c")).join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= p7_start, Review.created_at <= cur_end)
    week_prev = db.query(Prediction.aspect_raw.label("aspect"), func.count(Prediction.id).label("c")).join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= p14_start, Review.created_at < p7_start)
    if domain:
        week_current = week_current.filter(Review.domain == domain)
        week_prev = week_prev.filter(Review.domain == domain)
    week_current_map = {r.aspect: int(r.c or 0) for r in week_current.group_by(Prediction.aspect_raw).all()}
    week_prev_map = {r.aspect: int(r.c or 0) for r in week_prev.group_by(Prediction.aspect_raw).all()}

    for row in current_rows:
        freq = int(row.frequency or 0)
        pos = int(row.positive or 0)
        neu = int(row.neutral or 0)
        neg = int(row.negative or 0)
        total = max(freq, 1)
        prev = prev_counts.get(row.aspect, 0)
        denom = prev if prev > 0 else 1
        change = ((freq - prev) / denom) * 100
        c7 = week_current_map.get(row.aspect, 0)
        p7 = week_prev_map.get(row.aspect, 0)
        change_7d = ((c7 - p7) / (p7 if p7 > 0 else 1)) * 100
        ci_low, ci_high = _wilson_ci(neg, total)
        impl = implicit_counts.get(row.aspect, 0)
        expl = explicit_counts.get(row.aspect, 0)
        impl_pct = (impl / max(impl + expl, 1)) * 100
        out.append({
            "aspect": row.aspect,
            "frequency": freq,
            "sample_size": total,
            "mentions_per_100_reviews": round((freq * 100.0 / current_reviews), 2) if current_reviews else 0.0,
            "positive_pct": round((pos / total) * 100, 2),
            "neutral_pct": round((neu / total) * 100, 2),
            "negative_pct": round((neg / total) * 100, 2),
            "net_sentiment": round(((pos - neg) / total), 4),
            "change_vs_previous_period": round(change, 2),
            "change_7d_vs_prev_7d": round(change_7d, 2),
            "negative_ci_low": round(ci_low * 100, 2),
            "negative_ci_high": round(ci_high * 100, 2),
            "implicit_pct": round(impl_pct, 2),
        })

    out.sort(key=lambda item: (-item["frequency"], item["aspect"]))
    return out[: max(1, limit)]


def evidence_drilldown(db: Session, aspect: Optional[str] = None, sentiment: Optional[str] = None, limit: int = 50, domain: Optional[str] = None) -> list[dict]:
    q = db.query(Prediction, Review).join(Review, Review.id == Prediction.review_id)
    if aspect:
        q = q.filter(Prediction.aspect_raw == aspect)
    if sentiment:
        q = q.filter(Prediction.sentiment == sentiment)
    if domain:
        q = q.filter(Review.domain == domain)
    q = q.order_by(Review.created_at.desc()).limit(max(1, min(limit, 200)))

    rows = []
    for pred, review in q.all():
        span = pred.evidence_spans[0] if pred.evidence_spans else None
        snippet = span.snippet if span else None
        rows.append({
            "review_id": review.id,
            "review_text": review.text,
            "aspect": pred.aspect_raw,
            "sentiment": pred.sentiment,
            "origin": _infer_origin(pred.aspect_raw, snippet),
            "evidence": snippet,
            "evidence_start": span.start_char if span else None,
            "evidence_end": span.end_char if span else None,
            "created_at": review.created_at.isoformat() if review.created_at else None,
        })
    return rows


def aspect_trends(db: Session, interval: str = "day", domain: Optional[str] = None, limit: int = 200) -> list[dict]:
    out = []
    top = top_aspects(db, 12, None, None, domain)
    for row in top:
        aspect = row["aspect"]
        points = trends(db, interval, aspect, None, None, domain)
        for point in points:
            out.append({
                "bucket": point["bucket"],
                "aspect": aspect,
                "mentions": point["mentions"],
                "negative_pct": point["negative_pct"],
            })
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
    dist_rows = aspect_sentiment_distribution(db, 500, None, None, domain)
    dist = next((item for item in dist_rows if item["aspect"] == aspect), None)
    if dist is None:
        return {
            "aspect": aspect,
            "frequency": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "explicit_count": 0,
            "implicit_count": 0,
            "connected_aspects": [],
            "trend": [],
            "examples": [],
        }

    examples = evidence_drilldown(db, aspect=aspect, limit=8, domain=domain)
    explicit_count = sum(1 for item in examples if item["origin"] == "explicit")
    implicit_count = sum(1 for item in examples if item["origin"] == "implicit")

    connections = defaultdict(int)
    reviews = db.query(Review).join(Prediction, Prediction.review_id == Review.id).filter(Prediction.aspect_raw == aspect)
    if domain:
        reviews = reviews.filter(Review.domain == domain)
    for review in reviews.limit(400).all():
        others = {p.aspect_raw for p in review.predictions if p.aspect_raw != aspect}
        for other in others:
            connections[other] += 1

    connected = [{"aspect": key, "weight": val} for key, val in sorted(connections.items(), key=lambda item: (-item[1], item[0]))[:12]]
    trend = aspect_trends(db, interval=interval, domain=domain, limit=5000)
    trend = [t for t in trend if t["aspect"] == aspect]
    return {
        "aspect": aspect,
        "frequency": int(dist["positive"] + dist["neutral"] + dist["negative"]),
        "positive": int(dist["positive"]),
        "neutral": int(dist["neutral"]),
        "negative": int(dist["negative"]),
        "explicit_count": explicit_count,
        "implicit_count": implicit_count,
        "connected_aspects": connected,
        "trend": trend,
        "examples": examples,
    }


def alerts(db: Session, domain: Optional[str] = None) -> list[dict]:
    out = []
    leaderboard = aspect_leaderboard(db, limit=20, domain=domain)
    change_series = [float(row["change_7d_vs_prev_7d"]) for row in leaderboard]
    sigma = pstdev(change_series) if len(change_series) > 1 else 0.0
    mean_change = sum(change_series) / len(change_series) if change_series else 0.0

    for row in leaderboard:
        z_score = (row["change_7d_vs_prev_7d"] - mean_change) / sigma if sigma > 0 else 0.0
        if (row["change_vs_previous_period"] >= 50 or z_score >= 2.0) and row["frequency"] >= 5:
            out.append({
                "type": "frequency_spike",
                "aspect": row["aspect"],
                "severity": "high",
                "message": f"Frequency spike detected for {row['aspect']}",
                "value": round(max(row["change_vs_previous_period"], z_score), 2),
                "threshold": 2.0 if z_score >= 2.0 else 50.0,
            })
        if row["negative_pct"] >= 45 and row["frequency"] >= 5:
            out.append({
                "type": "negative_threshold",
                "aspect": row["aspect"],
                "severity": "medium",
                "message": f"Negative sentiment threshold breached for {row['aspect']}",
                "value": row["negative_pct"],
                "threshold": 45.0,
            })

    for item in emerging_aspects(db, interval="day", domain=domain):
        out.append({
            "type": "emerging_aspect",
            "aspect": item["aspect"],
            "severity": "medium",
            "message": f"Emerging issue detected: {item['aspect']}",
            "value": float(item["recent_mentions"]),
            "threshold": float(item["baseline_mentions"]),
        })
    return out[:50]


def impact_matrix(db: Session, domain: Optional[str] = None, limit: int = 20) -> list[dict]:
    rows = aspect_leaderboard(db, limit=max(limit, 30), domain=domain)
    out = []
    for row in rows:
        negative_rate = float(row["negative_pct"]) / 100.0
        growth_factor = 1 + max(0.0, float(row["change_7d_vs_prev_7d"]) / 100.0)
        priority = float(row["frequency"]) * negative_rate * growth_factor
        tier = "high" if priority >= 8 else "medium" if priority >= 3 else "low"
        out.append({
            "aspect": row["aspect"],
            "volume": int(row["frequency"]),
            "negative_rate": round(negative_rate, 4),
            "growth_pct": float(row["change_7d_vs_prev_7d"]),
            "priority_score": round(priority, 3),
            "action_tier": tier,
        })
    out.sort(key=lambda item: (-item["priority_score"], item["aspect"]))
    return out[: max(1, limit)]


def segment_drilldown(db: Session, domain: Optional[str] = None, limit: int = 20) -> list[dict]:
    out = []
    domain_q = db.query(
        Review.domain.label("segment"),
        func.count(func.distinct(Review.id)).label("reviews"),
        func.count(Prediction.id).label("mentions"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg"),
    ).join(Prediction, Prediction.review_id == Review.id).group_by(Review.domain)
    if domain:
        domain_q = domain_q.filter(Review.domain == domain)
    for row in domain_q.all():
        mentions = int(row.mentions or 0)
        neg = int(row.neg or 0)
        top_neg = db.query(Prediction.aspect_raw, func.count(Prediction.id).label("c")).join(Review, Review.id == Prediction.review_id).filter(Review.domain == row.segment, Prediction.sentiment == "negative").group_by(Prediction.aspect_raw).order_by(text("c DESC")).first()
        out.append({
            "segment_type": "domain",
            "segment_value": row.segment or "unknown",
            "review_count": int(row.reviews or 0),
            "mention_count": mentions,
            "negative_pct": round((neg / mentions) * 100, 2) if mentions else 0.0,
            "top_negative_aspect": top_neg[0] if top_neg else None,
        })

    product_q = db.query(
        Review.product_id.label("segment"),
        func.count(func.distinct(Review.id)).label("reviews"),
        func.count(Prediction.id).label("mentions"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg"),
    ).join(Prediction, Prediction.review_id == Review.id).filter(Review.product_id.isnot(None)).group_by(Review.product_id).order_by(text("mentions DESC")).limit(limit)
    if domain:
        product_q = product_q.filter(Review.domain == domain)
    for row in product_q.all():
        mentions = int(row.mentions or 0)
        neg = int(row.neg or 0)
        out.append({
            "segment_type": "product_id",
            "segment_value": row.segment or "unknown",
            "review_count": int(row.reviews or 0),
            "mention_count": mentions,
            "negative_pct": round((neg / mentions) * 100, 2) if mentions else 0.0,
            "top_negative_aspect": None,
        })

    out.sort(key=lambda item: (-item["negative_pct"], -item["mention_count"], item["segment_type"], item["segment_value"]))
    return out[: max(1, limit)]


def weekly_summary(db: Session, domain: Optional[str] = None) -> dict:
    now = datetime.utcnow()
    this_week_start = now - timedelta(days=7)
    prev_week_start = now - timedelta(days=14)

    current = aspect_leaderboard(db, limit=12, domain=domain)
    impact = impact_matrix(db, domain=domain, limit=3)
    emerging = emerging_aspects(db, interval="day", lookback_buckets=7, domain=domain)

    biggest = max(current, key=lambda row: row["change_7d_vs_prev_7d"], default=None)
    recommendations = [
        f"Prioritize remediation for {row['aspect']} (score {row['priority_score']})" for row in impact
    ]
    return {
        "period_label": f"{this_week_start.date()} to {now.date()} vs {prev_week_start.date()} to {this_week_start.date()}",
        "top_drivers": [row["aspect"] for row in impact],
        "biggest_increase_aspect": biggest["aspect"] if biggest else None,
        "biggest_increase_pct": float(biggest["change_7d_vs_prev_7d"]) if biggest else 0.0,
        "emerging_count": len(emerging),
        "action_recommendations": recommendations,
    }
