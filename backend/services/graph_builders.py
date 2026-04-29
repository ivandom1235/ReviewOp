from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from typing import Optional

from sqlalchemy.orm import Session, selectinload

from models.tables import NovelCandidate, Prediction, ProductCatalog, Review, UserProductReview
from services.analytics_common import aspect_label, canonical_aspect, parse_dt, prediction_origin


SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
MAX_BATCH_GRAPH_REVIEWS = 2000
MAX_BATCH_GRAPH_NOVEL_ROWS = 5000


def _clean_filter_value(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def _dominant_sentiment(counter: Counter) -> str:
    if not counter:
        return "neutral"
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _polarity_hint(source_sentiment: str | None, target_sentiment: str | None) -> str:
    src = (source_sentiment or "neutral").lower()
    dst = (target_sentiment or "neutral").lower()
    if src == dst:
        return src
    if "negative" in {src, dst} and "positive" in {src, dst}:
        return "mixed"
    if "negative" in {src, dst}:
        return "negative"
    if "positive" in {src, dst}:
        return "positive"
    return "neutral"


def _prediction_origin(prediction: Prediction, snippet: str | None) -> str:
    return prediction_origin(prediction, snippet)


def build_graph_filter_options(db: Session) -> dict:
    domain_rows = (
        db.query(Review.domain)
        .filter(Review.domain.isnot(None))
        .distinct()
        .order_by(Review.domain.asc())
        .all()
    )
    review_product_rows = (
        db.query(Review.product_id)
        .filter(Review.product_id.isnot(None))
        .filter(Review.product_id != "")
        .distinct()
        .order_by(Review.product_id.asc())
        .all()
    )
    user_review_product_rows = (
        db.query(UserProductReview.product_id)
        .filter(UserProductReview.product_id.isnot(None))
        .filter(UserProductReview.product_id != "")
        .distinct()
        .order_by(UserProductReview.product_id.asc())
        .all()
    )
    catalog_product_rows = (
        db.query(ProductCatalog.product_id)
        .filter(ProductCatalog.product_id.isnot(None))
        .filter(ProductCatalog.product_id != "")
        .distinct()
        .order_by(ProductCatalog.product_id.asc())
        .all()
    )

    domains = [str(row[0]).strip() for row in domain_rows if row and row[0] and str(row[0]).strip()]
    product_ids: list[str] = []
    for rows in (review_product_rows, user_review_product_rows, catalog_product_rows):
        for row in rows:
            if row and row[0] and str(row[0]).strip():
                product_ids.append(str(row[0]).strip())
    product_ids = list(dict.fromkeys(product_ids))
    return {"domains": domains, "product_ids": product_ids}


def build_single_review_graph(db: Session, review_id: int) -> dict | None:
    review = (
        db.query(Review)
        .options(selectinload(Review.predictions).selectinload(Prediction.evidence_spans))
        .filter(Review.id == review_id)
        .first()
    )
    if not review:
        return None

    aspect_nodes: dict[str, dict] = {}
    ordering: list[tuple[int, str]] = []

    for prediction in review.predictions:
        aspect_id = canonical_aspect(prediction)
        span = min(
            prediction.evidence_spans,
            key=lambda item: (item.start_char, item.end_char),
            default=None,
        )
        start_char = span.start_char if span else len(review.text or "")
        end_char = span.end_char if span else start_char
        snippet = span.snippet if span else None
        origin = _prediction_origin(prediction, snippet)

        current = aspect_nodes.get(aspect_id)
        if current is None:
            aspect_nodes[aspect_id] = {
                "id": aspect_id,
                "label": aspect_label(aspect_id),
                "sentiment": prediction.sentiment,
                "confidence": float(prediction.confidence or 0.0),
                "explicit_count": 1 if origin == "explicit" else 0,
                "implicit_count": 1 if origin == "implicit" else 0,
                "evidence": snippet,
                "evidence_start": start_char if span else None,
                "evidence_end": end_char if span else None,
                "origin": origin,
                "_confidence_total": float(prediction.confidence or 0.0),
                "_mentions": 1,
                "_sentiments": Counter([prediction.sentiment]),
            }
        else:
            current["_confidence_total"] += float(prediction.confidence or 0.0)
            current["_mentions"] += 1
            current["_sentiments"][prediction.sentiment] += 1
            current["explicit_count"] += 1 if origin == "explicit" else 0
            current["implicit_count"] += 1 if origin == "implicit" else 0
            if start_char < (current.get("evidence_start") if current.get("evidence_start") is not None else 10**9):
                current["evidence"] = snippet
                current["evidence_start"] = start_char
                current["evidence_end"] = end_char

        ordering.append((start_char, aspect_id))

    nodes = []
    for aspect_id, node in aspect_nodes.items():
        dominant = _dominant_sentiment(node["_sentiments"])
        explicit_count = int(node["explicit_count"])
        implicit_count = int(node["implicit_count"])
        origin = "explicit" if explicit_count and not implicit_count else "implicit" if implicit_count and not explicit_count else "mixed"
        nodes.append(
            {
                "id": aspect_id,
                "label": node["label"],
                "sentiment": dominant,
                "confidence": round(node["_confidence_total"] / max(node["_mentions"], 1), 4),
                "explicit_count": explicit_count,
                "implicit_count": implicit_count,
                "evidence": node.get("evidence"),
                "evidence_start": node.get("evidence_start"),
                "evidence_end": node.get("evidence_end"),
                "origin": origin,
            }
        )

    ordered_mentions = [aspect_id for _, aspect_id in sorted(ordering, key=lambda item: (item[0], item[1]))]
    first_position: dict[str, int] = {}
    for index, aspect_id in enumerate(ordered_mentions):
        first_position.setdefault(aspect_id, index)

    sentiment_lookup = {node["id"]: node["sentiment"] for node in nodes}
    transition_weights: defaultdict[tuple[str, str], int] = defaultdict(int)
    for source, target in zip(ordered_mentions, ordered_mentions[1:]):
        if source == target:
            continue
        transition_weights[(source, target)] += 1

    edges = []
    for (source, target), weight in sorted(
        transition_weights.items(),
        key=lambda item: (
            first_position.get(item[0][0], 10**9),
            first_position.get(item[0][1], 10**9),
            item[0][0],
            item[0][1],
        ),
    ):
        edges.append(
            {
                "source": source,
                "target": target,
                "type": "review_transition",
                "weight": float(weight),
                "pair_count": int(weight),
                "directional": True,
                "polarity_hint": _polarity_hint(sentiment_lookup.get(source), sentiment_lookup.get(target)),
            }
        )

    return {
        "scope": "single_review",
        "review_id": review.id,
        "generated_at": datetime.utcnow().isoformat(),
        "filters": {"time_bucket_ready": True},
        "nodes": nodes,
        "edges": edges,
    }


def build_batch_aspect_graph(
    db: Session,
    domain: str | None = None,
    product_id: str | None = None,
    dt_from: str | None = None,
    dt_to: str | None = None,
    min_edge_weight: int = 1,
    graph_mode: str = "accepted",
) -> dict:
    domain = _clean_filter_value(domain)
    product_id = _clean_filter_value(product_id)
    f = parse_dt(dt_from)
    t = parse_dt(dt_to)
    mode = (graph_mode or "accepted").strip().lower()

    if mode == "novel_side":
        return _build_batch_novel_graph(
            db=db,
            domain=domain,
            product_id=product_id,
            dt_from=dt_from,
            dt_to=dt_to,
            min_edge_weight=min_edge_weight,
            f=f,
            t=t,
        )

    reviews_query = (
        db.query(Review)
        .options(selectinload(Review.predictions).selectinload(Prediction.evidence_spans))
        .filter(Review.predictions.any())
    )

    if domain:
        reviews_query = reviews_query.filter(Review.domain == domain)
    if product_id:
        reviews_query = reviews_query.filter(Review.product_id == product_id)
    if f:
        reviews_query = reviews_query.filter(Review.created_at >= f)
    if t:
        reviews_query = reviews_query.filter(Review.created_at <= t)

    reviews = reviews_query.order_by(Review.created_at.desc(), Review.id.desc()).limit(MAX_BATCH_GRAPH_REVIEWS).all()

    node_stats: dict[str, dict] = {}
    edge_weights: defaultdict[tuple[str, str], int] = defaultdict(int)
    edge_examples: defaultdict[tuple[str, str], list[str]] = defaultdict(list)

    for review in reviews:
        review_aspects = set()
        for prediction in review.predictions:
            aspect_id = canonical_aspect(prediction)
            span = min(
                prediction.evidence_spans,
                key=lambda item: (item.start_char, item.end_char),
                default=None,
            )
            snippet = span.snippet if span else None
            origin = _prediction_origin(prediction, snippet)

            stats = node_stats.setdefault(
                aspect_id,
                {
                    "id": aspect_id,
                    "label": aspect_label(aspect_id),
                    "frequency": 0,
                    "explicit_count": 0,
                    "implicit_count": 0,
                    "_scores": [],
                    "_sentiments": Counter(),
                },
            )
            stats["_scores"].append(SENTIMENT_SCORE.get(prediction.sentiment, 0.0))
            stats["_sentiments"][prediction.sentiment] += 1
            stats["explicit_count"] += 1 if origin == "explicit" else 0
            stats["implicit_count"] += 1 if origin == "implicit" else 0
            review_aspects.add(aspect_id)

        for aspect_id in review_aspects:
            node_stats[aspect_id]["frequency"] += 1

        for source, target in combinations(sorted(review_aspects), 2):
            edge_weights[(source, target)] += 1
            if len(edge_examples[(source, target)]) < 3 and review.text:
                edge_examples[(source, target)].append(review.text[:220])

    nodes = []
    for aspect_id, stats in sorted(node_stats.items(), key=lambda item: (-item[1]["frequency"], item[0])):
        scores = stats["_scores"]
        dominant = _dominant_sentiment(stats["_sentiments"])
        negative_count = int(stats["_sentiments"].get("negative", 0))
        mention_count = sum(stats["_sentiments"].values())
        nodes.append(
            {
                "id": aspect_id,
                "label": stats["label"],
                "frequency": int(stats["frequency"]),
                "avg_sentiment": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "dominant_sentiment": dominant,
                "negative_ratio": round(negative_count / mention_count, 4) if mention_count else 0.0,
                "explicit_count": int(stats["explicit_count"]),
                "implicit_count": int(stats["implicit_count"]),
            }
        )

    dominant_lookup = {node["id"]: node["dominant_sentiment"] for node in nodes}
    edges = []
    for (source, target), weight in sorted(edge_weights.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        if weight < max(int(min_edge_weight or 1), 1):
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "type": "cooccurrence",
                "weight": float(weight),
                "directional": False,
                "pair_count": int(weight),
                "polarity_hint": _polarity_hint(dominant_lookup.get(source), dominant_lookup.get(target)),
                "example_reviews": edge_examples.get((source, target), []),
            }
        )

    return {
        "scope": "batch",
        "generated_at": datetime.utcnow().isoformat(),
        "filters": {
            "domain": domain,
            "product_id": product_id,
            "from": dt_from,
            "to": dt_to,
            "graph_mode": "accepted",
            "min_edge_weight": max(int(min_edge_weight or 1), 1),
            "time_bucket_ready": True,
            "review_limit": MAX_BATCH_GRAPH_REVIEWS,
            "truncated": len(reviews) >= MAX_BATCH_GRAPH_REVIEWS,
        },
        "nodes": nodes,
        "edges": edges,
    }


def _build_batch_novel_graph(
    db: Session,
    domain: str | None,
    product_id: str | None,
    dt_from: str | None,
    dt_to: str | None,
    min_edge_weight: int,
    f: datetime | None,
    t: datetime | None,
) -> dict:
    domain = _clean_filter_value(domain)
    product_id = _clean_filter_value(product_id)
    query = db.query(NovelCandidate, Review).join(Review, NovelCandidate.review_id == Review.id)
    if domain:
        query = query.filter(Review.domain == domain)
    if product_id:
        query = query.filter(Review.product_id == product_id)
    if f:
        query = query.filter(Review.created_at >= f)
    if t:
        query = query.filter(Review.created_at <= t)
    rows = query.order_by(Review.created_at.desc(), NovelCandidate.id.desc()).limit(MAX_BATCH_GRAPH_NOVEL_ROWS).all()

    aspects_by_review: dict[int, set[str]] = defaultdict(set)
    node_counts: Counter = Counter()
    novelty_sum: defaultdict[str, float] = defaultdict(float)
    edge_weights: defaultdict[tuple[str, str], int] = defaultdict(int)

    for novel_row, review in rows:
        aspect = (novel_row.aspect or "").strip()
        if not aspect:
            continue
        node_counts[aspect] += 1
        novelty_sum[aspect] += float(novel_row.novelty_score or 0.0)
        aspects_by_review[int(review.id)].add(aspect)

    for review_aspects in aspects_by_review.values():
        for source, target in combinations(sorted(review_aspects), 2):
            edge_weights[(source, target)] += 1

    nodes = []
    for aspect, count in sorted(node_counts.items(), key=lambda item: (-item[1], item[0])):
        nodes.append(
            {
                "id": aspect,
                "label": aspect_label(aspect),
                "frequency": int(count),
                "avg_sentiment": 0.0,
                "dominant_sentiment": "neutral",
                "negative_ratio": 0.0,
                "explicit_count": 0,
                "implicit_count": int(count),
                "origin": "novel_side",
                "confidence": round(novelty_sum[aspect] / max(count, 1), 4),
            }
        )

    edges = []
    for (source, target), weight in sorted(edge_weights.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        if weight < max(int(min_edge_weight or 1), 1):
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "type": "novel_cooccurrence",
                "weight": float(weight),
                "directional": False,
                "pair_count": int(weight),
                "polarity_hint": "neutral",
                "example_reviews": [],
            }
        )

    return {
        "scope": "batch",
        "generated_at": datetime.utcnow().isoformat(),
        "filters": {
            "domain": domain,
            "product_id": product_id,
            "from": dt_from,
            "to": dt_to,
            "graph_mode": "novel_side",
            "min_edge_weight": max(int(min_edge_weight or 1), 1),
            "time_bucket_ready": True,
        },
        "nodes": nodes,
        "edges": edges,
    }
