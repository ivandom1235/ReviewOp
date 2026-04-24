from __future__ import annotations
import json
import logging
import time
import threading

from sqlalchemy import delete
from sqlalchemy.orm import Session

from core.db import SessionLocal
from core.config import settings

from models.tables import EvidenceSpan, Prediction, Review
from services.evidence import find_evidence_for_aspect
from services.kg_build import KGBuilder, KGConfig
from services.open_aspect import extract_open_aspects

_GRAPH_REFRESH_THREADS: dict[str, threading.Thread] = {}
_GRAPH_REFRESH_LOCK = threading.Lock()


def _safe_extract_aspects(text: str, max_aspects: int = 8) -> list[str]:
    logger = logging.getLogger(__name__)
    try:
        aspects = extract_open_aspects(text, max_aspects=max_aspects)
    except Exception as exc:
        # Open-aspect extraction is optional; don't hard-fail inference if a local model is missing.
        logger.warning("Open-aspect extraction failed (%s); falling back to ['general']", type(exc).__name__)
        return ["general"]
    if aspects:
        return aspects
    return ["general"]


def run_single_review_pipeline(
    db: Session,
    *,
    engine,
    text: str,
    domain: str | None = None,
    product_id: str | None = None,
    review: Review | None = None,
    replace_existing: bool = False,
) -> Review:
    clean_text = (text or "").strip()
    if review is None:
        review = Review(text=clean_text, domain=domain, product_id=product_id)
        db.add(review)
        db.flush()
    else:
        review.text = clean_text
        review.domain = domain
        review.product_id = product_id
        if replace_existing:
            old_preds = db.query(Prediction).filter(Prediction.review_id == review.id).all()
            old_pred_ids = [pred.id for pred in old_preds if pred.id is not None]
            if old_pred_ids:
                db.execute(delete(EvidenceSpan).where(EvidenceSpan.prediction_id.in_(old_pred_ids)))
            for pred in old_preds:
                db.delete(pred)
            if old_preds:
                db.flush()
            db.expire(review, ["predictions"])

    aspects = _safe_extract_aspects(clean_text, max_aspects=8)

    for aspect_raw in aspects:
        start_char, end_char, snippet = find_evidence_for_aspect(clean_text, aspect_raw)
        sent, conf = engine.classify_sentiment_with_confidence(snippet, aspect_raw)

        pred = Prediction(
            aspect_raw=aspect_raw,
            aspect_cluster=aspect_raw,
            sentiment=sent,
            confidence=float(conf),
            rationale=None,
        )
        pred.review = review
        pred.evidence_spans.append(
            EvidenceSpan(
                start_char=start_char,
                end_char=end_char,
                snippet=snippet,
            )
        )
        db.add(pred)

    db.flush()
    return review


def run_single_review_pipeline_for_existing_review(
    db: Session,
    *,
    review: Review,
    engine,
    text: str,
    domain: str | None = None,
    product_id: str | None = None,
) -> Review:
    clean_text = (text or "").strip()
    review.text = clean_text
    review.domain = domain
    review.product_id = product_id

    old_preds = db.query(Prediction).filter(Prediction.review_id == review.id).all()
    old_pred_ids = [pred.id for pred in old_preds if pred.id is not None]
    if old_pred_ids:
        db.execute(delete(EvidenceSpan).where(EvidenceSpan.prediction_id.in_(old_pred_ids)))
    for pred in old_preds:
        db.delete(pred)
    if old_preds:
        db.flush()
    db.expire(review, ["predictions"])

    aspects = _safe_extract_aspects(clean_text, max_aspects=8)

    for aspect_raw in aspects:
        start_char, end_char, snippet = find_evidence_for_aspect(clean_text, aspect_raw)
        sent, conf = engine.classify_sentiment_with_confidence(snippet, aspect_raw)

        pred = Prediction(
            aspect_raw=aspect_raw,
            aspect_cluster=aspect_raw,
            sentiment=sent,
            confidence=float(conf),
            rationale=None,
        )
        pred.review = review
        pred.evidence_spans.append(
            EvidenceSpan(
                start_char=start_char,
                end_char=end_char,
                snippet=snippet,
            )
        )
        db.add(pred)

    db.flush()
    return review


def refresh_corpus_graph(db: Session, domain: str | None = None) -> dict:
    logger = logging.getLogger(__name__)
    started_at = time.perf_counter()
    logger.info("Refreshing corpus graph%s", f" for domain={domain}" if domain else "")
    builder = KGBuilder(model_name=settings.kg_embedding_model_name)
    result = builder.rebuild(db=db, domain=domain, cfg=KGConfig())
    result["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    logger.info(
        "Corpus graph refresh finished%s in %.3fs",
        f" for domain={domain}" if domain else "",
        result["elapsed_seconds"],
    )
    return result


def split_selective_states(predictions: list[dict]) -> dict:
    def stable_key(payload: dict) -> str:
        return json.dumps(payload, sort_keys=True, default=str)

    accepted: list[dict] = []
    abstained: list[dict] = []
    novel: list[dict] = []
    seen_abstained: set[str] = set()
    seen_novel: set[str] = set()
    for row in predictions or []:
        routing = str(row.get("routing") or "known").lower()
        for abstained_row in row.get("abstained_predictions") or []:
            if not isinstance(abstained_row, dict):
                continue
            abstained_key = stable_key(abstained_row)
            if abstained_key in seen_abstained:
                continue
            seen_abstained.add(abstained_key)
            abstained.append(abstained_row)
        if bool(row.get("abstain")) or str(row.get("decision") or "").lower() == "abstain" or routing == "boundary":
            abstained_key = stable_key(row)
            if abstained_key not in seen_abstained:
                seen_abstained.add(abstained_key)
                abstained.append(row)
            continue
        if routing == "known":
            accepted.append(row)
        elif routing == "novel" and not row.get("novel_candidates"):
            novel_key = stable_key(row)
            if novel_key not in seen_novel:
                seen_novel.add(novel_key)
                novel.append(row)
        for candidate in row.get("novel_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            novel_key = stable_key(candidate)
            if novel_key in seen_novel:
                continue
            seen_novel.add(novel_key)
            novel.append(candidate)
    return {
        "accepted_predictions": accepted,
        "abstained_predictions": abstained,
        "novel_candidates": novel,
    }


def _refresh_corpus_graph_task(domain: str | None) -> None:
    schedule_corpus_graph_refresh(domain)


def schedule_corpus_graph_refresh(domain: str | None) -> threading.Thread:
    logger = logging.getLogger(__name__)
    scope = domain or "all"
    with _GRAPH_REFRESH_LOCK:
        existing = _GRAPH_REFRESH_THREADS.get(scope)
        if existing is not None:
            logger.info("Reusing in-flight corpus graph refresh thread %s", existing.name)
            return existing
    thread_name = f"corpus-graph-refresh-{domain or 'all'}-{int(time.time() * 1000)}"

    def _worker() -> None:
        started_at = time.perf_counter()
        db = None
        try:
            db = SessionLocal()
            refresh_corpus_graph(db, domain=domain)
        except Exception:
            logger.exception("Failed refreshing corpus graph in background task")
        finally:
            if db is not None:
                db.close()
            logger.info(
                "Background corpus graph task closed%s after %.3fs",
                f" for domain={domain}" if domain else "",
                time.perf_counter() - started_at,
            )
            with _GRAPH_REFRESH_LOCK:
                if _GRAPH_REFRESH_THREADS.get(scope) is threading.current_thread():
                    _GRAPH_REFRESH_THREADS.pop(scope, None)

    thread = threading.Thread(target=_worker, name=thread_name, daemon=True)
    with _GRAPH_REFRESH_LOCK:
        _GRAPH_REFRESH_THREADS[scope] = thread
    thread.start()
    logger.info("Started background corpus graph refresh thread %s", thread.name)
    return thread
