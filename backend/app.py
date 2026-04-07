# proto/backend/app.py
from __future__ import annotations

import logging

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from core.db import engine, get_db, SessionLocal, init_db
from core.config import settings
from core.errors import AppError, DatabaseFailure
from models.schemas import (
    AbstainedPredictionOut,
    EvidenceSpanOut,
    InferReviewIn,
    InferReviewOut,
    NovelCandidateOut,
    PredictionOut,
    SelectivePredictionOut,
)
from services.seq2seq_infer import Seq2SeqEngine
from services.review_pipeline import refresh_corpus_graph, _refresh_corpus_graph_task, split_selective_states
from services.hybrid_pipeline import run_single_review_hybrid_pipeline
from services.implicit_client import ImplicitClient
from services.flash_client import FlashInferenceClient

from routes.analytics import router as analytics_router
from routes.graph import router as graph_router
from routes.jobs import router as jobs_router
from routes.infer import router as infer_router
from routes.user_portal import router as user_portal_router, seed_default_accounts


app = FastAPI(title="ReviewOps V6", version="6.0.0")
logger = logging.getLogger(__name__)





@app.on_event("startup")
async def on_startup():
    init_db()
    _apply_schema_patches()

    db = SessionLocal()
    try:
        seed_default_accounts(db)
    finally:
        db.close()

    app.state.seq2seq_engine = Seq2SeqEngine.load()
    app.state.implicit_client = ImplicitClient()

    # RunPod Flash: initialize and warmup in background
    flash_client = FlashInferenceClient()
    app.state.flash_client = flash_client
    if settings.flash_enabled:
        logger.info("Initiating RunPod Flash warmup (background)...")
        import asyncio
        asyncio.create_task(_flash_warmup_task(flash_client))
    else:
        logger.info("RunPod Flash disabled via config")


async def _flash_warmup_task(client: FlashInferenceClient) -> None:
    """Background task to warm up Flash endpoint without blocking startup."""
    try:
        await client.warmup()
    except Exception as exc:
        logger.warning("Flash warmup background task failed: %s", exc)


@app.get("/flash/status")
async def flash_status():
    """Diagnostic endpoint for RunPod Flash health."""
    client: FlashInferenceClient = getattr(app.state, "flash_client", None)
    if client is None:
        return {"ok": False, "error": "Flash client not initialized"}
    return {"ok": True, **client.status()}


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": exc.to_payload()},
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(_: Request, exc: SQLAlchemyError):
    logger.exception("Database failure", exc_info=exc)
    failure = DatabaseFailure()
    return JSONResponse(
        status_code=failure.status_code,
        content={"ok": False, "error": failure.to_payload()},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    logger.exception("Unhandled server error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred.",
            },
        },
    )


def _apply_schema_patches() -> None:
    inspector = inspect(engine)
    with engine.begin() as conn:
        if inspector.has_table("products"):
            product_cols = {c["name"] for c in inspector.get_columns("products")}
            if "cached_average_rating" not in product_cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN cached_average_rating FLOAT NOT NULL DEFAULT 0.0"))
            if "cached_review_count" not in product_cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN cached_review_count INTEGER NOT NULL DEFAULT 0"))
            if "cached_latest_review_at" not in product_cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN cached_latest_review_at DATETIME NULL"))
            if "cached_helpful_count" not in product_cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN cached_helpful_count INTEGER NOT NULL DEFAULT 0"))

        if not inspector.has_table("admin_dismissed_alerts"):
            conn.execute(
                text(
                    """
                    CREATE TABLE admin_dismissed_alerts (
                        id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        type VARCHAR(64) NOT NULL,
                        aspect VARCHAR(255) NOT NULL,
                        message VARCHAR(512) NOT NULL,
                        domain VARCHAR(64) NULL,
                        signature VARCHAR(64) NOT NULL UNIQUE,
                        dismissed_at DATETIME NOT NULL
                    )
                    """
                )
            )
        if not inspector.has_table("abstained_predictions"):
            conn.execute(
                text(
                    """
                    CREATE TABLE abstained_predictions (
                        id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        review_id INTEGER NOT NULL,
                        reason VARCHAR(128) NOT NULL,
                        confidence FLOAT NOT NULL DEFAULT 0.0,
                        ambiguity_score FLOAT NOT NULL DEFAULT 0.0,
                        created_at DATETIME NOT NULL,
                        INDEX ix_abstained_predictions_review_id (review_id),
                        CONSTRAINT fk_abstained_predictions_review_id
                          FOREIGN KEY (review_id) REFERENCES reviews(id)
                          ON DELETE CASCADE
                    )
                    """
                )
            )
        if not inspector.has_table("novel_candidates"):
            conn.execute(
                text(
                    """
                    CREATE TABLE novel_candidates (
                        id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        review_id INTEGER NOT NULL,
                        aspect VARCHAR(255) NOT NULL,
                        novelty_score FLOAT NOT NULL DEFAULT 0.0,
                        confidence FLOAT NULL,
                        evidence TEXT NULL,
                        evidence_start INTEGER NULL,
                        evidence_end INTEGER NULL,
                        created_at DATETIME NOT NULL,
                        INDEX ix_novel_candidates_review_id (review_id),
                        INDEX ix_novel_candidates_aspect (aspect),
                        CONSTRAINT fk_novel_candidates_review_id
                          FOREIGN KEY (review_id) REFERENCES reviews(id)
                          ON DELETE CASCADE
                    )
                    """
                )
            )


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {
            "ok": True,
            "env": settings.app_env,
            "db": "connected",
            "explicit_model": settings.seq2seq_model_name,
            "implicit_enabled": settings.enable_implicit,
            "llm_verifier_enabled": False,
            "llm_model": None,
        }
    except SQLAlchemyError as ex:
        logger.warning("Health check failed due to database connectivity", exc_info=ex)
        return {
            "ok": False,
            "error": {
                "code": "database_unavailable",
                "message": "Database connectivity check failed.",
            },
        }
    except Exception as ex:
        logger.exception("Health check failed", exc_info=ex)
        return {
            "ok": False,
            "error": {
                "code": "health_check_failed",
                "message": "Health check failed.",
            },
        }


@app.post("/infer/review", response_model=InferReviewOut, tags=["infer"])
def infer_review(payload: InferReviewIn, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    text_in = (payload.text or "").strip()
    if not text_in:
        raise HTTPException(status_code=400, detail="text is required")

    explicit_engine = app.state.seq2seq_engine
    implicit_client = app.state.implicit_client

    review_obj, explicit_preds, implicit_preds, final_preds = run_single_review_hybrid_pipeline(
        db,
        explicit_engine=explicit_engine,
        implicit_client=implicit_client,
        text=text_in,
        domain=payload.domain,
        product_id=payload.product_id,
    )
    db.commit()

    background_tasks.add_task(_refresh_corpus_graph_task, review_obj.domain)

    db.refresh(review_obj)

    preds_out: list[PredictionOut] = []
    for pred in final_preds:
        spans = [
            EvidenceSpanOut(
                start_char=int(ev.get("start_char", 0)),
                end_char=int(ev.get("end_char", 0)),
                snippet=str(ev.get("snippet", "")),
            )
            for ev in pred.get("evidence_spans", []) or []
        ]
        source = pred.get("source") or "verified"
        preds_out.append(
            PredictionOut(
                aspect_raw=pred["aspect_raw"],
                aspect_cluster=pred.get("aspect_cluster") or pred["aspect_raw"],
                sentiment=pred.get("sentiment") or "neutral",
                confidence=float(pred.get("confidence", 0.0)),
                evidence_spans=spans,
                rationale=pred.get("rationale") or "",
                source=source,
                is_implicit=(source == "implicit"),
                verification_status="kept" if source in {"explicit", "implicit", "verified"} else None,
                decision=pred.get("decision"),
                routing=pred.get("routing"),
                ambiguity_score=pred.get("ambiguity_score"),
                novelty_score=pred.get("novelty_score"),
            )
        )

    selective_states = split_selective_states(implicit_preds)
    accepted_out = [
        SelectivePredictionOut(
            aspect=str(row.get("aspect_cluster") or row.get("aspect_raw") or row.get("aspect") or ""),
            sentiment=str(row.get("sentiment") or "neutral"),
            confidence=float(row.get("confidence", 0.0)),
            routing=str(row.get("routing") or "known"),
            evidence=str((row.get("evidence_spans") or [{}])[0].get("snippet") or "") if row.get("evidence_spans") else None,
            evidence_start=(
                int((row.get("evidence_spans") or [{}])[0].get("start_char"))
                if row.get("evidence_spans") and (row.get("evidence_spans") or [{}])[0].get("start_char") is not None
                else None
            ),
            evidence_end=(
                int((row.get("evidence_spans") or [{}])[0].get("end_char"))
                if row.get("evidence_spans") and (row.get("evidence_spans") or [{}])[0].get("end_char") is not None
                else None
            ),
        )
        for row in selective_states.get("accepted_predictions", [])
        if str(row.get("aspect_cluster") or row.get("aspect_raw") or row.get("aspect") or "").strip()
    ]
    abstained_out = [
        AbstainedPredictionOut(
            reason=str(row.get("reason") or "low_selective_confidence"),
            confidence=float(row.get("confidence", 0.0)),
            ambiguity_score=float(row.get("ambiguity_score", 0.0)),
        )
        for row in selective_states.get("abstained_predictions", [])
    ]
    novel_out = [
        NovelCandidateOut(
            aspect=str(row.get("aspect") or row.get("aspect_raw") or ""),
            novelty_score=float(row.get("novelty_score", 0.0)),
            confidence=float(row.get("confidence", 0.0)) if row.get("confidence") is not None else None,
        )
        for row in selective_states.get("novel_candidates", [])
        if str(row.get("aspect") or row.get("aspect_raw") or "").strip()
    ]

    return InferReviewOut(
        review_id=review_obj.id,
        domain=review_obj.domain,
        product_id=review_obj.product_id,
        predictions=preds_out,
        overall_sentiment=review_obj.overall_sentiment,
        overall_score=review_obj.overall_score,
        overall_confidence=review_obj.overall_confidence,
        accepted_predictions=accepted_out,
        abstained_predictions=abstained_out,
        novel_candidates=novel_out,
    )


app.include_router(infer_router)   # keep /infer/csv etc.
app.include_router(jobs_router)
app.include_router(analytics_router)
app.include_router(graph_router)
app.include_router(user_portal_router)
