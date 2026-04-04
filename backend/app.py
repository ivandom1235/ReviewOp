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
from models.schemas import InferReviewIn, InferReviewOut, PredictionOut, EvidenceSpanOut
from services.seq2seq_infer import Seq2SeqEngine
from services.review_pipeline import refresh_corpus_graph, _refresh_corpus_graph_task
from services.hybrid_pipeline import run_single_review_hybrid_pipeline
from services.implicit_client import ImplicitClient
from services.llm_verifier import LLMVerifier

from routes.analytics import router as analytics_router
from routes.graph import router as graph_router
from routes.jobs import router as jobs_router
from routes.infer import router as infer_router
from routes.user_portal import router as user_portal_router, seed_default_accounts


app = FastAPI(title="Proto ReviewOps MVP (Hybrid)", version="5.5.0")
logger = logging.getLogger(__name__)





@app.on_event("startup")
def on_startup():
    init_db()
    _apply_schema_patches()

    db = SessionLocal()
    try:
        seed_default_accounts(db)
    finally:
        db.close()

    app.state.seq2seq_engine = Seq2SeqEngine.load()
    app.state.implicit_client = ImplicitClient()
    app.state.llm_verifier = LLMVerifier()


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
            "llm_verifier_enabled": settings.enable_llm_verifier,
            "llm_model": settings.llm_model_name if settings.enable_llm_verifier else None,
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
    llm_verifier = app.state.llm_verifier

    review_obj, explicit_preds, implicit_preds, final_preds = run_single_review_hybrid_pipeline(
        db,
        explicit_engine=explicit_engine,
        implicit_client=implicit_client,
        llm_verifier=llm_verifier,
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
            )
        )

    return InferReviewOut(
        review_id=review_obj.id,
        domain=review_obj.domain,
        product_id=review_obj.product_id,
        predictions=preds_out,
        overall_sentiment=review_obj.overall_sentiment,
        overall_score=review_obj.overall_score,
        overall_confidence=review_obj.overall_confidence,
    )


app.include_router(infer_router)   # keep /infer/csv etc.
app.include_router(jobs_router)
app.include_router(analytics_router)
app.include_router(graph_router)
app.include_router(user_portal_router)
