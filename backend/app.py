# proto/backend/app.py
from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from core.db import engine, get_db, SessionLocal, init_db
from core.config import settings
from core.errors import AppError, DatabaseFailure
from models.schemas import (
    InferReviewIn,
    InferReviewOut,
)
from services.seq2seq_infer import Seq2SeqEngine
from services.review_pipeline import _refresh_corpus_graph_task
from services.hybrid_pipeline import run_single_review_hybrid_pipeline
from services.implicit_client import ImplicitClient
from services.open_aspect import open_aspect_model_status
from services.responses import ContractMapper

from routes.analytics import router as analytics_router
from routes.graph import router as graph_router
from routes.jobs import router as jobs_router
from routes.infer import router as infer_router
from routes.user_portal import router as user_portal_router, seed_default_accounts

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _apply_schema_patches()

    db = SessionLocal()
    try:
        seed_default_accounts(db)
    finally:
        db.close()

    app.state.seq2seq_engine = Seq2SeqEngine.load()
    try:
        app.state.implicit_client = ImplicitClient()
    except Exception as exc:
        logger.warning("Implicit client unavailable at startup: %s", exc)
        app.state.implicit_client = _DisabledImplicitClient()

    yield


app = FastAPI(
    title="ReviewOps V6 - Research Benchmark Backend",
    description="""
    Backend for the ReviewOp research benchmark system.
    Supports hybrid aspect-based sentiment analysis (Explicit + Implicit), 
    knowledge graph generation, and administrative analytics.
    """,
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "syntaxHighlight.theme": "obsidian",
        "docExpansion": "list",
        "filter": True,
        "tryItOutEnabled": True,
    },
    lifespan=lifespan,
)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class _DisabledImplicitClient:
    def predict(self, review_text: str, domain: str | None = None, top_k: int | None = None):  # noqa: ARG002
        return []



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
    with engine.begin() as conn:
        inspector = inspect(conn)

        def has_column(table_name: str, column_name: str) -> bool:
            return any(col["name"] == column_name for col in inspector.get_columns(table_name))

        def add_column_if_missing(table_name: str, column_name: str, ddl: str) -> None:
            if inspector.has_table(table_name) and not has_column(table_name, column_name):
                conn.execute(text(ddl))

        def create_table_if_missing(table_name: str, ddl: str) -> None:
            if not inspector.has_table(table_name):
                conn.execute(text(ddl))

        add_column_if_missing("products", "cached_average_rating", "ALTER TABLE products ADD COLUMN cached_average_rating FLOAT NOT NULL DEFAULT 0.0")
        add_column_if_missing("products", "cached_review_count", "ALTER TABLE products ADD COLUMN cached_review_count INTEGER NOT NULL DEFAULT 0")
        add_column_if_missing("products", "cached_latest_review_at", "ALTER TABLE products ADD COLUMN cached_latest_review_at DATETIME NULL")
        add_column_if_missing("products", "cached_helpful_count", "ALTER TABLE products ADD COLUMN cached_helpful_count INTEGER NOT NULL DEFAULT 0")
        add_column_if_missing("user_product_reviews", "reply_to_review_id", "ALTER TABLE user_product_reviews ADD COLUMN reply_to_review_id INTEGER NULL")

        create_table_if_missing(
            "admin_dismissed_alerts",
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
            """,
        )
        create_table_if_missing(
            "abstained_predictions",
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
            """,
        )
        create_table_if_missing(
            "novel_candidates",
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
            """,
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
            "open_aspect_model": open_aspect_model_status(),
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

    review_obj, _, implicit_preds, final_preds = run_single_review_hybrid_pipeline(
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

    mapper = ContractMapper()
    return mapper.to_infer_review_out(
        review_obj=review_obj,
        final_predictions=final_preds,
        implicit_predictions=implicit_preds
    )
app.include_router(infer_router)   # keep /infer/csv etc.
app.include_router(jobs_router)
app.include_router(analytics_router)
app.include_router(graph_router)
app.include_router(user_portal_router)
