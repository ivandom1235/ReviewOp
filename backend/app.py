# proto/backend/app.py
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from core.bootstrap import bootstrap_database, load_application_services
from core.db import get_db
from core.config import settings
from core.errors import AppError, DatabaseFailure
from models.schemas import (
    InferReviewIn,
    InferReviewOut,
)
from services.review_use_case import infer_review as run_review_inference
from services.open_aspect import open_aspect_model_status

from routes.analytics import router as analytics_router
from routes.graph import router as graph_router
from routes.jobs import router as jobs_router
from routes.infer import router as infer_router
from routes.user_portal import router as user_portal_router, seed_default_accounts, require_admin

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_database()

    from core.db import SessionLocal

    try:
        db = SessionLocal()
        try:
            seed_default_accounts(
                db,
                app_env=settings.app_env,
                seed_demo_users=str(os.getenv("SEED_DEMO_USERS") or os.getenv("REVIEWOP_SEED_DEMO_USERS") or "").lower()
                in {"1", "true", "yes", "on"},
            )
        finally:
            db.close()
    except OperationalError as exc:
        logger.warning("Default account seeding check skipped because the database is unavailable: %s", exc)

    services = load_application_services()
    app.state.services = services
    app.state.seq2seq_engine = services.seq2seq_engine
    app.state.implicit_client = services.implicit_client

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

@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": exc.to_payload()},
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(_: Request, exc: SQLAlchemyError):
    logger.warning("Database failure: %s", exc)
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
        logger.warning("Health check failed due to database connectivity: %s", ex)
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
def infer_review(
    payload: InferReviewIn,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    text_in = (payload.text or "").strip()
    if not text_in:
        raise HTTPException(status_code=400, detail="text is required")

    services = app.state.services
    return run_review_inference(
        db,
        text=text_in,
        domain=payload.domain,
        product_id=payload.product_id,
        persist=payload.persist,
        explicit_engine=services.seq2seq_engine,
        implicit_client=services.implicit_client,
        background_tasks=background_tasks,
    )


app.include_router(infer_router)
app.include_router(jobs_router)
app.include_router(analytics_router)
app.include_router(graph_router)
app.include_router(user_portal_router)
