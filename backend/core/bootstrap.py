from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationServices:
    seq2seq_engine: Any
    implicit_client: Any


class DisabledImplicitClient:
    def predict(self, review_text: str, domain: str | None = None, top_k: int | None = None):  # noqa: ARG002
        return []


def apply_schema_patches(engine) -> None:
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
        add_column_if_missing("user_product_reviews", "deleted_at", "ALTER TABLE user_product_reviews ADD COLUMN deleted_at DATETIME NULL")
        add_column_if_missing("predictions", "source", "ALTER TABLE predictions ADD COLUMN source VARCHAR(16) NULL")
        add_column_if_missing("predictions", "aspect_weight", "ALTER TABLE predictions ADD COLUMN aspect_weight FLOAT NULL")
        add_column_if_missing("predictions", "aspect_score", "ALTER TABLE predictions ADD COLUMN aspect_score FLOAT NULL")
        add_column_if_missing("predictions", "aspect_normalized", "ALTER TABLE predictions ADD COLUMN aspect_normalized VARCHAR(255) NULL")
        add_column_if_missing("predictions", "aspect_canonical", "ALTER TABLE predictions ADD COLUMN aspect_canonical VARCHAR(255) NULL")
        add_column_if_missing("predictions", "extraction_rule", "ALTER TABLE predictions ADD COLUMN extraction_rule VARCHAR(64) NULL")
        add_column_if_missing("predictions", "quality_score", "ALTER TABLE predictions ADD COLUMN quality_score FLOAT NULL")
        add_column_if_missing("predictions", "evidence_quality", "ALTER TABLE predictions ADD COLUMN evidence_quality FLOAT NULL")
        add_column_if_missing("predictions", "mapping_scope", "ALTER TABLE predictions ADD COLUMN mapping_scope VARCHAR(32) NULL")

        add_column_if_missing("reviews", "overall_sentiment", "ALTER TABLE reviews ADD COLUMN overall_sentiment VARCHAR(16) NULL")
        add_column_if_missing("reviews", "overall_score", "ALTER TABLE reviews ADD COLUMN overall_score FLOAT NULL")
        add_column_if_missing("reviews", "overall_confidence", "ALTER TABLE reviews ADD COLUMN overall_confidence FLOAT NULL")

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
        create_table_if_missing(
            "rejected_aspect_candidates",
            """
            CREATE TABLE rejected_aspect_candidates (
                id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
                review_id INTEGER NOT NULL,
                raw_text VARCHAR(255) NOT NULL,
                normalized_text VARCHAR(255) NOT NULL,
                reason VARCHAR(128) NOT NULL,
                quality_score FLOAT NOT NULL DEFAULT 0.0,
                evidence_text TEXT NULL,
                source_rule VARCHAR(64) NULL,
                created_at DATETIME NOT NULL,
                INDEX ix_rejected_aspect_candidates_review_id (review_id),
                CONSTRAINT fk_rejected_aspect_candidates_review_id
                  FOREIGN KEY (review_id) REFERENCES reviews(id)
                  ON DELETE CASCADE
            )
            """,
        )


def bootstrap_database() -> None:
    from core.db import engine, init_db

    try:
        init_db()
        apply_schema_patches(engine)
    except OperationalError as exc:
        logger.warning("Database bootstrap skipped because the database is unavailable: %s", exc)


def load_application_services() -> ApplicationServices:
    from services.implicit_client import ImplicitClient
    from services.seq2seq_infer import Seq2SeqEngine

    seq2seq_engine = Seq2SeqEngine.load()
    if not settings.enable_implicit:
        implicit_client = DisabledImplicitClient()
    else:
        try:
            implicit_client = ImplicitClient()
        except Exception as exc:
            logger.warning("Implicit client unavailable at startup: %s", exc)
            implicit_client = DisabledImplicitClient()
    return ApplicationServices(seq2seq_engine=seq2seq_engine, implicit_client=implicit_client)
