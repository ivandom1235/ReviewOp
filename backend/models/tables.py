# proto/backend/models/tables.py
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Float,
    Enum,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db import Base


SentimentEnum = Enum("positive", "neutral", "negative", name="sentiment_enum")
JobStatusEnum = Enum("queued", "running", "done", "failed", name="job_status_enum")
JobItemStatusEnum = Enum("queued", "done", "failed", name="job_item_status_enum")


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str | None] = mapped_column(String(64), nullable=True)
    product_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # NEW: overall sentiment fields
    overall_sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="review",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"), nullable=False)

    aspect_raw: Mapped[str] = mapped_column(String(255), nullable=False)
    aspect_cluster: Mapped[str] = mapped_column(String(255), nullable=False)

    sentiment: Mapped[str] = mapped_column(SentimentEnum, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    # NEW: per-aspect weight + numeric score
    aspect_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    aspect_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    review: Mapped["Review"] = relationship(back_populates="predictions")
    evidence_spans: Mapped[list["EvidenceSpan"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class EvidenceSpan(Base):
    __tablename__ = "evidence_spans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False)

    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)

    prediction: Mapped["Prediction"] = relationship(back_populates="evidence_spans")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(JobStatusEnum, nullable=False, default="queued")
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    items: Mapped[list["JobItem"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class JobItem(Base):
    __tablename__ = "job_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)

    review_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(JobItemStatusEnum, nullable=False, default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="items")


# =========================
# NEW: Knowledge Graph Tables
# =========================

class AspectNode(Base):
    __tablename__ = "aspect_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    aspect_cluster: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str | None] = mapped_column(String(64), nullable=True)

    df: Mapped[int | None] = mapped_column(Integer, nullable=True)
    idf: Mapped[float | None] = mapped_column(Float, nullable=True)
    centrality: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("aspect_cluster", "domain", name="uq_aspect_nodes_cluster_domain"),
        Index("ix_aspect_nodes_aspect_cluster", "aspect_cluster"),
        Index("ix_aspect_nodes_domain", "domain"),
    )


class AspectEdge(Base):
    __tablename__ = "aspect_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    src_aspect: Mapped[str] = mapped_column(String(255), nullable=False)
    dst_aspect: Mapped[str] = mapped_column(String(255), nullable=False)

    edge_type: Mapped[str] = mapped_column(String(32), nullable=False)  # similarity|cooccurrence
    weight: Mapped[float] = mapped_column(Float, nullable=False)

    domain: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_aspect_edges_src", "src_aspect"),
        Index("ix_aspect_edges_dst", "dst_aspect"),
        Index("ix_aspect_edges_type", "edge_type"),
        Index("ix_aspect_edges_domain", "domain"),
        Index("ix_aspect_edges_src_dst_type", "src_aspect", "dst_aspect", "edge_type"),
    )


UserRoleEnum = Enum("admin", "user", name="user_role_enum")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    password_salt: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(UserRoleEnum, nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    token: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(lazy="joined")


class ProductCatalog(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    category: Mapped[str | None] = mapped_column(String(120), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # NEW: Caching for performance
    cached_average_rating: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cached_review_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_latest_review_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cached_helpful_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)



class UserProductReview(Base):
    __tablename__ = "user_product_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    review_text: Mapped[str] = mapped_column(Text, nullable=False)
    pros: Mapped[str | None] = mapped_column(Text, nullable=True)
    cons: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommendation: Mapped[bool | None] = mapped_column(nullable=True)
    helpful_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    linked_review_id: Mapped[int | None] = mapped_column(ForeignKey("reviews.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(lazy="joined")
    linked_review: Mapped["Review"] = relationship(lazy="joined")
