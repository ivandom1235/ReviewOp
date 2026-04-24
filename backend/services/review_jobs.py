from __future__ import annotations

from datetime import datetime
import logging
import threading

from sqlalchemy.orm import Session

from core.db import SessionLocal
from models.tables import Job, JobItem, Review
from services.hybrid_pipeline import run_single_review_hybrid_pipeline

logger = logging.getLogger(__name__)


def _error_code(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return "invalid_input"
    return type(exc).__name__


def create_review_analysis_job(db: Session, review: Review) -> Job:
    job = Job(status="queued", total=1, processed=0, failed=0, updated_at=datetime.utcnow())
    db.add(job)
    db.flush()
    db.add(JobItem(job_id=job.id, row_index=0, review_id=review.id, status="queued"))
    db.flush()
    return job


def process_review_analysis_job(
    db: Session,
    *,
    job_id: str,
    explicit_engine,
    implicit_client,
) -> None:
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise ValueError(f"job not found: {job_id}")
    item = db.query(JobItem).filter(JobItem.job_id == job.id, JobItem.row_index == 0).first()
    if not item or item.review_id is None:
        job.status = "failed"
        job.failed = 1
        job.error = "review_missing"
        job.updated_at = datetime.utcnow()
        db.add(job)
        db.commit()
        return
    review = db.query(Review).filter(Review.id == item.review_id).first()
    if not review:
        item.status = "failed"
        item.error = "review_missing"
        job.status = "failed"
        job.failed = 1
        job.error = "review_missing"
        job.updated_at = datetime.utcnow()
        db.add(item)
        db.add(job)
        db.commit()
        return

    job.status = "running"
    job.updated_at = datetime.utcnow()
    item.status = "queued"
    db.add(job)
    db.add(item)
    db.commit()

    try:
        run_single_review_hybrid_pipeline(
            db,
            explicit_engine=explicit_engine,
            implicit_client=implicit_client,
            text=review.text,
            domain=review.domain,
            product_id=review.product_id,
            review=review,
            replace_existing=True,
        )
        item.status = "done"
        item.error = None
        job.status = "done"
        job.processed = 1
        job.failed = 0
        job.error = None
        job.updated_at = datetime.utcnow()
        db.add(item)
        db.add(job)
        db.commit()
    except Exception as exc:
        db.rollback()
        item = db.query(JobItem).filter(JobItem.job_id == job_id, JobItem.row_index == 0).first()
        job = db.query(Job).filter(Job.id == job_id).first()
        if item:
            item.status = "failed"
            item.error = _error_code(exc)
            db.add(item)
        if job:
            job.status = "failed"
            job.failed = 1
            job.error = _error_code(exc)
            job.updated_at = datetime.utcnow()
            db.add(job)
        db.commit()


def schedule_review_analysis_job(
    *,
    job_id: str,
    explicit_engine,
    implicit_client,
) -> threading.Thread:
    def _worker() -> None:
        db = SessionLocal()
        try:
            process_review_analysis_job(
                db,
                job_id=job_id,
                explicit_engine=explicit_engine,
                implicit_client=implicit_client,
            )
        except Exception:
            logger.exception("Review analysis job worker crashed (job_id=%s)", job_id)
        finally:
            db.close()

    thread = threading.Thread(target=_worker, name=f"review-analysis-job-{job_id}", daemon=True)
    thread.start()
    return thread
