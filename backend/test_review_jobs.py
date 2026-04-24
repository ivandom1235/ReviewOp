from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

warnings.simplefilter("ignore", DeprecationWarning)

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.db import Base
from fastapi import BackgroundTasks

from models.schemas import SubmitReviewIn
from models.tables import EvidenceSpan, Job, JobItem, Prediction, ProductCatalog, Review, User, UserProductReview
from routes.user_portal import get_my_review_job, submit_review
from services.review_jobs import create_review_analysis_job, process_review_analysis_job


class ReviewAnalysisJobTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_create_review_analysis_job_queues_without_predictions(self) -> None:
        db = self.make_db()
        review = Review(text="Battery dies before lunch.", domain="electronics", product_id="p1")
        db.add(review)
        db.flush()

        job = create_review_analysis_job(db, review)
        db.commit()

        self.assertEqual(job.status, "queued")
        self.assertEqual(job.total, 1)
        self.assertEqual(job.processed, 0)
        self.assertEqual(db.query(Prediction).filter(Prediction.review_id == review.id).count(), 0)

    def test_process_review_analysis_job_updates_status_and_predictions(self) -> None:
        db = self.make_db()
        review = Review(text="Battery dies before lunch.", domain="electronics", product_id="p1")
        db.add(review)
        db.flush()
        job = create_review_analysis_job(db, review)
        db.commit()

        def fake_pipeline(db, review, **_kwargs):
            prediction = Prediction(
                review_id=review.id,
                aspect_raw="battery",
                aspect_cluster="battery",
                sentiment="negative",
                confidence=0.81,
            )
            prediction.evidence_spans.append(EvidenceSpan(start_char=0, end_char=7, snippet="Battery"))
            db.add(prediction)
            db.flush()
            return review, [], [], [{"aspect_raw": "battery"}]

        with patch("services.review_jobs.run_single_review_hybrid_pipeline", side_effect=fake_pipeline):
            process_review_analysis_job(
                db,
                job_id=job.id,
                explicit_engine=object(),
                implicit_client=object(),
            )

        refreshed = db.query(Job).filter(Job.id == job.id).one()
        self.assertEqual(refreshed.status, "done")
        self.assertEqual(refreshed.processed, 1)
        self.assertEqual(refreshed.failed, 0)
        self.assertGreater(db.query(Prediction).filter(Prediction.review_id == review.id).count(), 0)

    def test_submit_review_enqueues_analysis_without_inline_inference(self) -> None:
        db = self.make_db()
        user = User(username="alice", password_hash="x", password_salt="s", role="user")
        product = ProductCatalog(product_id="p1", name="Phone", category="electronics")
        db.add_all([user, product])
        db.commit()
        request = type(
            "Request",
            (),
            {"app": type("App", (), {"state": type("State", (), {"seq2seq_engine": object(), "implicit_client": object()})()})()},
        )()

        with patch("routes.user_portal.run_single_review_hybrid_pipeline", side_effect=AssertionError("inline inference called")):
            out = submit_review(
                SubmitReviewIn(product_id="p1", product_name="Phone", rating=2, review_text="Battery dies before lunch."),
                BackgroundTasks(),
                request,
                user,
                db,
            )

        self.assertIsNotNone(out.linked_review_id)
        self.assertIsNotNone(out.job_id)
        self.assertEqual(out.analysis_status, "queued")
        self.assertEqual(db.query(Job).filter(Job.id == out.job_id).one().status, "queued")
        self.assertEqual(db.query(Prediction).filter(Prediction.review_id == out.linked_review_id).count(), 0)

    def test_user_can_poll_owned_review_job(self) -> None:
        db = self.make_db()
        user = User(username="alice", password_hash="x", password_salt="s", role="user")
        other = User(username="bob", password_hash="x", password_salt="s", role="user")
        review = Review(text="Battery dies.", domain="electronics", product_id="p1")
        db.add_all([user, other, review])
        db.flush()
        user_review = UserProductReview(user_id=user.id, product_id="p1", rating=2, review_text="Battery dies.", linked_review_id=review.id)
        db.add(user_review)
        job = Job(status="queued", total=1, processed=0, failed=0)
        db.add(job)
        db.flush()
        db.add(JobItem(job_id=job.id, row_index=0, review_id=review.id, status="queued"))
        db.commit()

        status = get_my_review_job(job.id, user, db)

        self.assertEqual(status.job_id, job.id)
        self.assertEqual(status.status, "queued")
        with self.assertRaises(Exception):
            get_my_review_job(job.id, other, db)


if __name__ == "__main__":
    unittest.main()
