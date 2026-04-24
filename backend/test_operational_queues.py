from __future__ import annotations

import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.db import Base
from models.tables import AbstainedPrediction, NovelCandidate, Review
from services.analytics import needs_review_queue, novel_candidates_queue


class OperationalQueueTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_needs_review_queue_lists_abstained_predictions_with_review_text(self) -> None:
        db = self.make_db()
        review = Review(text="Screen hurts my eyes.", domain="electronics", product_id="p1")
        db.add(review)
        db.flush()
        db.add(AbstainedPrediction(review_id=review.id, reason="boundary", confidence=0.42, ambiguity_score=0.8))
        db.commit()

        rows = needs_review_queue(db)

        self.assertEqual(rows[0]["review_id"], review.id)
        self.assertEqual(rows[0]["reason"], "boundary")
        self.assertEqual(rows[0]["review_text"], "Screen hurts my eyes.")

    def test_novel_candidates_queue_lists_evidence_and_scores(self) -> None:
        db = self.make_db()
        review = Review(text="The hinge sparks.", domain="electronics", product_id="p1")
        db.add(review)
        db.flush()
        db.add(
            NovelCandidate(
                review_id=review.id,
                aspect="hinge_sparks",
                novelty_score=0.91,
                confidence=0.66,
                evidence="hinge sparks",
                evidence_start=4,
                evidence_end=16,
            )
        )
        db.commit()

        rows = novel_candidates_queue(db)

        self.assertEqual(rows[0]["aspect"], "hinge_sparks")
        self.assertEqual(rows[0]["evidence"], "hinge sparks")
        self.assertEqual(rows[0]["review_text"], "The hinge sparks.")


if __name__ == "__main__":
    unittest.main()
