from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.db import Base
from models.tables import NovelCandidate, Prediction, Review
from services.hybrid_pipeline import run_single_review_hybrid_pipeline


class FakeImplicitClient:
    def predict(self, **_kwargs):
        return [
            {"aspect": "battery_life", "sentiment": "negative", "confidence": 0.88, "routing": "known"},
            {
                "aspect": "hinge_sparks",
                "sentiment": "negative",
                "confidence": 0.72,
                "routing": "novel",
                "novelty_score": 0.93,
                "evidence_spans": [{"start_char": 12, "end_char": 24, "snippet": "hinge sparks"}],
            },
        ]


class HybridPersistenceTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_only_known_implicit_predictions_merge_into_final_predictions(self) -> None:
        db = self.make_db()

        def fake_explicit_pipeline(db, **kwargs):
            review = Review(text=kwargs["text"], domain=kwargs.get("domain"), product_id=kwargs.get("product_id"))
            db.add(review)
            db.flush()
            db.add(
                Prediction(
                    review_id=review.id,
                    aspect_raw="screen",
                    aspect_cluster="screen",
                    sentiment="positive",
                    confidence=0.91,
                )
            )
            db.flush()
            return review

        with patch("services.hybrid_pipeline.run_single_review_pipeline", side_effect=fake_explicit_pipeline):
            review, _, _, merged = run_single_review_hybrid_pipeline(
                db,
                explicit_engine=object(),
                implicit_client=FakeImplicitClient(),
                text="The screen is crisp but the hinge sparks and battery fades.",
                domain="electronics",
                product_id="p1",
            )

        prediction_aspects = {
            row.aspect_cluster for row in db.query(Prediction).filter(Prediction.review_id == review.id).all()
        }
        novel_aspects = {
            row.aspect for row in db.query(NovelCandidate).filter(NovelCandidate.review_id == review.id).all()
        }

        self.assertIn("screen", prediction_aspects)
        self.assertIn("battery_life", prediction_aspects)
        self.assertNotIn("hinge_sparks", prediction_aspects)
        self.assertEqual(novel_aspects, {"hinge_sparks"})
        self.assertTrue(any(row.get("aspect") == "battery_life" for row in merged))


if __name__ == "__main__":
    unittest.main()
