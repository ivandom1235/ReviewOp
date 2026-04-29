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
from models.tables import AbstainedPrediction, Prediction, RejectedAspectCandidate
from services.hybrid_pipeline import run_single_review_hybrid_pipeline
from services.responses import ContractMapper
from services.review_pipeline import run_single_review_pipeline


class _FakeEngine:
    def classify_sentiment_with_confidence(self, snippet: str, aspect: str):
        return ("neutral", 0.8)


class _FakeImplicitClient:
    def __init__(self, rows):
        self.rows = rows

    def predict(self, review_text: str, domain: str | None = None):
        return list(self.rows)


class ExplicitRuntimePersistenceTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_runtime_persists_prediction_quality_and_rejections(self) -> None:
        db = self.make_db()
        with patch("services.review_pipeline.extract_open_aspects", return_value=["my cam", "something", "battery life"]):
            review = run_single_review_pipeline(
                db,
                engine=_FakeEngine(),
                text="My cam is okay. Battery life is great. Something else happened.",
                domain="electronics",
                product_id="p1",
            )

        preds = db.query(Prediction).filter(Prediction.review_id == review.id).all()
        aspects = {p.aspect_raw for p in preds}
        self.assertIn("camera", aspects)
        self.assertIn("battery life", aspects)
        self.assertNotIn("something", aspects)
        for p in preds:
            self.assertIsNotNone(p.aspect_normalized)
            self.assertIsNotNone(p.aspect_canonical)
            self.assertIsNotNone(p.extraction_rule)
            self.assertIsNotNone(p.quality_score)
            self.assertIsNotNone(p.evidence_quality)
            self.assertIsNotNone(p.mapping_scope)

        rejected = db.query(RejectedAspectCandidate).filter(RejectedAspectCandidate.review_id == review.id).all()
        self.assertTrue(any(r.raw_text == "something" and r.reason for r in rejected))

    def test_runtime_rejects_noisy_explicit_candidates_without_dropping_valid_laptop_aspects(self) -> None:
        db = self.make_db()
        with patch(
            "services.review_pipeline.extract_open_aspects",
            return_value=["screen", "keyboard", "friend", "desk", "clock"],
        ):
            review = run_single_review_pipeline(
                db,
                engine=_FakeEngine(),
                text="The screen looks crisp and the keyboard feels steady. My friend placed it beside the desk clock.",
                domain="laptop",
                product_id="l1",
            )

        accepted = {
            row.aspect_raw
            for row in db.query(Prediction).filter(Prediction.review_id == review.id).all()
        }
        self.assertEqual({"screen", "keyboard"}, accepted)

        rejected = {
            row.normalized_text: row.reason
            for row in db.query(RejectedAspectCandidate).filter(RejectedAspectCandidate.review_id == review.id).all()
        }
        self.assertEqual("low_quality_aspect", rejected.get("friend"))
        self.assertEqual("low_quality_aspect", rejected.get("desk"))
        self.assertEqual("low_quality_aspect", rejected.get("clock"))

    def test_hybrid_pipeline_abstains_wrong_domain_implicit_predictions(self) -> None:
        db = self.make_db()
        implicit_rows = [
            {
                "aspect_raw": "battery_life",
                "aspect_cluster": "battery_life",
                "sentiment": "positive",
                "confidence": 0.72,
                "routing": "known",
                "source": "implicit",
                "evidence_spans": [{"start_char": 0, "end_char": 27, "snippet": "It lasted through the flight"}],
            },
            {
                "aspect_raw": "food_quality",
                "aspect_cluster": "food_quality",
                "sentiment": "positive",
                "confidence": 0.68,
                "routing": "known",
                "source": "implicit",
                "evidence_spans": [{"start_char": 29, "end_char": 62, "snippet": "the colors looked delicious"}],
            },
            {
                "aspect_raw": "ambience",
                "aspect_cluster": "ambience",
                "sentiment": "positive",
                "confidence": 0.66,
                "routing": "known",
                "source": "implicit",
                "evidence_spans": [{"start_char": 64, "end_char": 87, "snippet": "the room felt quiet"}],
            },
        ]
        with patch("services.review_pipeline.extract_open_aspects", return_value=[]):
            review, _, implicit_predictions, final_predictions = run_single_review_hybrid_pipeline(
                db,
                explicit_engine=_FakeEngine(),
                implicit_client=_FakeImplicitClient(implicit_rows),
                text="It lasted through the flight and the colors looked delicious while the room felt quiet.",
                domain="laptop",
                product_id="l2",
            )

        final_aspects = {row["aspect_cluster"] for row in final_predictions}
        self.assertEqual({"battery_life"}, final_aspects)

        response = ContractMapper().to_infer_review_out(review, final_predictions, implicit_predictions)
        accepted_aspects = {row.aspect for row in response.accepted_predictions}
        self.assertEqual({"battery_life"}, accepted_aspects)

        abstained = db.query(AbstainedPrediction).filter(AbstainedPrediction.review_id == review.id).all()
        self.assertTrue(any(row.reason == "domain_mismatch" for row in abstained))


if __name__ == "__main__":
    unittest.main()
