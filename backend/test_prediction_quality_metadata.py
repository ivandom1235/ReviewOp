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
from models.tables import Prediction, RejectedAspectCandidate, Review


class PredictionQualityMetadataTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_prediction_quality_fields_persist(self) -> None:
        db = self.make_db()
        review = Review(text="Battery life is good", domain="electronics")
        db.add(review)
        db.flush()

        pred = Prediction(
            review_id=review.id,
            aspect_raw="battery life",
            aspect_cluster="battery life",
            sentiment="positive",
            confidence=0.91,
            source="explicit",
            aspect_normalized="battery life",
            aspect_canonical="battery life",
            extraction_rule="noun_chunk",
            quality_score=0.87,
            evidence_quality=1.0,
            mapping_scope="generic",
        )
        db.add(pred)
        db.commit()

        saved = db.query(Prediction).one()
        self.assertEqual(saved.aspect_normalized, "battery life")
        self.assertEqual(saved.aspect_canonical, "battery life")
        self.assertEqual(saved.extraction_rule, "noun_chunk")
        self.assertAlmostEqual(float(saved.quality_score or 0), 0.87, places=6)
        self.assertAlmostEqual(float(saved.evidence_quality or 0), 1.0, places=6)
        self.assertEqual(saved.mapping_scope, "generic")

    def test_rejected_aspect_candidate_persists(self) -> None:
        db = self.make_db()
        review = Review(text="Something is weird", domain="electronics")
        db.add(review)
        db.flush()

        rej = RejectedAspectCandidate(
            review_id=review.id,
            raw_text="something",
            normalized_text="something",
            reason="vague_head",
            quality_score=0.12,
            evidence_text="something is weird",
            source_rule="noun_chunk",
        )
        db.add(rej)
        db.commit()

        saved = db.query(RejectedAspectCandidate).one()
        self.assertEqual(saved.reason, "vague_head")
        self.assertEqual(saved.raw_text, "something")


if __name__ == "__main__":
    unittest.main()
