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
from models.tables import AspectNode, EvidenceSpan, Prediction, Review
from services.kg_build import KGBuilder


class KGQualityGateTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_rebuild_excludes_blocklisted_and_low_quality_aspects(self) -> None:
        db = self.make_db()
        review = Review(text="battery is bad and something is odd", domain="electronics")
        db.add(review)
        db.flush()

        keep = Prediction(
            review_id=review.id,
            aspect_raw="battery life",
            aspect_cluster="battery life",
            sentiment="negative",
            confidence=0.9,
            source="explicit",
            aspect_weight=0.8,
        )
        keep.evidence_spans.append(EvidenceSpan(start_char=0, end_char=12, snippet="battery"))

        blocked = Prediction(
            review_id=review.id,
            aspect_raw="something",
            aspect_cluster="something",
            sentiment="neutral",
            confidence=0.7,
            source="explicit",
            aspect_weight=0.2,
        )
        blocked.evidence_spans.append(EvidenceSpan(start_char=20, end_char=29, snippet="something"))
        db.add_all([keep, blocked])
        db.commit()

        out = KGBuilder().rebuild(db, domain="electronics")
        self.assertTrue(out.get("ok"))

        node_aspects = {n.aspect_cluster for n in db.query(AspectNode).all()}
        self.assertIn("battery life", node_aspects)
        self.assertNotIn("something", node_aspects)


if __name__ == "__main__":
    unittest.main()
