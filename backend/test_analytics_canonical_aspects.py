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
from models.tables import EvidenceSpan, Prediction, Review
from services.analytics_aspects import aspect_detail, aspect_leaderboard, top_aspects


class AnalyticsCanonicalAspectTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_analytics_merges_equivalent_explicit_and_implicit_aspects(self) -> None:
        db = self.make_db()
        review = Review(
            text="I loved the atmosphere, however, the prices were a bit expensive for what you get.",
            domain="restaurant",
            product_id="p1",
        )
        db.add(review)
        db.flush()

        explicit = Prediction(
            review_id=review.id,
            aspect_raw="prices",
            aspect_cluster="prices",
            sentiment="negative",
            confidence=0.91,
            source="explicit",
        )
        explicit.evidence_spans.append(EvidenceSpan(start_char=37, end_char=43, snippet="prices"))
        implicit = Prediction(
            review_id=review.id,
            aspect_raw="price",
            aspect_cluster="price",
            sentiment="negative",
            confidence=0.83,
            source="implicit",
        )
        implicit.evidence_spans.append(EvidenceSpan(start_char=65, end_char=77, snippet="what you get"))
        db.add_all([explicit, implicit])
        db.commit()

        self.assertEqual(top_aspects(db, 10, None, None, "restaurant"), [{"aspect": "price", "count": 2}])

        leaderboard = aspect_leaderboard(db, limit=10, domain="restaurant")
        self.assertEqual(len(leaderboard), 1)
        self.assertEqual(leaderboard[0]["aspect"], "price")
        self.assertEqual(leaderboard[0]["frequency"], 2)
        self.assertEqual(leaderboard[0]["implicit_pct"], 50.0)

        detail = aspect_detail(db, "prices", domain="restaurant")
        self.assertEqual(detail["aspect"], "price")
        self.assertEqual(detail["frequency"], 2)
        self.assertEqual(detail["explicit_count"], 1)
        self.assertEqual(detail["implicit_count"], 1)


if __name__ == "__main__":
    unittest.main()
