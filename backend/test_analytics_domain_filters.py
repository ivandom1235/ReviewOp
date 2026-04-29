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
import services.analytics_aspects as analytics_aspects
from services.analytics_kpis import overview
from services.analytics_segments import segment_drilldown


class AnalyticsDomainFilterTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def add_prediction(self, db, *, domain: str, product_id: str, aspect: str, sentiment: str, confidence: float = 0.9):
        review = Review(text=f"{aspect} review", domain=domain, product_id=product_id)
        db.add(review)
        db.flush()
        prediction = Prediction(
            review_id=review.id,
            aspect_raw=aspect,
            aspect_cluster=aspect,
            sentiment=sentiment,
            confidence=confidence,
            source="explicit",
        )
        prediction.evidence_spans.append(EvidenceSpan(start_char=0, end_char=len(aspect), snippet=aspect))
        db.add(prediction)
        return review

    def test_overview_filters_all_prediction_aggregates_by_domain(self) -> None:
        db = self.make_db()
        self.add_prediction(db, domain="restaurant", product_id="r1", aspect="price", sentiment="negative", confidence=0.9)
        self.add_prediction(db, domain="electronics", product_id="e1", aspect="battery", sentiment="positive", confidence=0.3)
        db.commit()

        payload = overview(db, None, None, "restaurant")

        self.assertEqual(payload["total_reviews"], 1)
        self.assertEqual(payload["total_aspect_mentions"], 1)
        self.assertEqual(payload["unique_aspects_raw"], 1)
        self.assertEqual(payload["avg_confidence"], 0.9)
        self.assertEqual(payload["sentiment_counts"], {"positive": 0, "neutral": 0, "negative": 1})

    def test_segment_drilldown_applies_domain_filter_before_limit(self) -> None:
        db = self.make_db()
        self.add_prediction(db, domain="restaurant", product_id="r1", aspect="price", sentiment="negative")
        self.add_prediction(db, domain="electronics", product_id="e1", aspect="battery", sentiment="negative")
        db.commit()

        rows = segment_drilldown(db, domain="restaurant")

        product_rows = [row for row in rows if row["segment_type"] == "product_id"]
        self.assertEqual([row["segment_value"] for row in product_rows], ["r1"])

    def test_canonical_aspect_filter_limits_rows_before_python_canonical_check(self) -> None:
        db = self.make_db()
        self.add_prediction(db, domain="restaurant", product_id="r1", aspect="prices", sentiment="negative")
        for index in range(30):
            self.add_prediction(
                db,
                domain="restaurant",
                product_id=f"other-{index}",
                aspect=f"other aspect {index}",
                sentiment="positive",
            )
        db.commit()

        calls = 0
        real_canonical_aspect = analytics_aspects.canonical_aspect

        def counting_canonical_aspect(prediction):
            nonlocal calls
            calls += 1
            return real_canonical_aspect(prediction)

        analytics_aspects.canonical_aspect = counting_canonical_aspect
        try:
            rows = analytics_aspects._prediction_rows(db, aspect="price", domain="restaurant")
        finally:
            analytics_aspects.canonical_aspect = real_canonical_aspect

        self.assertEqual(len(rows), 1)
        self.assertLessEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
