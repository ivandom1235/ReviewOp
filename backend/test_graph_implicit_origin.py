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
from services.graph_builders import build_batch_aspect_graph, build_single_review_graph


class GraphImplicitOriginTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_batch_graph_uses_prediction_source_for_implicit_count(self) -> None:
        db = self.make_db()
        review = Review(text="battery life is bad", domain="electronics", product_id="p1")
        db.add(review)
        db.flush()

        pred = Prediction(
            review_id=review.id,
            aspect_raw="battery_life",
            aspect_cluster="battery_life",
            sentiment="negative",
            confidence=0.92,
            source="implicit",
        )
        pred.evidence_spans.append(
            EvidenceSpan(
                start_char=0,
                end_char=12,
                snippet="battery life",
            )
        )
        db.add(pred)
        db.commit()

        graph = build_batch_aspect_graph(db, domain="electronics")
        node = next((n for n in graph.get("nodes", []) if n.get("id") == "battery_life"), None)

        self.assertIsNotNone(node)
        self.assertEqual(node.get("implicit_count"), 1)
        self.assertEqual(node.get("explicit_count"), 0)

    def test_single_review_graph_collapses_canonical_variants(self) -> None:
        db = self.make_db()
        text = "I loved the atmosphere, however, the prices were a bit expensive for what you get."
        review = Review(text=text, domain="restaurant", product_id="p1")
        db.add(review)
        db.flush()

        atmosphere = Prediction(
            review_id=review.id,
            aspect_raw="atmosphere",
            aspect_cluster="atmosphere",
            sentiment="positive",
            confidence=0.91,
            source="explicit",
        )
        atmosphere.evidence_spans.append(EvidenceSpan(start_char=12, end_char=22, snippet="atmosphere"))
        prices = Prediction(
            review_id=review.id,
            aspect_raw="prices",
            aspect_cluster="prices",
            sentiment="negative",
            confidence=0.88,
            source="explicit",
        )
        prices.evidence_spans.append(EvidenceSpan(start_char=37, end_char=43, snippet="prices"))
        price = Prediction(
            review_id=review.id,
            aspect_raw="price",
            aspect_cluster="price",
            sentiment="negative",
            confidence=0.78,
            source="implicit",
        )
        price.evidence_spans.append(EvidenceSpan(start_char=65, end_char=77, snippet="what you get"))
        db.add_all([atmosphere, prices, price])
        db.commit()

        graph = build_single_review_graph(db, review.id)
        nodes = {node["id"]: node for node in graph.get("nodes", [])}

        self.assertEqual(sorted(nodes), ["atmosphere", "price"])
        self.assertEqual(nodes["price"]["explicit_count"], 1)
        self.assertEqual(nodes["price"]["implicit_count"], 1)
        self.assertEqual(nodes["price"]["origin"], "mixed")


if __name__ == "__main__":
    unittest.main()
