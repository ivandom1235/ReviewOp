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
from services.graph_builders import build_batch_aspect_graph
import services.review_pipeline as review_pipeline


class FakeThread:
    created = []

    def __init__(self, *, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        FakeThread.created.append(self)

    def start(self):
        self.started = True


class GraphRefreshDebounceTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeThread.created = []
        review_pipeline._GRAPH_REFRESH_THREADS.clear()

    def test_same_scope_refresh_reuses_in_flight_thread(self) -> None:
        with patch("services.review_pipeline.threading.Thread", FakeThread):
            first = review_pipeline.schedule_corpus_graph_refresh("electronics")
            second = review_pipeline.schedule_corpus_graph_refresh("electronics")

        self.assertIs(first, second)
        self.assertEqual(len(FakeThread.created), 1)

    def test_different_scopes_can_refresh_independently(self) -> None:
        with patch("services.review_pipeline.threading.Thread", FakeThread):
            first = review_pipeline.schedule_corpus_graph_refresh("electronics")
            second = review_pipeline.schedule_corpus_graph_refresh("restaurant")

        self.assertIsNot(first, second)
        self.assertEqual(len(FakeThread.created), 2)


class GraphMetadataPropagationTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_batch_graph_includes_prediction_reliability_signals(self) -> None:
        db = self.make_db()
        review = Review(text="The battery is weird.", domain="electronics", product_id="p1")
        db.add(review)
        db.flush()
        db.add(
            Prediction(
                review_id=review.id,
                aspect_raw="battery life",
                aspect_cluster="battery_life",
                sentiment="negative",
                confidence=0.82,
                contradiction_score=0.47,
                contradiction_types=["evidence_mismatch"],
                quarantine_status="watch",
            )
        )
        db.commit()

        payload = build_batch_aspect_graph(db, domain="electronics", graph_mode="accepted")
        node = next(item for item in payload["nodes"] if item["id"] == "battery_life")
        self.assertEqual(node["contradiction_score"], 0.47)
        self.assertEqual(node["contradiction_types"], ["evidence_mismatch"])
        self.assertEqual(node["quarantine_status"], "watch")
        self.assertAlmostEqual(node["graph_support_score"], 0.53, places=2)

    def test_novel_side_graph_uses_row_specific_reliability_signals(self) -> None:
        db = self.make_db()
        review = Review(text="The hinge sparks and the camera blurs.", domain="electronics", product_id="p2")
        db.add(review)
        db.flush()
        db.add(
            NovelCandidate(
                review_id=review.id,
                aspect="hinge_sparks",
                novelty_score=0.91,
                confidence=0.7,
                evidence="hinge sparks",
                contradiction_score=0.61,
                contradiction_types=["prototype_instability"],
                quarantine_status="quarantined",
            )
        )
        db.add(
            NovelCandidate(
                review_id=review.id,
                aspect="camera_blur",
                novelty_score=0.84,
                confidence=0.64,
                evidence="camera blurs",
                contradiction_score=0.22,
                contradiction_types=["low_support"],
                quarantine_status="watch",
            )
        )
        db.commit()

        payload = build_batch_aspect_graph(db, domain="electronics", graph_mode="novel_side")
        node_map = {item["id"]: item for item in payload["nodes"]}
        self.assertEqual(node_map["hinge_sparks"]["contradiction_score"], 0.61)
        self.assertEqual(node_map["hinge_sparks"]["contradiction_types"], ["prototype_instability"])
        self.assertEqual(node_map["hinge_sparks"]["quarantine_status"], "quarantined")
        self.assertAlmostEqual(node_map["hinge_sparks"]["graph_support_score"], 0.39, places=2)
        self.assertEqual(node_map["camera_blur"]["contradiction_score"], 0.22)
        self.assertEqual(node_map["camera_blur"]["contradiction_types"], ["low_support"])
        self.assertEqual(node_map["camera_blur"]["quarantine_status"], "watch")

    def test_batch_graph_derives_contradiction_score_from_conflicting_sentiments(self) -> None:
        db = self.make_db()
        review = Review(text="Battery is great, but the battery also failed.", domain="electronics", product_id="p3")
        db.add(review)
        db.flush()
        db.add(
            Prediction(
                review_id=review.id,
                aspect_raw="battery life",
                aspect_cluster="battery_life",
                sentiment="positive",
                confidence=0.82,
            )
        )
        db.add(
            Prediction(
                review_id=review.id,
                aspect_raw="battery life",
                aspect_cluster="battery_life",
                sentiment="negative",
                confidence=0.79,
            )
        )
        db.commit()

        payload = build_batch_aspect_graph(db, domain="electronics", graph_mode="accepted")
        node = next(item for item in payload["nodes"] if item["id"] == "battery_life")
        self.assertGreater(node["contradiction_score"], 0.0)
        self.assertIn("sentiment_conflict", node["contradiction_types"])


if __name__ == "__main__":
    unittest.main()
