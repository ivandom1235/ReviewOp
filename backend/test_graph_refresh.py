from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

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


if __name__ == "__main__":
    unittest.main()
