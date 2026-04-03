from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from build_dataset import _merge_gold_labels
from evaluation import gold_eval


class GoldAnnotationTests(unittest.TestCase):
    def test_merge_gold_labels_by_record_id_and_text(self) -> None:
        rows = [
            {"id": "row1", "domain": "restaurant", "source_text": "Great pizza", "gold_labels": []},
            {"id": "row2", "domain": "laptop", "source_text": "Battery is weak", "gold_labels": []},
            {"id": "row3", "domain": "laptop", "source_text": "Screen is bright", "gold_labels": [{"aspect": "display quality"}]},
        ]
        annotations = [
            {"record_id": "row1", "domain": "restaurant", "text": "Great pizza", "gold_labels": [{"aspect": "food quality", "sentiment": "positive", "start": 0, "end": 11}]},
            {"record_id": None, "domain": "laptop", "text": "Battery is weak", "gold_labels": [{"aspect": "power", "sentiment": "negative", "start": 0, "end": 15}]},
        ]
        merged = _merge_gold_labels(rows, annotations)
        self.assertEqual(len(merged[0]["gold_labels"]), 1)
        self.assertEqual(len(merged[1]["gold_labels"]), 1)
        self.assertEqual(len(merged[2]["gold_labels"]), 1)

    def test_gold_eval_returns_non_empty_scores(self) -> None:
        rows = [
            {
                "id": "r1",
                "domain": "restaurant",
                "gold_labels": [{"aspect": "food quality", "sentiment": "positive", "start": 0, "end": 5}],
                "implicit": {
                    "aspects": ["food quality"],
                    "aspect_sentiments": {"food quality": "positive"},
                    "spans": [{"latent_aspect": "food quality", "start_char": 0, "end_char": 5}],
                },
            }
        ]
        metrics = gold_eval(rows)
        self.assertTrue(metrics["has_gold_labels"])
        self.assertEqual(metrics["num_rows_with_gold"], 1)
        self.assertGreaterEqual(metrics["aspect_f1"], 0.0)
        self.assertGreaterEqual(metrics["sentiment_f1"], 0.0)
        self.assertGreaterEqual(metrics["span_overlap_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
