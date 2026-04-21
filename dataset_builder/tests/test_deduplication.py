from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class DeduplicationTests(unittest.TestCase):
    def test_semantic_cluster_id_groups_near_duplicates(self) -> None:
        from deduplication import semantic_cluster_id

        row_a = {
            "domain": "telecom",
            "sentiment": "negative",
            "source_text": "Battery dies fast",
            "implicit": {"aspects": ["battery"], "dominant_sentiment": "negative", "spans": [{"snippet": "battery dies fast"}]},
        }
        row_b = {
            "domain": "telecom",
            "sentiment": "negative",
            "source_text": "Battery dies fast!",
            "implicit": {"aspects": ["battery"], "dominant_sentiment": "negative", "spans": [{"snippet": "battery dies fast"}]},
        }

        self.assertEqual(semantic_cluster_id(row_a), semantic_cluster_id(row_b))

    def test_deduplicate_by_cluster_applies_split_limit(self) -> None:
        from deduplication import deduplicate_by_cluster

        rows = [
            {"split": "train", "domain": "telecom", "source_text": "Battery dies fast", "implicit": {"aspects": ["battery"], "dominant_sentiment": "negative", "spans": [{"snippet": "battery dies fast"}]}},
            {"split": "train", "domain": "telecom", "source_text": "Battery dies fast!", "implicit": {"aspects": ["battery"], "dominant_sentiment": "negative", "spans": [{"snippet": "battery dies fast"}]}},
            {"split": "val", "domain": "telecom", "source_text": "Battery dies fast", "implicit": {"aspects": ["battery"], "dominant_sentiment": "negative", "spans": [{"snippet": "battery dies fast"}]}},
        ]

        kept, stats = deduplicate_by_cluster(rows, max_per_cluster=2, max_per_split=1)

        self.assertEqual(len(kept), 2)
        self.assertEqual(stats["split_limit_rows_removed"], 1)
        self.assertEqual(stats["duplicate_rows_removed"], 0)


if __name__ == "__main__":
    unittest.main()
