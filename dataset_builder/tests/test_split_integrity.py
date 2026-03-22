from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(ROOT))

from data_ops import enforce_split_integrity, leakage_report, split_map


class SplitIntegrityTests(unittest.TestCase):
    def test_enforce_split_integrity_dedupes_and_aligns_groups(self) -> None:
        rows = [
            {"review_id": "r1", "group_id": "g1", "product_id": "p1", "clean_text": "Great battery life and display.", "split": "train"},
            {"review_id": "r2", "group_id": "g1", "product_id": "p1", "clean_text": "Great battery life and display.", "split": "test"},
            {"review_id": "r3", "group_id": "g2", "product_id": "p2", "clean_text": "Works well overall.", "split": "val"},
            {"review_id": "r4", "group_id": "g3", "product_id": "p3", "clean_text": "Works well overall.", "split": "test"},
        ]

        cleaned = enforce_split_integrity(rows, similarity_threshold=0.8)
        grouped = split_map(cleaned)
        report = leakage_report(grouped)

        # duplicate group rows should not survive in different splits
        splits = {row["review_id"]: row["split"] for row in cleaned}
        self.assertIn("r1", splits)
        self.assertNotIn("r2", splits)

        # near-duplicate test rows should not survive in different splits
        self.assertEqual(report["near_exact_text_overlap"]["train_test"], 0)
        self.assertEqual(report["near_exact_text_overlap"]["train_val"], 0)
        self.assertEqual(report["near_exact_text_overlap"]["val_test"], 0)


if __name__ == "__main__":
    unittest.main()
