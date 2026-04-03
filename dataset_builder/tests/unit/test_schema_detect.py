from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from schema_detect import detect_schema


class SchemaDetectTests(unittest.TestCase):
    def test_detect_schema_finds_text_and_fingerprint(self) -> None:
        frame = pd.DataFrame(
            {
                "review_text": ["Great battery life", "Slow but usable"],
                "rating": [5, 2],
                "brand": ["A", "B"],
            }
        )
        profile = detect_schema(frame)
        self.assertEqual(profile.primary_text_column, "review_text")
        self.assertTrue(profile.schema_fingerprint)
        self.assertIn("review_text", profile.implicit_ready_columns)
        self.assertTrue("rating" in profile.numeric_columns or "rating" in profile.categorical_columns)


if __name__ == "__main__":
    unittest.main()
