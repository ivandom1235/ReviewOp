from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from explicit_features import build_explicit_row, fit_explicit_artifacts


class ExplicitFeatureTests(unittest.TestCase):
    def test_explicit_features_include_text_stats(self) -> None:
        frame = pd.DataFrame(
            {
                "num": [1.0, 2.0],
                "cat": ["x", "y"],
                "review_text": ["Bright screen and great battery", "Slow keyboard"],
            }
        )
        artifacts = fit_explicit_artifacts(frame, ["num"], ["cat"])
        row = build_explicit_row(
            {"id": "1", "split": "train", "num": 1.0, "cat": "x", "review_text": "Bright screen and great battery"},
            artifacts=artifacts,
            numeric_columns=["num"],
            categorical_columns=["cat"],
            datetime_columns=[],
            text_column="review_text",
        )
        self.assertIn("explicit", row)
        self.assertIn("review_text_stats", row["explicit"])
        self.assertEqual(row["explicit"]["cat"], "x")


if __name__ == "__main__":
    unittest.main()
