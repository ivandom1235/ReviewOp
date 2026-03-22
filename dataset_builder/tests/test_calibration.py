from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(ROOT))

from calibration import ConfidenceCalibrator, build_calibration_summary


class CalibrationTests(unittest.TestCase):
    def test_fit_and_apply_preserves_raw_and_marks_uncertain(self) -> None:
        rows = [
            {
                "labels": [
                    {
                        "aspect_type": "explicit",
                        "confidence": 0.92,
                        "evidence_quality": 0.9,
                        "is_sentence_fallback": False,
                        "sentiment_ambiguous": False,
                        "sentiment_unresolved": False,
                    },
                    {
                        "aspect_type": "implicit",
                        "confidence": 0.61,
                        "evidence_quality": 0.45,
                        "is_sentence_fallback": True,
                        "sentiment_ambiguous": True,
                        "sentiment_unresolved": False,
                    },
                ]
            }
        ]

        calibrator = ConfidenceCalibrator.fit(rows, threshold=0.75, blend=0.55)
        self.assertEqual(calibrator.n_bins, 10)

        applied = calibrator.apply(rows)
        explicit = applied[0]["labels"][0]
        implicit = applied[0]["labels"][1]

        self.assertIn("raw_confidence", explicit)
        self.assertIn("calibrated_confidence", explicit)
        self.assertLessEqual(explicit["calibrated_confidence"], 0.99)
        self.assertFalse(explicit["uncertain"])
        self.assertTrue(implicit["uncertain"])
        self.assertLess(implicit["calibrated_confidence"], 0.75)

        summary = build_calibration_summary(applied)
        self.assertEqual(summary["n_bins"], 10)
        self.assertGreaterEqual(summary["total_points"], 2)


if __name__ == "__main__":
    unittest.main()
