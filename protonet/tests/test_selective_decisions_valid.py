from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class SelectiveDecisionValidTests(unittest.TestCase):
    def test_calibration_accepts_balanced_validation_rows(self) -> None:
        from selective_decisions import calibrate_novelty_thresholds

        result = calibrate_novelty_thresholds(
            novelty_calibration={
                "thresholds": {"T_known": 0.28, "T_novel": 0.71},
                "scorer": "distance_energy",
            },
            default_known=0.35,
            default_novel=0.65,
            validation_rows=[
                {"novel_acceptable": False},
                {"novel_acceptable": True},
                {"novel_acceptable": False},
                {"novel_acceptable": True},
            ],
        )

        self.assertTrue(result["applicable"])
        self.assertGreaterEqual(result["T_novel"], result["T_known"])
        self.assertIsNone(result["reason"])


if __name__ == "__main__":
    unittest.main()
