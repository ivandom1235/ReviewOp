from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class SelectiveDecisionTests(unittest.TestCase):
    def test_calibration_skips_without_novel_support(self) -> None:
        from selective_decisions import calibrate_novelty_thresholds

        result = calibrate_novelty_thresholds(
            novelty_calibration={"thresholds": {"T_known": 0.2, "T_novel": 0.8}, "scorer": "distance_energy"},
            default_known=0.35,
            default_novel=0.65,
            validation_rows=[{"novel_acceptable": False}, {"novel_acceptable": False}],
        )

        self.assertFalse(result["applicable"])
        self.assertEqual(result["T_known"], 0.35)
        self.assertEqual(result["T_novel"], 0.65)
        self.assertEqual(result["reason"], "insufficient_validation_support")

    def test_decide_selective_routing_labels_boundary_abstain_explicitly(self) -> None:
        from selective_decisions import decide_selective_routing

        decision = decide_selective_routing(
            novelty_score=0.61,
            selective_confidence=0.8,
            abstain_threshold=0.25,
            known_threshold=0.5,
            novel_threshold=0.8,
        )

        self.assertEqual(decision.decision, "abstain")
        self.assertEqual(decision.decision_band, "boundary")
        self.assertEqual(decision.abstain_reason, "boundary_uncertain_novelty")


if __name__ == "__main__":
    unittest.main()
