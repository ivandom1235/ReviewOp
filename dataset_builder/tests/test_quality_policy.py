from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class QualityPolicyTests(unittest.TestCase):
    def test_quality_decision_routes_borderline_rows_to_silver(self) -> None:
        from quality_policy import quality_decision_record

        row = {
            "id": "r1",
            "domain": "telecom",
            "implicit": {
                "aspects": ["battery"],
                "spans": [{"support_type": "symptom_based", "confidence": 0.61}],
                "aspect_confidence": {"battery": 0.61},
                "review_reason": "low_confidence",
                "implicit_ready": True,
            },
            "explicit": {"aspects": []},
            "abstain_acceptable": False,
            "novel_acceptable": False,
        }

        record = quality_decision_record(
            row,
            min_confidence=0.7,
            recovery_confidence_threshold=0.5,
            accepted_support_types=("exact", "near_exact", "symptom_based"),
        )

        self.assertEqual(record["decision"], "silver")
        self.assertEqual(record["bucket"], "borderline")
        self.assertIn("low_confidence", record["reason_codes"])

    def test_quality_decision_routes_strict_rows_to_train_keep(self) -> None:
        from quality_policy import quality_decision_record

        row = {
            "id": "r2",
            "domain": "electronics",
            "implicit": {
                "aspects": ["battery"],
                "spans": [{"support_type": "exact", "confidence": 0.93}],
                "aspect_confidence": {"battery": 0.93},
                "implicit_ready": True,
            },
            "explicit": {"aspects": []},
            "abstain_acceptable": True,
            "novel_acceptable": False,
        }

        record = quality_decision_record(
            row,
            min_confidence=0.7,
            recovery_confidence_threshold=0.5,
            accepted_support_types=("exact", "near_exact", "symptom_based"),
        )

        self.assertEqual(record["decision"], "train_keep")
        self.assertEqual(record["bucket"], "train_keep")
        self.assertEqual(record["reason_codes"], [])
        self.assertGreater(record["usefulness_score"], 0.0)

    def test_quality_decision_routes_malformed_rows_to_reject(self) -> None:
        from quality_policy import quality_decision_record

        row = {
            "id": "r3",
            "domain": "telecom",
            "implicit": {
                "aspects": ["general"],
                "spans": [],
                "aspect_confidence": {},
                "implicit_ready": False,
                "review_reason": "fallback_general",
            },
            "explicit": {"aspects": ["general"]},
            "abstain_acceptable": False,
            "novel_acceptable": False,
        }

        record = quality_decision_record(
            row,
            min_confidence=0.7,
            recovery_confidence_threshold=0.5,
            accepted_support_types=("exact", "near_exact", "symptom_based"),
        )

        self.assertEqual(record["decision"], "hard_reject")
        self.assertIn("fallback_general", record["reason_codes"])
        self.assertIn("general_only", record["reason_codes"])


if __name__ == "__main__":
    unittest.main()
