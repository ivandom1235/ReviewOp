from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class ReportSidecarTests(unittest.TestCase):
    def test_why_not_promoted_sidecar_summarizes_reasons(self) -> None:
        from exporters import _why_not_promoted_sidecar

        sidecar = _why_not_promoted_sidecar(
            {
                "decision_counts": {"train_keep": 2, "review_queue": 3, "hard_reject": 1},
                "review_queue_rows": [
                    {"reason_codes": ["weak_grounding", "rare_domain"]},
                    {"reason_codes": ["weak_grounding"]},
                ],
                "hard_reject_rows": [{}, {}],
                "train_keep_rows": [{}, {}],
                "silver_rows": [{}],
            }
        )

        self.assertEqual(sidecar["decision_counts"]["review_queue"], 3)
        self.assertEqual(sidecar["review_queue_rows"], 2)
        self.assertEqual(sidecar["hard_reject_rows"], 2)
        self.assertEqual(sidecar["train_keep_rows"], 2)
        self.assertEqual(sidecar["silver_rows"], 1)
        self.assertEqual(
            sidecar["top_reasons"],
            [
                {"reason": "weak_grounding", "count": 2},
                {"reason": "rare_domain", "count": 1},
            ],
        )
        self.assertEqual(sidecar["implicit_rejection_reason_counts"], {})

    def test_why_not_promoted_sidecar_exposes_implicit_rejection_reason_counts(self) -> None:
        from exporters import _why_not_promoted_sidecar

        sidecar = _why_not_promoted_sidecar(
            {
                "decision_counts": {"train_keep": 1, "review_queue": 1, "hard_reject": 0},
                "review_queue_rows": [{"reason_codes": ["low_mapping_confidence"]}],
                "hard_reject_rows": [],
                "train_keep_rows": [{}],
                "silver_rows": [],
                "implicit_rejection_reason_counts": {"low_mapping_confidence": 4, "weak_evidence": 2},
            }
        )

        self.assertEqual(
            sidecar["implicit_rejection_reason_counts"],
            {"low_mapping_confidence": 4, "weak_evidence": 2},
        )


if __name__ == "__main__":
    unittest.main()
