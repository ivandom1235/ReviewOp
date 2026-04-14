from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_dataset import (
    _build_quality_analysis_artifact,
    _collect_recoverable_quality_rows,
    _quality_recovery_eligible,
)


def _make_row(*, row_id: str, review_reason: str, confidence: float, supports: list[str], aspect: str = "performance") -> dict[str, object]:
    return {
        "id": row_id,
        "split": "train",
        "source_file": "Laptop_train.csv",
        "source_text": "The performance is decent but the charger is not great.",
        "domain": "electronics",
        "implicit": {
            "needs_review": True,
            "review_reason": review_reason,
            "implicit_ready": True,
            "aspects": [aspect],
            "spans": [
                {
                    "support_type": support,
                    "confidence": confidence,
                }
                for support in supports
            ],
            "aspect_confidence": {aspect: confidence},
        },
        "explicit": {
            "aspects": [],
        },
    }


class QuarantineRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.support_types = ("exact", "near_exact", "gold")

    def test_recovery_eligibility_distinguishes_borderline_from_terminal(self) -> None:
        low_confidence_row = _make_row(
            row_id="row-low",
            review_reason="low_confidence",
            confidence=0.54,
            supports=["exact"],
        )
        fallback_row = _make_row(
            row_id="row-fallback",
            review_reason="fallback_general",
            confidence=0.9,
            supports=["exact"],
        )

        low_confidence_ok, low_confidence_reasons = _quality_recovery_eligible(
            low_confidence_row,
            min_confidence=0.58,
            recovery_confidence_threshold=0.52,
            accepted_support_types=self.support_types,
            candidate_aspects_by_domain=None,
            allow_weak_support=True,
        )
        fallback_ok, fallback_reasons = _quality_recovery_eligible(
            fallback_row,
            min_confidence=0.58,
            recovery_confidence_threshold=0.52,
            accepted_support_types=self.support_types,
            candidate_aspects_by_domain=None,
            allow_weak_support=True,
        )

        self.assertTrue(low_confidence_ok)
        self.assertIn("low_confidence", low_confidence_reasons)
        self.assertFalse(fallback_ok)
        self.assertIn("fallback_general", fallback_reasons)

    def test_recoverable_rows_are_extracted_from_quality_quarantine(self) -> None:
        weak_support_row = _make_row(
            row_id="row-weak",
            review_reason="weak_support",
            confidence=0.53,
            supports=["exact"],
        )
        low_confidence_row = _make_row(
            row_id="row-low",
            review_reason="low_confidence",
            confidence=0.54,
            supports=["exact"],
        )
        terminal_row = _make_row(
            row_id="row-terminal",
            review_reason="fallback_general",
            confidence=0.9,
            supports=["exact"],
        )

        recoverable_rows, stats = _collect_recoverable_quality_rows(
            [weak_support_row, low_confidence_row, terminal_row],
            min_confidence=0.58,
            recovery_confidence_threshold=0.52,
            accepted_support_types=self.support_types,
            candidate_aspects_by_domain=None,
            allow_weak_support=True,
        )

        self.assertEqual([row["id"] for row in recoverable_rows], ["row-weak", "row-low"])
        self.assertEqual(stats["borderline_rows"], 2)
        self.assertEqual(stats["recoverable_rows"], 2)
        self.assertEqual(stats["terminal_rows"], 1)

    def test_quality_artifact_tracks_recoverable_rows(self) -> None:
        borderline_row = _make_row(
            row_id="row-low",
            review_reason="low_confidence",
            confidence=0.54,
            supports=["exact"],
        )
        weak_support_row = _make_row(
            row_id="row-weak",
            review_reason="weak_support",
            confidence=0.53,
            supports=["exact"],
        )
        rejected_row = _make_row(
            row_id="row-fallback",
            review_reason="fallback_general",
            confidence=0.9,
            supports=["exact"],
        )

        artifact = _build_quality_analysis_artifact(
            [borderline_row, weak_support_row, rejected_row],
            [],
            min_confidence=0.58,
            recovery_confidence_threshold=0.52,
            accepted_support_types=self.support_types,
            candidate_aspects_by_domain=None,
            allow_weak_support_in_recovery=True,
        )

        self.assertEqual(artifact["borderline_count"], 2)
        self.assertEqual(artifact["recoverable_count"], 2)
        self.assertEqual(artifact["rejected_count"], 1)
        self.assertEqual([item["row"]["id"] for item in artifact["recoverable_rows"]], ["row-low", "row-weak"])


if __name__ == "__main__":
    unittest.main()
