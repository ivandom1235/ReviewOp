from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class RowContractsTests(unittest.TestCase):
    def test_row_lifecycle_record_round_trips_to_dict(self) -> None:
        from row_contracts import RowLifecycleRecord

        record = RowLifecycleRecord(
            row_id="row-1",
            state="quality_scored",
            reason_codes=["grounded", "useful"],
            details={"quality_score": 0.81},
        )

        self.assertEqual(
            record.to_dict(),
            {
                "row_id": "row-1",
                "state": "quality_scored",
                "reason_codes": ["grounded", "useful"],
                "details": {"quality_score": 0.81},
            },
        )

    def test_quality_decision_serializes_core_scores(self) -> None:
        from row_contracts import QualityDecision

        decision = QualityDecision(
            decision="benchmark_silver",
            quality_score=0.74,
            usefulness_score=0.62,
            redundancy_score=0.11,
            reason_codes=["borderline_grounding", "rare_domain"],
        )

        self.assertEqual(
            decision.to_dict(),
            {
                "decision": "benchmark_silver",
                "quality_score": 0.74,
                "usefulness_score": 0.62,
                "redundancy_score": 0.11,
                "reason_codes": ["borderline_grounding", "rare_domain"],
            },
        )

    def test_stage_profile_serializes_runtime_metrics(self) -> None:
        from row_contracts import StageProfile

        profile = StageProfile(
            stage_name="implicit_scoring",
            rows_in=50,
            rows_out=42,
            elapsed_ms=12.5,
            cache_hits=8,
            cache_misses=3,
            extra={"provider": "mock"},
        )

        self.assertEqual(
            profile.to_dict(),
            {
                "stage_name": "implicit_scoring",
                "rows_in": 50,
                "rows_out": 42,
                "elapsed_ms": 12.5,
                "cache_hits": 8,
                "cache_misses": 3,
                "extra": {"provider": "mock"},
            },
        )

    def test_restaurant_aspect_aliases_collapse_to_canonical_labels(self) -> None:
        from aspect_registry import canonicalize_domain_aspect

        self.assertEqual(
            canonicalize_domain_aspect(domain="restaurant", aspect_label="wait staff", surface_rationale_tag="wait staff was slow"),
            "service_speed",
        )
        self.assertEqual(
            canonicalize_domain_aspect(domain="restaurant", aspect_label="decor", surface_rationale_tag="nice decor"),
            "ambience",
        )
        self.assertEqual(
            canonicalize_domain_aspect(domain="restaurant", aspect_label="meals", surface_rationale_tag="the meals were cold"),
            "food_quality",
        )

    def test_generic_canonical_labels_do_not_enter_the_registry(self) -> None:
        from aspect_registry import canonicalize_domain_aspect

        self.assertIsNone(
            canonicalize_domain_aspect(domain="electronics", aspect_label="quality", surface_rationale_tag="quality"),
        )
        self.assertIsNone(
            canonicalize_domain_aspect(domain="telecom", aspect_label="good", surface_rationale_tag="good"),
        )


if __name__ == "__main__":
    unittest.main()
