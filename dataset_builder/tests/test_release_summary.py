from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class ReleaseSummaryTests(unittest.TestCase):
    def test_release_summary_includes_stable_counts(self) -> None:
        from exporters import _release_summary

        summary = _release_summary(
            {
                "pipeline_version": "v6",
                "generated_at": "2026-04-20T00:00:00Z",
                "run_profile": "research",
                "artifact_mode": "research_release",
                "output_quality": {"silver_count": 3, "train_keep_count": 10, "hard_reject_count": 2, "recoverable_count": 1},
                "strict_artifacts": {"strict_train_rows": 42},
                "blocking_reasons": [{"code": "TRAIN_DOMAIN_LEAKAGE"}],
                "validation": {"benchmark_artifact_counts_match": True},
                "benchmark_artifact_counts": {"train": 10, "val": 2, "test": 2},
                "topup_effectiveness": {"coverage_of_shortfall": 0.75},
                "size_recovery_stage": "C",
                "size_recovery_shortfall_remaining": 0,
            },
            {"silver_count": 3, "train_keep_count": 10, "hard_reject_count": 2, "recoverable_count": 1},
        )

        self.assertEqual(summary["gold_rows"], 42)
        self.assertEqual(summary["silver_rows"], 3)
        self.assertEqual(summary["train_keep_rows"], 10)
        self.assertEqual(summary["hard_reject_rows"], 2)
        self.assertEqual(summary["recoverable_rows"], 1)
        self.assertTrue(summary["blocked"])
        self.assertEqual(summary["blocking_codes"], ["TRAIN_DOMAIN_LEAKAGE"])
        self.assertTrue(summary["benchmark_artifact_counts_match"])

    def test_sampled_research_runs_are_not_reported_as_debug_blocked(self) -> None:
        from build_dataset import _resolve_promotion_eligibility

        self.assertEqual(
            _resolve_promotion_eligibility(
                run_profile="research",
                sampled=True,
                validation={
                    "train_target_blocking_failure": False,
                    "train_general_excluded": True,
                    "train_domain_leakage_ok": True,
                    "no_generic_aspects": True,
                    "no_rejected_aspects": True,
                    "strict_explicit_contamination_ok": True,
                    "strict_boundary_fp_ok": True,
                    "strict_h2_h3_ok": True,
                    "strict_multi_aspect_ok": True,
                    "strict_challenge_ok": True,
                    "benchmark_val_non_empty": True,
                    "benchmark_grounded_evidence_ok": True,
                    "benchmark_duplicate_rate_ok": True,
                    "benchmark_duplicate_logical_row_rate_adjusted_ok": True,
                    "benchmark_thermal_share_ok": True,
                    "benchmark_ontology_compatibility_ok": True,
                    "sentiment_mismatch_rate_ok": True,
                    "promotion_guard_ok": True,
                    "benchmark_artifact_counts_match": True,
                },
            ),
            "diagnostic_slice",
        )

    def test_logical_duplication_gate_blocks_bad_benchmark_slices(self) -> None:
        from build_dataset import _resolve_promotion_eligibility

        self.assertEqual(
            _resolve_promotion_eligibility(
                run_profile="research",
                sampled=False,
                validation={
                    "train_target_blocking_failure": False,
                    "train_general_excluded": True,
                    "train_domain_leakage_ok": True,
                    "no_generic_aspects": True,
                    "no_rejected_aspects": True,
                    "strict_explicit_contamination_ok": True,
                    "strict_boundary_fp_ok": True,
                    "strict_h2_h3_ok": True,
                    "strict_multi_aspect_ok": True,
                    "strict_challenge_ok": True,
                    "benchmark_val_non_empty": True,
                    "benchmark_grounded_evidence_ok": True,
                    "benchmark_duplicate_rate_ok": True,
                    "benchmark_duplicate_logical_row_rate_adjusted_ok": False,
                    "benchmark_thermal_share_ok": True,
                    "benchmark_ontology_compatibility_ok": True,
                    "sentiment_mismatch_rate_ok": True,
                    "promotion_guard_ok": True,
                    "benchmark_artifact_counts_match": True,
                },
            ),
            "blocked_quality",
        )

    def test_debug_runs_remain_blocked(self) -> None:
        from build_dataset import _resolve_promotion_eligibility

        self.assertEqual(
            _resolve_promotion_eligibility(run_profile="debug", sampled=True, validation={}),
            "blocked_debug",
        )

    def test_family_floor_applies_to_diagnostic_slices(self) -> None:
        from build_dataset import _enforce_benchmark_family_floor

        rows_by_split = {
            "train": [],
            "val": [],
            "test": [],
        }

        result = _enforce_benchmark_family_floor(
            rows_by_split,
            source_domain_family_counts={"telecom": 1, "restaurant": 0, "electronics": 0},
            fallback_rows_by_family={
                "telecom": [
                    {
                        "domain": "telecom",
                        "review_text": "The signal was reliable all day.",
                        "split": "val",
                    }
                ]
            },
            artifact_mode="diagnostic_slice",
            seed=42,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["restored_rows"], 1)
        self.assertIn("telecom", result["restored_families"])

    def test_family_floor_applies_to_research_release(self) -> None:
        from build_dataset import _enforce_benchmark_family_floor

        rows_by_split = {"train": [], "val": [], "test": []}

        result = _enforce_benchmark_family_floor(
            rows_by_split,
            source_domain_family_counts={"telecom": 1, "restaurant": 0, "electronics": 0},
            fallback_rows_by_family={
                "telecom": [
                    {
                        "domain": "telecom",
                        "review_text": "The signal was reliable all day.",
                        "split": "val",
                    }
                ]
            },
            artifact_mode="research_release",
            seed=42,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["restored_rows"], 1)
        self.assertIn("telecom", result["restored_families"])

    def test_semantic_cap_limits_repeated_clusters(self) -> None:
        from build_dataset import _apply_benchmark_semantic_cap

        rows_by_split = {
            "train": [
                {
                    "domain": "restaurant",
                    "review_text": "A",
                    "gold_interpretations": [{"aspect_label": "service", "sentiment": "negative", "evidence_text": "service"}],
                    "implicit_grounded_interpretations": [{"aspect_label": "service"}],
                    "explicit_grounded_interpretations": [],
                    "hardness_tier": "H3",
                    "abstain_acceptable": True,
                    "novel_acceptable": False,
                    "split": "train",
                },
                {
                    "domain": "restaurant",
                    "review_text": "B",
                    "gold_interpretations": [{"aspect_label": "service", "sentiment": "negative", "evidence_text": "service"}],
                    "implicit_grounded_interpretations": [{"aspect_label": "service"}],
                    "explicit_grounded_interpretations": [],
                    "hardness_tier": "H3",
                    "abstain_acceptable": True,
                    "novel_acceptable": False,
                    "split": "train",
                },
                {
                    "domain": "restaurant",
                    "review_text": "C",
                    "gold_interpretations": [{"aspect_label": "service", "sentiment": "negative", "evidence_text": "service"}],
                    "implicit_grounded_interpretations": [{"aspect_label": "service"}],
                    "explicit_grounded_interpretations": [],
                    "hardness_tier": "H3",
                    "abstain_acceptable": True,
                    "novel_acceptable": False,
                    "split": "train",
                },
            ],
            "val": [],
            "test": [],
        }

        result = _apply_benchmark_semantic_cap(rows_by_split, max_per_cluster=2)

        self.assertTrue(result["applied"])
        self.assertEqual(result["removed_rows"], 1)
        self.assertEqual(len(rows_by_split["train"]), 2)


if __name__ == "__main__":
    unittest.main()
