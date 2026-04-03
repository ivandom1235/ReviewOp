from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from run_experiment import _bounded_v4_candidates, _meets_quality_gates, _rank_key, build_parser


class RunExperimentTests(unittest.TestCase):
    def test_bounded_v4_candidates_count_without_coref(self) -> None:
        candidates = _bounded_v4_candidates(include_coref=False, implicit_min_tokens_values=[6, 8], min_text_tokens_values=[3, 4])
        self.assertEqual(len(candidates), 48)
        self.assertTrue(all(candidate["use_coref"] is False for candidate in candidates))

    def test_bounded_v4_candidates_count_with_coref(self) -> None:
        candidates = _bounded_v4_candidates(include_coref=True, implicit_min_tokens_values=[6, 8], min_text_tokens_values=[3, 4])
        self.assertEqual(len(candidates), 96)
        self.assertIn(True, {candidate["use_coref"] for candidate in candidates})
        self.assertIn(False, {candidate["use_coref"] for candidate in candidates})

    def test_parser_supports_llm_fallback_inverse_flag(self) -> None:
        parser = build_parser()
        defaults = parser.parse_args([])
        self.assertTrue(defaults.enable_llm_fallback)
        self.assertEqual(defaults.gold_min_rows_for_promotion, 600)
        self.assertEqual(defaults.domain_conditioning_mode, "adaptive_soft")
        self.assertIsNone(defaults.train_domain_conditioning_mode)
        self.assertIsNone(defaults.eval_domain_conditioning_mode)
        disabled = parser.parse_args(["--no-enable-llm-fallback"])
        self.assertFalse(disabled.enable_llm_fallback)

    def test_quality_gate_and_ranking(self) -> None:
        passing = {
            "metrics": {
                "fallback_only_rate": 0.2,
                "needs_review_rows": 1700,
                "generic_implicit_aspects": 0,
                "rejected_implicit_aspects": 0,
                "domain_leakage_row_rate": 0.01,
                "grounded_prediction_rate": 0.9,
                "train_general_dominance_rate": 0.15,
                "train_domain_leakage_row_rate": 0.0,
                "train_negative_ratio": 0.16,
                "train_positive_ratio": 0.14,
                "train_neutral_ratio": 0.52,
                "train_target_size_compliant": True,
                "unseen_non_general_coverage": 0.65,
                "unseen_implicit_not_ready_rate": 0.2,
                "unseen_domain_leakage_row_rate": 0.01,
                "has_gold_eval": True,
                "gold_rows": 650,
                "gold_aspect_f1": 0.7,
                "gold_sentiment_f1": 0.7,
                "gold_span_overlap_f1": 0.5,
            }
        }
        failing = {
            "metrics": {
                "fallback_only_rate": 0.3,
                "needs_review_rows": 1900,
                "generic_implicit_aspects": 1,
                "rejected_implicit_aspects": 0,
                "domain_leakage_row_rate": 0.2,
                "grounded_prediction_rate": 0.4,
                "train_general_dominance_rate": 0.5,
                "train_domain_leakage_row_rate": 0.2,
                "train_negative_ratio": 0.03,
                "train_positive_ratio": 0.04,
                "train_neutral_ratio": 0.8,
                "train_target_size_compliant": False,
                "unseen_non_general_coverage": 0.1,
                "unseen_implicit_not_ready_rate": 0.8,
                "unseen_domain_leakage_row_rate": 0.2,
                "has_gold_eval": False,
                "gold_rows": 0,
                "gold_aspect_f1": 0.0,
                "gold_sentiment_f1": 0.0,
                "gold_span_overlap_f1": 0.0,
            }
        }
        self.assertTrue(_meets_quality_gates(passing["metrics"]))
        self.assertFalse(_meets_quality_gates(failing["metrics"]))

        ranked_best = {
            "candidate_id": "cand_001",
            "meets_quality_gates": True,
            "metrics": passing["metrics"],
        }
        ranked_worse = {
            "candidate_id": "cand_002",
            "meets_quality_gates": False,
            "metrics": failing["metrics"],
        }
        self.assertLess(_rank_key(ranked_best), _rank_key(ranked_worse))


if __name__ == "__main__":
    unittest.main()
