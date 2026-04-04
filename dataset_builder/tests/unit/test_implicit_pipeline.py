from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from implicit_pipeline import build_implicit_row, collect_diagnostics, discover_aspects


class ImplicitPipelineTests(unittest.TestCase):
    def test_discover_aspects_rejects_generic_tokens(self) -> None:
        rows = [
            {"review_text": "The battery life is great", "aspect": "battery life"},
            {"review_text": "The screen is bright", "aspect": "screen"},
            {"review_text": "The service was nice", "aspect": "the"},
        ]
        aspects = discover_aspects(rows, text_column="review_text", max_aspects=10, implicit_mode="heuristic")
        self.assertIn("power", aspects)
        self.assertIn("display quality", aspects)
        self.assertNotIn("the", aspects)

    def test_build_implicit_row_rejects_explicit_leakage_in_strict_mode(self) -> None:
        supported = build_implicit_row(
            {"id": "1", "split": "train", "review_text": "The battery life is great", "aspect": "battery life"},
            text_column="review_text",
            candidate_aspects=["battery life"],
            confidence_threshold=0.6,
            row_index=0,
        )
        implicit = supported["implicit"]
        self.assertEqual(implicit["aspects"], ["general"])
        self.assertEqual(implicit["mode"], "zeroshot")
        self.assertTrue(implicit["needs_review"])
        self.assertEqual(implicit["review_reason"], "strict_leakage")
        self.assertEqual(implicit["implicit_quality_tier"], "rejected")
        self.assertEqual(implicit["strict_rejected_match_count"], 1)
        self.assertIn("explicit_span_in_implicit", implicit["leakage_flags"])
        self.assertEqual(len(implicit["spans"]), 0)

        fallback = build_implicit_row(
            {"id": "2", "split": "train", "review_text": "Nothing useful here", "aspect": "battery life"},
            text_column="review_text",
            candidate_aspects=["battery life"],
            confidence_threshold=0.6,
            row_index=1,
        )
        fallback_implicit = fallback["implicit"]
        self.assertEqual(fallback_implicit["aspects"], ["general"])
        self.assertTrue(fallback_implicit["needs_review"])
        self.assertEqual(fallback_implicit["review_reason"], "fallback_general")
        self.assertFalse(fallback_implicit["llm_fallback_used"])

    def test_build_implicit_row_maps_surface_to_latent_facet_with_non_explicit_cues(self) -> None:
        row = build_implicit_row(
            {"id": "3", "split": "train", "review_text": "The laptop gets hot and drains in two hours."},
            text_column="review_text",
            candidate_aspects=["thermal", "power"],
            confidence_threshold=0.6,
            row_index=0,
        )
        implicit = row["implicit"]
        self.assertIn("thermal", implicit["aspects"])
        self.assertIn("power", implicit["aspects"])
        self.assertGreaterEqual(len(implicit["aspects"]), 2)
        self.assertIn("H1", {span["hardness_tier"] for span in implicit["spans"]})
        self.assertTrue(all(span["label_type"] == "implicit" for span in implicit["spans"]))

    def test_token_boundary_prevents_service_to_ice_false_positive(self) -> None:
        row = build_implicit_row(
            {"id": "svc1", "split": "train", "review_text": "The service was kind and responsive."},
            text_column="review_text",
            candidate_aspects=["service quality"],
            confidence_threshold=0.6,
            row_index=0,
        )
        implicit = row["implicit"]
        self.assertNotIn("thermal", implicit["aspects"])
        self.assertFalse(any(str(span.get("aspect", "")).lower() == "ice" for span in implicit["spans"]))

    def test_build_implicit_row_avoids_cross_domain_keyword_leakage(self) -> None:
        gym_row = build_implicit_row(
            {"id": "4", "split": "train", "review_text": "The treadmill was broken and the room felt crowded."},
            text_column="review_text",
            candidate_aspects=[],
            confidence_threshold=0.6,
            row_index=0,
            domain="gym",
        )
        taxi_row = build_implicit_row(
            {"id": "5", "split": "train", "review_text": "The driver was polite and arrived on time."},
            text_column="review_text",
            candidate_aspects=[],
            confidence_threshold=0.6,
            row_index=1,
            domain="taxi",
        )
        self.assertNotIn("dining experience", gym_row["implicit"]["aspects"])
        self.assertNotIn("compatibility", taxi_row["implicit"]["aspects"])
        self.assertEqual(taxi_row["implicit"]["aspects"], ["general"])
        self.assertEqual(taxi_row["implicit"]["review_reason"], "strict_leakage")

    def test_strict_domain_conditioning_filters_out_of_domain_latent_matches(self) -> None:
        row = build_implicit_row(
            {"id": "7", "split": "train", "review_text": "The food was great but checkout was slow."},
            text_column="review_text",
            candidate_aspects=[],
            confidence_threshold=0.6,
            row_index=0,
            domain="laptop",
            candidate_aspects_by_domain={"laptop": ["battery", "screen", "performance"]},
            strict_domain_conditioning=True,
        )
        implicit = row["implicit"]
        self.assertNotIn("food quality", implicit["aspects"])
        self.assertIn("performance", implicit["aspects"])

    def test_adaptive_soft_conditioning_keeps_out_of_domain_when_evidence_is_strong(self) -> None:
        strict_row = build_implicit_row(
            {"id": "8", "split": "train", "review_text": "The food was great."},
            text_column="review_text",
            candidate_aspects=[],
            confidence_threshold=0.6,
            row_index=0,
            domain="laptop",
            candidate_aspects_by_domain={"laptop": ["battery", "screen", "performance"]},
            domain_conditioning_mode="strict_hard",
        )
        adaptive_row = build_implicit_row(
            {"id": "9", "split": "train", "review_text": "The food was great."},
            text_column="review_text",
            candidate_aspects=[],
            confidence_threshold=0.6,
            row_index=0,
            domain="laptop",
            candidate_aspects_by_domain={"laptop": ["battery", "screen", "performance"]},
            domain_conditioning_mode="adaptive_soft",
            domain_prior_penalty=0.08,
            domain_support_rows=20,
            weak_domain_support_row_threshold=80,
        )
        self.assertIn("general", strict_row["implicit"]["aspects"])
        self.assertEqual(adaptive_row["implicit"]["aspects"], ["general"])
        self.assertEqual(adaptive_row["implicit"]["review_reason"], "strict_leakage")

    def test_collect_diagnostics_counts_support_and_fallback(self) -> None:
        rows = [
            {
                "review_text": "The battery life is great",
                "implicit": {
                    "aspects": ["power"],
                    "spans": [{"support_type": "exact"}],
                    "needs_review": False,
                },
            },
            {
                "review_text": "Nothing useful here",
                "implicit": {
                    "aspects": ["general"],
                    "spans": [],
                    "needs_review": True,
                },
            },
        ]
        diagnostics = collect_diagnostics(rows, text_column="review_text", candidate_aspects=["battery life"])
        self.assertEqual(diagnostics["fallback_only_count"], 1)
        self.assertEqual(diagnostics["needs_review_count"], 1)
        self.assertEqual(diagnostics["span_support"]["exact"], 1)
        self.assertIn("review_reason_counts", diagnostics)
        self.assertIn("fallback_branch_counts", diagnostics)

    def test_hybrid_gold_support_does_not_force_review(self) -> None:
        row = build_implicit_row(
            {"id": "6", "split": "train", "review_text": "It gets hot while gaming.", "aspect": "thermal"},
            text_column="review_text",
            candidate_aspects=[],
            confidence_threshold=0.6,
            row_index=0,
            implicit_mode="hybrid",
        )
        implicit = row["implicit"]
        self.assertIn("thermal", implicit["aspects"])
        self.assertFalse(implicit["needs_review"])
        self.assertIsNone(implicit["review_reason"])


if __name__ == "__main__":
    unittest.main()
