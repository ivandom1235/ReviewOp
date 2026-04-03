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

    def test_build_implicit_row_prefers_exact_support_and_falls_back_cleanly(self) -> None:
        supported = build_implicit_row(
            {"id": "1", "split": "train", "review_text": "The battery life is great", "aspect": "battery life"},
            text_column="review_text",
            candidate_aspects=["battery life"],
            confidence_threshold=0.6,
            row_index=0,
        )
        implicit = supported["implicit"]
        self.assertEqual(implicit["aspects"], ["power"])
        self.assertEqual(implicit["mode"], "zeroshot")
        self.assertFalse(implicit["needs_review"])
        self.assertEqual(implicit["spans"][0]["support_type"], "exact")
        self.assertEqual(implicit["spans"][0]["matched_surface"].lower(), "battery life")
        self.assertEqual(implicit["spans"][0]["latent_aspect"], "power")
        self.assertEqual(implicit["spans"][0]["surface_aspect"], "battery life")

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

    def test_build_implicit_row_maps_surface_to_latent_facet(self) -> None:
        row = build_implicit_row(
            {"id": "3", "split": "train", "review_text": "Disappointing for such a lovely screen and at a reasonable price", "aspect": "screen"},
            text_column="review_text",
            candidate_aspects=["screen", "price"],
            confidence_threshold=0.6,
            row_index=0,
        )
        implicit = row["implicit"]
        self.assertIn("display quality", implicit["aspects"])
        self.assertIn("value", implicit["aspects"])
        self.assertTrue(all(span["latent_aspect"] in {"display quality", "value"} for span in implicit["spans"]))

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
        self.assertIn("timeliness", taxi_row["implicit"]["aspects"])

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
        self.assertGreaterEqual(int(implicit.get("domain_filtered_matches", 0)), 1)

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
        self.assertIn("food quality", adaptive_row["implicit"]["aspects"])
        self.assertGreaterEqual(int(adaptive_row["implicit"].get("domain_prior_penalty_count", 0)), 1)

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
            {"id": "6", "split": "train", "review_text": "The battery life is excellent", "aspect": "battery life"},
            text_column="review_text",
            candidate_aspects=[],
            confidence_threshold=0.6,
            row_index=0,
            implicit_mode="hybrid",
        )
        implicit = row["implicit"]
        self.assertIn("power", implicit["aspects"])
        self.assertFalse(implicit["needs_review"])
        self.assertIsNone(implicit["review_reason"])


if __name__ == "__main__":
    unittest.main()
