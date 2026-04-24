from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.review_pipeline import split_selective_states


class SelectiveRoutingTests(unittest.TestCase):
    def test_known_predictions_are_accepted(self) -> None:
        states = split_selective_states([
            {"aspect": "battery", "routing": "known", "confidence": 0.9}
        ])

        self.assertEqual(len(states["accepted_predictions"]), 1)
        self.assertEqual(states["accepted_predictions"][0]["aspect"], "battery")
        self.assertEqual(states["abstained_predictions"], [])
        self.assertEqual(states["novel_candidates"], [])

    def test_novel_predictions_are_not_accepted(self) -> None:
        states = split_selective_states([
            {
                "aspect": "hinge_sparks",
                "routing": "novel",
                "confidence": 0.65,
                "novelty_score": 0.91,
                "novel_candidates": [{"aspect": "hinge_sparks", "novelty_score": 0.91}],
            }
        ])

        self.assertEqual(states["accepted_predictions"], [])
        self.assertEqual(len(states["novel_candidates"]), 1)
        self.assertEqual(states["novel_candidates"][0]["aspect"], "hinge_sparks")

    def test_boundary_predictions_are_abstained(self) -> None:
        states = split_selective_states([
            {"aspect": "screen_comfort", "routing": "boundary", "confidence": 0.42, "ambiguity_score": 0.7}
        ])

        self.assertEqual(states["accepted_predictions"], [])
        self.assertEqual(len(states["abstained_predictions"]), 1)
        self.assertEqual(states["abstained_predictions"][0]["aspect"], "screen_comfort")

    def test_nested_novel_candidates_are_deduped(self) -> None:
        candidate = {"aspect": "hinge_sparks", "novelty_score": 0.91}
        states = split_selective_states([
            {"routing": "novel", "novel_candidates": [candidate, dict(candidate)]}
        ])

        self.assertEqual(len(states["novel_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
