import unittest
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

class CurationV7Tests(unittest.TestCase):
    def test_full_curation_pipe(self) -> None:
        from curation.scoring import apply_quality_scoring
        from curation.bucketing import assign_bucket
        from curation.deduplication import apply_semantic_dedup
        from row_contracts import Grounded, GroundedInterpretation
        
        # 1. Start with Grounded rows
        g_rows = [
            Grounded(
                row_id="R1", review_text="Excellent battery", domain="elect", group_id="G1",
                interpretations=[GroundedInterpretation(aspect="power", sentiment="positive", interpretation_type="explicit", confidence=1.0, evidence_text="battery", evidence_span=[10, 17])]
            ),
            Grounded(
                row_id="R2", review_text="Excellent battery", domain="elect", group_id="G2",
                interpretations=[GroundedInterpretation(aspect="power", sentiment="positive", interpretation_type="explicit", confidence=1.0, evidence_text="battery", evidence_span=[10, 17])]
            )
        ]
        
        # 2. Score
        s_rows = [apply_quality_scoring(r) for r in g_rows]
        self.assertEqual(s_rows[0].quality_score, 1.0)
        
        # 3. Bucket
        b_rows = [assign_bucket(r) for r in s_rows]
        self.assertEqual(b_rows[0].bucket, "benchmark_gold")
        
        # 4. Dedup
        d_rows = apply_semantic_dedup(b_rows)
        
        self.assertEqual(len(d_rows), 2)
        # Verify deduplication worked
        self.assertTrue(d_rows[1].is_duplicate)
        self.assertEqual(d_rows[1].duplicate_of, "R1")

    def test_sanitize_gold_spans_marks_light_repair(self) -> None:
        from build_dataset import _sanitize_gold_interpretation_spans

        cleaned, repaired, hard_failures = _sanitize_gold_interpretation_spans(
            "Food was delicious and service was slow.",
            [
                {
                    "aspect_label": "food_quality",
                    "evidence_text": "service was slow",
                    "evidence_span": [-1, -1],
                }
            ],
        )

        self.assertEqual(repaired, 1)
        self.assertEqual(hard_failures, 0)
        self.assertEqual(cleaned[0]["span_quality"], "light_repair")

    def test_sanitize_gold_spans_drops_hard_failures(self) -> None:
        from build_dataset import _sanitize_gold_interpretation_spans

        cleaned, repaired, hard_failures = _sanitize_gold_interpretation_spans(
            "Food was delicious and service was slow.",
            [
                {
                    "aspect_label": "food_quality",
                    "evidence_text": "nonexistent phrase",
                    "evidence_span": [-1, -1],
                }
            ],
        )

        self.assertEqual(repaired, 0)
        self.assertEqual(hard_failures, 1)
        self.assertEqual(cleaned, [])

if __name__ == "__main__":
    unittest.main()
