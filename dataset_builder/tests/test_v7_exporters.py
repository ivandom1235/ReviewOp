import unittest
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

class ExportersV7Tests(unittest.TestCase):
    def test_v7_export_pipeline_mapping(self) -> None:
        from benchmark.exporters import v7_export_pipeline
        from row_contracts import SplitAssigned, GroundedInterpretation
        
        # Mock a row in 'val' split
        rows = [
            SplitAssigned(
                row_id="R1", review_text="Excellent", domain="elect", group_id="G1",
                interpretations=[GroundedInterpretation(aspect="general", sentiment="positive", interpretation_type="explicit", confidence=1.0, evidence_span="Excellent")],
                quality_score=1.0, is_v7_gold=True, reason="gold", bucket="benchmark_gold", is_duplicate=False,
                split="val"
            )
        ]
        
        bm, tr = v7_export_pipeline(rows)
        
        self.assertEqual(len(bm), 1)
        self.assertEqual(len(tr), 0) # Val rows shouldn't be in train examples
        self.assertEqual(bm[0].metadata["split"], "val")
        self.assertTrue(any(hasattr(i, 'evidence_span') for i in bm[0].gold_interpretations))

if __name__ == "__main__":
    unittest.main()
