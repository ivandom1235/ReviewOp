import unittest
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

class SplittingV7Tests(unittest.TestCase):
    def test_splitter_assigns_splits_correctly(self) -> None:
        from splitting.splitter import v7_split_pipeline
        from row_contracts import DedupChecked, GroundedInterpretation
        
        # Mock rows
        rows = []
        for i in range(10):
            rows.append(DedupChecked(
                row_id=f"R{i}", review_text="Excellent", domain="elect", group_id=f"G{i}",
                interpretations=[GroundedInterpretation(aspect="general", sentiment="positive", interpretation_type="explicit", confidence=1.0, evidence_span="Excellent")],
                quality_score=1.0, is_v7_gold=True, reason="gold", bucket="benchmark_gold", is_duplicate=False
            ))
        
        # Split 80/10/10
        results = v7_split_pipeline(rows, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42)
        
        self.assertEqual(len(results), 10)
        splits = [r.split for r in results]
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)

if __name__ == "__main__":
    unittest.main()
