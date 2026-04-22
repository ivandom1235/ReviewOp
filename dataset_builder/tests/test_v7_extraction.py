import unittest
import sys
from pathlib import Path
import asyncio

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Mocking modules that might fail due to missing utils/parents
# In a real scenario, we'd fix the imports first, but for TDD we want to see it fail.

class ExtractionV7Tests(unittest.TestCase):
    def test_implicit_pipeline_consumes_prepared_and_returns_grounded(self) -> None:
        from extraction.implicit_pipeline import build_implicit_row
        from row_contracts import Prepared
        
        # Define a prepared row
        prepared = Prepared(
            row_id="R-EXT-1",
            review_text="The battery dies in 2 hours but the screen is gorgeous.",
            domain="electronics",
            group_id="G-EXT-1"
        )
        
        # This is async, so we'd normally run it with an event loop.
        # For now, we are testing that it can be CALLED with these types.
        
        # We expect build_implicit_row to be updated to accept Prepared object
        # instead of a dict.
        
        # Since I haven't updated the code yet, I expect this to fail OR 
        # fail inside if it tries to do row.get() on a Pydantic object.
        
        loop = asyncio.get_event_loop()
        try:
            # We use dummy candidate aspects
            result = loop.run_until_complete(build_implicit_row(
                prepared, 
                candidate_aspects=["power", "display quality"],
                confidence_threshold=0.5,
                row_index=0
            ))
            
            self.assertIsInstance(result, dict)
            self.assertIn("implicit", result)
            self.assertIn("spans", result["implicit"])
        except Exception as e:
            print(f"Extraction failed as expected: {e}")
            raise e

    def test_structured_canonical_mapping_exposes_source_and_confidence(self) -> None:
        from extraction.aspect_registry import resolve_domain_canonical_mapping

        mapping = resolve_domain_canonical_mapping(
            domain="restaurant",
            latent_aspect="waiter",
            surface_rationale_tag="slow service from waiter",
        )

        self.assertIsNotNone(mapping)
        assert mapping is not None
        self.assertEqual(mapping["canonical_label"], "service_speed")
        self.assertEqual(mapping["mapping_source"], "domain_map")
        self.assertGreaterEqual(float(mapping["mapping_confidence"]), 0.9)

    def test_structured_canonical_mapping_rejects_generic_label(self) -> None:
        from extraction.aspect_registry import resolve_domain_canonical_mapping

        mapping = resolve_domain_canonical_mapping(
            domain="restaurant",
            latent_aspect="quality",
            surface_rationale_tag="quality",
        )
        self.assertIsNone(mapping)

    def test_implicit_pipeline_marks_sentence_aligned_span_quality(self) -> None:
        from extraction.implicit_pipeline import build_implicit_row
        from row_contracts import Prepared

        prepared = Prepared(
            row_id="R-EXT-2",
            review_text="The laptop gets hot quickly.",
            domain="electronics",
            group_id="G-EXT-2",
        )
        result = asyncio.get_event_loop().run_until_complete(
            build_implicit_row(
                prepared,
                candidate_aspects=["thermal"],
                confidence_threshold=0.5,
                row_index=0,
            )
        )
        spans = list((result or {}).get("implicit", {}).get("spans") or [])
        self.assertGreaterEqual(len(spans), 1)
        self.assertTrue(all(str(span.get("span_quality") or "") in {"exact_sentence_match", "light_repair"} for span in spans))

if __name__ == "__main__":
    unittest.main()
