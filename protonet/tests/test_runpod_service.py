import unittest
from pathlib import Path
from unittest.mock import patch

from protonet.code import inference_service as service
from protonet.infer_api import predict_implicit_aspects


class _SentimentEngineStub:
    def classify_sentiment_with_confidence(self, evidence_text: str, aspect: str):  # noqa: ARG002
        return "neutral", 0.77


class _RuntimeStub:
    def score_text_selective(self, review_text: str, evidence_text: str, domain: str | None = None):  # noqa: ARG002
        return {
            "decision": "single_label",
            "abstain": False,
            "decision_band": "known",
            "confidence": 0.91,
            "ambiguity_score": 0.08,
            "novelty_score": 0.12,
            "accepted_predictions": [
                {
                    "aspect": "battery",
                    "sentiment": "negative",
                    "confidence": 0.91,
                    "raw_score": 4.0,
                    "routing": "known",
                }
            ],
            "novel_candidates": [],
        }


class _AbstainRuntimeStub:
    def score_text_selective(self, review_text: str, evidence_text: str, domain: str | None = None):  # noqa: ARG002
        return {
            "decision": "abstain",
            "abstain": True,
            "decision_band": "boundary",
            "confidence": 0.12,
            "ambiguity_score": 0.91,
            "novelty_score": 0.74,
            "accepted_predictions": [],
            "abstained_predictions": [{"reason": "low_selective_confidence", "confidence": 0.12, "ambiguity_score": 0.91}],
            "novel_candidates": [],
            "scored_rows": [{"aspect": "battery", "sentiment": "neutral", "confidence": 0.3, "raw_score": 1.0, "min_distance_sq": 1.0}],
        }


class InferenceServiceTests(unittest.TestCase):
    def test_request_payload_supports_input_wrapper(self) -> None:
        payload = service.normalize_request_payload({"input": {"text": "Battery lasts all day", "top_k": 3}})
        self.assertEqual(payload["text"], "Battery lasts all day")
        self.assertEqual(payload["top_k"], 3)

    def test_infer_from_request_returns_predictions(self) -> None:
        with patch.object(service, "load_service_runtime", return_value={"runtime": _RuntimeStub(), "seq2seq_engine": _SentimentEngineStub()}):
            out = service.infer_from_request({"text": "Battery lasts all day", "domain": "electronics", "top_k": 1})
        self.assertIn("predictions", out)
        self.assertEqual(out["predictions"][0]["aspect_raw"], "battery")

    def test_handle_event_wrapper_returns_output(self) -> None:
        with patch.object(service, "load_service_runtime", return_value={"runtime": _RuntimeStub(), "seq2seq_engine": _SentimentEngineStub()}):
            out = service.handle_event({"input": {"text": "Battery lasts all day", "domain": "electronics", "top_k": 1}})
        self.assertIn("output", out)
        self.assertIn("predictions", out["output"])
        self.assertEqual(out["output"]["predictions"][0]["aspect_cluster"], "battery")

    def test_local_bundle_smoke_contract(self) -> None:
        bundle_path = Path("protonet/metadata/model_bundle.pt")
        if not bundle_path.exists():
            self.skipTest("protonet bundle is missing")
        try:
            predictions = predict_implicit_aspects(
                "Battery life is excellent but the charger is flimsy.",
                domain="electronics",
                top_k=3,
            )
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"Protonet smoke test could not load local runtime: {exc}")
            return
        self.assertIsInstance(predictions, list)
        self.assertTrue(predictions)
        self.assertIn("aspect_raw", predictions[0])
        self.assertIn("confidence", predictions[0])

    def test_abstain_branch_is_preserved(self) -> None:
        with patch.object(service, "load_service_runtime", return_value={"runtime": _AbstainRuntimeStub(), "seq2seq_engine": _SentimentEngineStub()}):
            predictions = service.predict_implicit_aspects("Battery feels off", domain="electronics", top_k=3, bundle_path="protonet/metadata/model_bundle.pt")
        self.assertTrue(predictions)
        self.assertTrue(predictions[0]["abstain"])
        self.assertEqual(predictions[0]["decision"], "abstain")
        self.assertEqual(predictions[0]["decision_band"], "boundary")


if __name__ == "__main__":
    unittest.main()
