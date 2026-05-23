import pytest

from protonet.config import ECProtoNetV2Config
from protonet.scorer import ScoringContext, score_examples
from protonet.schema import GoldAspect, ReviewExample, to_runtime_example


class _DummyEncoder:
    def encode(self, texts):
        import numpy as np
        return np.zeros((len(texts), 2), dtype=np.float32)


class _DummyProto:
    aspects = []
    matrix = __import__("numpy").zeros((0, 0), dtype=__import__("numpy").float32)
    support_counts = {}
    prototype_source = "test"
    label_descriptions = {}

    def topk(self, q, k):
        import numpy as np
        return np.zeros((len(q), 0), dtype=np.float32), np.zeros((len(q), 0), dtype=np.int32)


def test_score_examples_requires_runtime_examples() -> None:
    ctx = ScoringContext(
        encoder=_DummyEncoder(),
        prototypes=_DummyProto(),
        memory=None,
        config=ECProtoNetV2Config(require_artifact_pass=False, require_active_contract=False),
    )
    ex = ReviewExample(
        row_id="r1",
        review_id="r1",
        text="The prices are wonderfully low.",
        domain="restaurant",
        split="test",
        source_type="implicit",
    )
    with pytest.raises(TypeError):
        score_examples([ex], ctx)  # type: ignore[arg-type]


def test_runtime_conversion_ignores_gold_like_fields_for_inference_invariance() -> None:
    ctx = ScoringContext(
        encoder=_DummyEncoder(),
        prototypes=_DummyProto(),
        memory=None,
        config=ECProtoNetV2Config(require_artifact_pass=False, require_active_contract=False),
    )
    base = ReviewExample(
        row_id="r2",
        review_id="r2",
        text="Network drops near tower handoff.",
        domain="telecom",
        split="test",
        source_type="implicit",
        query_text="battery charge drain",
        matched_terms=["battery", "drain"],
        evidence_text="battery drains too fast",
        gold_aspects=[GoldAspect(aspect="battery_life")],
    )
    poisoned = ReviewExample(
        row_id="r2",
        review_id="r2",
        text="Network drops near tower handoff.",
        domain="telecom",
        split="test",
        source_type="implicit",
        query_text="refund invoice payment",
        matched_terms=["refund", "invoice"],
        evidence_text="refund process unclear",
        gold_aspects=[GoldAspect(aspect="billing")],
    )

    runtime_base = to_runtime_example(base)
    runtime_poisoned = to_runtime_example(poisoned)
    assert runtime_base == runtime_poisoned

    pred_base = score_examples([runtime_base], ctx)["r2"]
    pred_poisoned = score_examples([runtime_poisoned], ctx)["r2"]
    assert pred_base == pred_poisoned
