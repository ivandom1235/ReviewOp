from __future__ import annotations

from dataset_builder.canonical.cluster_validator import ClusterValidator


def test_same_aspect_behavioral_repeat_relaxes_behavior_rate_threshold() -> None:
    validator = ClusterValidator()
    cluster_metrics = {
        "support_count": 3,
        "unique_review_count": 3,
        "cluster_consistency": 0.20,
        "evidence_quality_mean": 0.80,
        "contradiction_score": 0.0,
        "aspect_raw": "crust",
        "trigger_patterns": (
            "the crust was soggy",
            "pizza crust stayed soggy",
            "the crust came out undercooked again",
        ),
        "source_types": ("implicit_json",),
        "unique_surface_form_count": 3,
    }
    assert validator.validate_for_review_queue(cluster_metrics) is True
