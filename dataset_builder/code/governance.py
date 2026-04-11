from __future__ import annotations

from typing import Any


def governance_signoff(*, report: dict[str, Any]) -> dict[str, Any]:
    validation = report.get("validation", {})
    benchmark_gold = report.get("benchmark_gold_eval", {})
    sentiment = report.get("sentiment_quality", {})
    robustness = report.get("robust_training_eval", {})

    checks = {
        "artifact_parity": bool(validation.get("benchmark_artifact_counts_match", False)),
        "grounded_evidence": float(benchmark_gold.get("grounded_evidence_rate", 0.0)) >= 0.98,
        "duplicate_interpretation_rate": float(benchmark_gold.get("duplicate_interpretation_rate", 1.0)) <= 0.01,
        "implicit_purity": float(benchmark_gold.get("implicit_purity_rate", 0.0)) >= 0.7,
        "ontology_compatibility": float(benchmark_gold.get("ontology_compatibility_rate", 0.0)) >= 0.9,
        "sentiment_mismatch_rate": float(sentiment.get("sentiment_mismatch_rate", 1.0)) <= 0.2,
        "worst_domain_floor": float((robustness.get("groupdro") or {}).get("worst_domain_f1", 0.0)) >= 0.4,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "blocking_failures": [name for name, ok in checks.items() if not ok],
    }
