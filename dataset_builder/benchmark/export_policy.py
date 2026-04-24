from __future__ import annotations

from dataclasses import dataclass

from ..implicit.implicit_gate import evaluate_implicit_candidate
from ..schemas.interpretation import Interpretation


@dataclass(frozen=True)
class ExportDecision:
    destination: str
    reason_codes: list[str]


def classify_for_export(items: list[Interpretation], *, split: str) -> ExportDecision:
    if not items:
        return ExportDecision("hard_reject", ["no_interpretations"])
    reasons: list[str] = []
    for item in items:
        reasons.extend(evaluate_implicit_candidate(item).reason_codes)
    unique_reasons = sorted(set(reasons))
    if split == "train" and unique_reasons:
        return ExportDecision("review_queue", unique_reasons)
    if "missing_evidence" in unique_reasons:
        return ExportDecision("hard_reject", unique_reasons)
    return ExportDecision("benchmark_gold", unique_reasons)


def benchmark_acceptance_gate(items: list[Interpretation]) -> bool:
    return classify_for_export(items, split="eval").destination == "benchmark_gold"
