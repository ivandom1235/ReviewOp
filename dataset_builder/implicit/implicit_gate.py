from __future__ import annotations

from dataclasses import dataclass

from ..schemas.interpretation import Interpretation


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason_codes: list[str]
    hard_failure: bool = False


def evaluate_implicit_candidate(interp: Interpretation) -> GateDecision:
    reasons: list[str] = []
    if interp.canonical_confidence < 0.3:
        reasons.append("low_confidence")
    if interp.support_type not in {"exact", "near_exact", "gold"}:
        reasons.append("weak_support")
    if not interp.evidence_text.strip():
        reasons.append("missing_evidence")
    hard = "missing_evidence" in reasons
    return GateDecision(not reasons, reasons, hard)


def implicit_export_gate(interp: Interpretation) -> bool:
    return evaluate_implicit_candidate(interp).accepted


def implicit_failure_breakdown(items: list[Interpretation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for reason in evaluate_implicit_candidate(item).reason_codes:
            counts[reason] = counts.get(reason, 0) + 1
    return counts
