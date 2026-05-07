from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SelectiveDecision:
    decision: str
    decision_band: str
    abstain_reason: str | None
    route_novel: bool
    route_boundary: bool


def decide_prediction_state(
    *,
    novelty_score: float,
    selective_confidence: float,
    evidence_support: float = 0.0,
    verifier_support: float = 0.0,
    memory_support: float = 0.0,
    ambiguity_penalty: float = 0.0,
    novelty_risk: float = 0.0,
    contradiction_score: float = 0.0,
    abstain_threshold: float,
    known_threshold: float,
    novel_threshold: float,
) -> dict[str, Any]:
    combined_score = combine_routing_score(
        prototype_similarity=float(selective_confidence),
        evidence_support=float(evidence_support),
        verifier_support=float(verifier_support),
        memory_support=float(memory_support),
        ambiguity_penalty=float(ambiguity_penalty),
        novelty_risk=float(novelty_risk),
        contradiction_score=float(contradiction_score),
    )
    if novelty_score >= novel_threshold:
        decision = "novel"
        band = "novel"
        reason = None
    elif novelty_score > known_threshold:
        decision = "abstain"
        band = "boundary"
        reason = "boundary_uncertain_novelty"
    elif combined_score < abstain_threshold:
        decision = "abstain"
        band = "known"
        reason = "low_selective_confidence"
    else:
        decision = "single_label"
        band = "known"
        reason = None
    return {
        "decision": decision,
        "decision_band": band,
        "abstain_reason": reason,
        "route_novel": decision == "novel",
        "route_boundary": decision == "abstain" and band == "boundary",
        "selective_score": float(combined_score),
        "novelty_score": float(novelty_score),
        "evidence_support": float(evidence_support),
        "ambiguity_penalty": float(ambiguity_penalty),
        "verifier_support": float(verifier_support),
        "memory_support": float(memory_support),
        "contradiction_score": float(contradiction_score),
    }


def combine_routing_score(
    *,
    prototype_similarity: float,
    evidence_support: float = 0.0,
    verifier_support: float = 0.0,
    ambiguity_penalty: float = 0.0,
    memory_support: float = 0.0,
    novelty_risk: float = 0.0,
    contradiction_score: float = 0.0,
) -> float:
    score = (
        0.35 * float(prototype_similarity)
        + 0.20 * float(evidence_support)
        + 0.15 * float(verifier_support)
        + 0.10 * float(memory_support)
        - 0.10 * float(ambiguity_penalty)
        - 0.10 * float(novelty_risk)
        - 0.10 * float(contradiction_score)
    )
    return max(0.0, min(1.0, score))


def calibrate_novelty_thresholds(
    *,
    novelty_calibration: dict[str, Any] | None,
    default_known: float,
    default_novel: float,
    validation_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = dict(novelty_calibration or {})
    thresholds = dict(payload.get("thresholds") or {})
    t_known = float(thresholds.get("T_known", default_known))
    t_novel = float(thresholds.get("T_novel", default_novel))
    if t_known > t_novel:
        t_known, t_novel = t_novel, t_known
    has_known = True
    has_novel = True
    if validation_rows is not None:
        truth = [1 if bool(row.get("novel_acceptable", False)) else 0 for row in validation_rows]
        has_known = any(value == 0 for value in truth)
        has_novel = any(value == 1 for value in truth)
    applicable = bool(payload) and has_known and has_novel and not bool(payload.get("not_applicable", False))
    if not applicable:
        return {
            "T_known": float(default_known),
            "T_novel": float(default_novel),
            "applicable": False,
            "reason": "insufficient_validation_support",
            "source": payload.get("scorer", "distance_energy"),
        }
    return {
        "T_known": float(max(0.0, min(1.0, t_known))),
        "T_novel": float(max(0.0, min(1.0, t_novel))),
        "applicable": True,
        "reason": None,
        "source": payload.get("scorer", "distance_energy"),
    }


def decide_selective_routing(
    *,
    novelty_score: float,
    selective_confidence: float,
    abstain_threshold: float,
    known_threshold: float,
    novel_threshold: float,
) -> SelectiveDecision:
    state = decide_prediction_state(
        novelty_score=novelty_score,
        selective_confidence=selective_confidence,
        abstain_threshold=abstain_threshold,
        known_threshold=known_threshold,
        novel_threshold=novel_threshold,
    )
    return SelectiveDecision(
        state["decision"],
        state["decision_band"],
        state["abstain_reason"],
        bool(state["route_novel"]),
        bool(state["route_boundary"]),
    )

