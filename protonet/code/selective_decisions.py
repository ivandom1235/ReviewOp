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
    if novelty_score >= novel_threshold:
        return SelectiveDecision("novel", "novel", None, True, False)
    if novelty_score > known_threshold:
        return SelectiveDecision("abstain", "boundary", "boundary_uncertain_novelty", False, True)
    if selective_confidence < abstain_threshold:
        return SelectiveDecision("abstain", "known", "low_selective_confidence", False, False)
    return SelectiveDecision("single_label", "known", None, False, False)

