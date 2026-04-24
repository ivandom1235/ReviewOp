from __future__ import annotations


def select_policy(profile: dict[str, object]) -> str:
    mode = str(profile.get("profile_mode") or "mixed")
    if mode == "implicit_heavy":
        return "evidence_strict"
    if mode == "explicit_heavy":
        return "canonical_balanced"
    return "balanced"


def select_thresholds(profile: dict[str, object]) -> dict[str, float]:
    policy = select_policy(profile)
    if policy == "evidence_strict":
        return {"min_confidence_train": 0.6}
    return {"min_confidence_train": 0.5}
