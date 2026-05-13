from __future__ import annotations

from typing import Any


PROFILE_DEFAULTS = {
    "smoke": 50,
    "development": 200,
    "stability": 500,
    "journal": 1000,
    "diagnostic_strict": 200,
}


def percentage_target(expected_rows: int, ratio: float, minimum: int = 0) -> int:
    return max(minimum, int(round(expected_rows * ratio)))


def profile_thresholds(profile: str, expected_rows: int) -> dict[str, Any]:
    profile = str(profile or "development").lower()

    return {
        "profile": profile,
        "expected_rows": expected_rows,
        "min_exported_rows": percentage_target(expected_rows, 0.85),
        "min_domain_holdout_val": max(
            25 if profile in {"stability", "journal"} else 10,
            percentage_target(expected_rows, 0.05),
        ),
        "min_counterfactual_validated": max(
            60 if profile == "stability" else 125 if profile == "journal" else 30,
            percentage_target(expected_rows, 0.10 if profile in {"stability", "journal"} else 0.075),
        ),
        "min_anchor_modifier_count": max(
            25 if profile == "stability" else 100 if profile == "journal" else 20,
            percentage_target(expected_rows, 0.05),
        ),
        "full_review_evidence_rate_max": 0.10,
        "abstain_rate_min": 0.10,
        "abstain_rate_max": 0.25,
        "novel_rate_min": 0.05 if profile in {"stability", "journal"} else 0.01,
        "novel_rate_max": 0.15,
        "review_queue_min": 2 if profile == "stability" else 5 if profile == "journal" else 1,
        "min_aspect_swap_count": 8 if profile == "stability" else 25 if profile == "journal" else 3,
        "broad_noun_rate_max": 0.20,
        "unknown_candidate_count_max": 0,
    }
