from __future__ import annotations

"""Single source of truth for novelty scoring.

Both the offline evaluator (evaluator.py) and the live runtime (runtime_infer.py)
must use this function so that novelty scores are identical at calibration time and
inference time. Any change to the formula here applies everywhere.
"""

_DISTANCE_W: float = 0.50
_AMBIGUITY_W: float = 0.30
_ENERGY_W: float = 0.20


def compute_novelty_score(
    distance_score: float,
    ambiguity_score: float,
    energy_score: float,
) -> float:
    """Return a novelty score in [0, 1].

    Args:
        distance_score: Normalised prototype distance (0 = close/known, 1 = far/novel).
        ambiguity_score: Normalised margin between top-1 and top-2 probs (0 = clear, 1 = ambiguous).
        energy_score: Normalised free-energy indicator (0 = confident, 1 = uncertain).

    Returns:
        Weighted combination clamped to [0, 1].
    """
    raw = _DISTANCE_W * distance_score + _AMBIGUITY_W * ambiguity_score + _ENERGY_W * energy_score
    return max(0.0, min(1.0, raw))
