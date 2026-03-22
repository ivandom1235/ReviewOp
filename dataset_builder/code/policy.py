from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class DecisionChoice:
    label: str
    confidence: float
    candidate_set: List[Dict]
    decision_policy: str
    random_seed_used: int | None


def _normalize_probs(candidates: Sequence[Dict]) -> List[float]:
    scores = [max(0.0, float(c.get("probability", c.get("confidence", 0.0)))) for c in candidates]
    total = sum(scores)
    if total <= 0:
        return [1.0 / max(1, len(candidates)) for _ in candidates]
    return [s / total for s in scores]


def _top_k(candidates: Sequence[Dict], k: int) -> List[Dict]:
    return sorted(candidates, key=lambda x: float(x.get("probability", x.get("confidence", 0.0))), reverse=True)[: max(1, k)]


def _sample_weighted(candidates: Sequence[Dict], temperature: float, seed: int | None) -> Tuple[Dict, int | None]:
    if not candidates:
        return {}, seed
    rng = random.Random(seed)
    if temperature <= 0:
        return max(candidates, key=lambda x: float(x.get("probability", x.get("confidence", 0.0)))), seed

    raw = [max(1e-9, float(c.get("probability", c.get("confidence", 0.0)))) for c in candidates]
    scaled = [math.pow(v, 1.0 / max(temperature, 1e-6)) for v in raw]
    total = sum(scaled)
    pick = rng.random() * total
    upto = 0.0
    for cand, score in zip(candidates, scaled):
        upto += score
        if upto >= pick:
            return cand, seed
    return candidates[-1], seed


def choose_label(
    *,
    policy: str,
    deterministic_label: str,
    deterministic_confidence: float,
    candidates: Sequence[Dict],
    temperature: float,
    seed: int | None,
    min_confidence_for_hard_map: float,
    hybrid_top_k: int = 3,
) -> DecisionChoice:
    policy = (policy or "deterministic").strip().lower()
    candidates = list(candidates or [])
    candidate_set = [
        {
            "label": c.get("label", ""),
            "probability": float(c.get("probability", c.get("confidence", 0.0))),
            "source": c.get("source", ""),
        }
        for c in candidates
        if str(c.get("label", "")).strip()
    ]
    if not candidate_set:
        candidate_set = [{"label": deterministic_label, "probability": float(deterministic_confidence), "source": "deterministic"}]

    if policy == "deterministic":
        return DecisionChoice(deterministic_label, float(deterministic_confidence), candidate_set, policy, seed)

    if policy == "hybrid":
        if float(deterministic_confidence) >= float(min_confidence_for_hard_map):
            return DecisionChoice(deterministic_label, float(deterministic_confidence), candidate_set, policy, seed)
        topk = _top_k(candidate_set, hybrid_top_k)
        chosen, used_seed = _sample_weighted(topk, temperature, seed)
        return DecisionChoice(str(chosen.get("label", deterministic_label)), float(chosen.get("probability", deterministic_confidence)), topk, policy, used_seed)

    topk = _top_k(candidate_set, len(candidate_set))
    chosen, used_seed = _sample_weighted(topk, temperature, seed)
    return DecisionChoice(str(chosen.get("label", deterministic_label)), float(chosen.get("probability", deterministic_confidence)), topk, policy, used_seed)


def reliability_bins(pairs: Iterable[Tuple[float, bool]], n_bins: int = 10) -> List[Dict]:
    pairs = list(pairs)
    bins: List[Dict] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bin_pairs = [(conf, correct) for conf, correct in pairs if (conf >= lo and (conf < hi or (i == n_bins - 1 and conf <= hi)))]
        total = len(bin_pairs)
        acc = sum(1 for _, correct in bin_pairs if correct) / total if total else 0.0
        avg_conf = sum(conf for conf, _ in bin_pairs) / total if total else 0.0
        bins.append({"bin": i, "lower": lo, "upper": hi, "count": total, "accuracy": acc, "avg_confidence": avg_conf})
    return bins
