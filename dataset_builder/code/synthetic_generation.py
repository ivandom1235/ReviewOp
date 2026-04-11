from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from utils import normalize_whitespace, tokenize, write_jsonl


DEFAULT_DOMAIN_ASPECTS: dict[str, list[str]] = {
    "restaurant": ["food_quality", "service_speed", "ambience", "portion_size"],
    "electronics": ["battery", "performance", "display_quality", "build_quality"],
    "telecom": ["connectivity", "service_speed", "value"],
    "hotel": ["cleanliness", "service_quality", "comfort", "location"],
    "airline": ["timeliness", "service_quality", "comfort", "value"],
    "grocery": ["freshness", "value", "availability", "checkout_speed"],
    "banking": ["service_speed", "fees", "app_reliability", "support_quality"],
    "insurance": ["claim_speed", "coverage", "value", "support_quality"],
    "education": ["teaching_quality", "content_quality", "support_quality", "value"],
    "healthcare": ["wait_time", "care_quality", "cleanliness", "communication"],
    "ride_hailing": ["wait_time", "driver_behavior", "route_quality", "value"],
    "ecommerce": ["delivery_speed", "product_quality", "return_process", "value"],
    "fitness": ["equipment_quality", "cleanliness", "staff_support", "crowding"],
    "cinema": ["audio_visual_quality", "seating_comfort", "cleanliness", "value"],
    "streaming": ["content_quality", "app_performance", "value", "recommendation_quality"],
    "automotive": ["service_speed", "repair_quality", "value", "communication"],
    "real_estate": ["agent_responsiveness", "property_quality", "value", "documentation"],
    "delivery": ["delivery_speed", "package_condition", "support_quality", "value"],
    "beauty": ["service_quality", "cleanliness", "value", "wait_time"],
    "gaming": ["performance", "content_quality", "community_quality", "value"],
}

POSITIVE_TEMPLATES = [
    "The {aspect} in this {domain} experience was excellent and felt like the best so far.",
    "Great {aspect} overall in this {domain} setting, and I would recommend it.",
    "Fantastic {aspect} with smooth execution in this {domain} interaction.",
]
NEGATIVE_TEMPLATES = [
    "The {aspect} in this {domain} experience was the worst so far and very disappointing.",
    "Poor {aspect} overall in this {domain} context, and it felt frustrating.",
    "The {aspect} was bad and clearly below expectations in this {domain} service.",
]
NEUTRAL_TEMPLATES = [
    "The {aspect} in this {domain} experience was okay and mostly average.",
    "The {aspect} seemed standard in this {domain} context without major surprises.",
    "The {aspect} was neither great nor bad in this {domain} interaction.",
]


@dataclass
class SyntheticGateDecision:
    accepted: bool
    reasons: list[str]


def _template_pool(sentiment: str) -> list[str]:
    if sentiment == "positive":
        return POSITIVE_TEMPLATES
    if sentiment == "negative":
        return NEGATIVE_TEMPLATES
    return NEUTRAL_TEMPLATES


def _realism_gate(text: str) -> bool:
    tokens = tokenize(text)
    return 8 <= len(tokens) <= 48


def _diversity_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _near_duplicate_key(text: str) -> str:
    return " ".join(tokenize(text))


def _sentiment_gate(text: str, target_sentiment: str) -> bool:
    lowered = normalize_whitespace(text).lower()
    if target_sentiment == "positive":
        return any(tok in lowered for tok in ("great", "excellent", "fantastic", "best", "recommend"))
    if target_sentiment == "negative":
        return any(tok in lowered for tok in ("worst", "bad", "poor", "disappointing", "frustrating"))
    return any(tok in lowered for tok in ("okay", "average", "standard", "neither"))


def _evaluate_sample(*, text: str, domain: str, aspect: str, sentiment: str, seen_keys: set[str]) -> SyntheticGateDecision:
    reasons: list[str] = []

    if not _realism_gate(text):
        reasons.append("realism_gate_failed")

    key = _near_duplicate_key(text)
    if key in seen_keys:
        reasons.append("duplicate_gate_failed")

    if aspect not in DEFAULT_DOMAIN_ASPECTS.get(domain, []):
        reasons.append("aspect_domain_compatibility_failed")

    if not _sentiment_gate(text, sentiment):
        reasons.append("sentiment_consistency_failed")

    if _diversity_score(text) < 0.45:
        reasons.append("lexical_diversity_failed")

    return SyntheticGateDecision(accepted=not reasons, reasons=reasons)


def generate_synthetic_multidomain(
    *,
    domains: list[str] | None = None,
    samples_per_domain: int = 100,
    sentiment_mix: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target_domains = domains or list(DEFAULT_DOMAIN_ASPECTS.keys())[:20]
    mix = sentiment_mix or {"positive": 0.35, "negative": 0.35, "neutral": 0.30}

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    gate_counter: Counter[str] = Counter()

    for domain in target_domains:
        aspects = DEFAULT_DOMAIN_ASPECTS.get(domain, ["general_quality"])
        sentiment_plan: list[str] = []
        pos = int(round(samples_per_domain * float(mix.get("positive", 0.35))))
        neg = int(round(samples_per_domain * float(mix.get("negative", 0.35))))
        neu = max(0, samples_per_domain - pos - neg)
        sentiment_plan.extend(["positive"] * pos)
        sentiment_plan.extend(["negative"] * neg)
        sentiment_plan.extend(["neutral"] * neu)

        for index, sentiment in enumerate(sentiment_plan):
            aspect = aspects[index % len(aspects)]
            templates = _template_pool(sentiment)
            template = templates[index % len(templates)]
            text = template.format(aspect=aspect.replace("_", " "), domain=domain.replace("_", " "))
            decision = _evaluate_sample(text=text, domain=domain, aspect=aspect, sentiment=sentiment, seen_keys=seen_keys)
            row = {
                "id": f"syn_{domain}_{index}",
                "domain": domain,
                "text": text,
                "target_aspect": aspect,
                "target_sentiment": sentiment,
            }
            if decision.accepted:
                accepted.append(row)
                seen_keys.add(_near_duplicate_key(text))
            else:
                row["rejection_reasons"] = decision.reasons
                rejected.append(row)
                for reason in decision.reasons:
                    gate_counter[reason] += 1

    audit = {
        "target_domains": target_domains,
        "samples_per_domain": samples_per_domain,
        "target_total": len(target_domains) * samples_per_domain,
        "accepted_total": len(accepted),
        "rejected_total": len(rejected),
        "acceptance_rate": round(len(accepted) / max(1, (len(accepted) + len(rejected))), 4),
        "rejection_reason_counts": dict(gate_counter),
        "sentiment_mix_target": mix,
        "sentiment_mix_accepted": dict(Counter(row["target_sentiment"] for row in accepted)),
        "domain_mix_accepted": dict(Counter(row["domain"] for row in accepted)),
    }
    return accepted, rejected, audit


def write_synthetic_outputs(*, output_dir: Any, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    from pathlib import Path

    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    write_jsonl(base / "accepted.jsonl", accepted)
    write_jsonl(base / "rejected.jsonl", rejected)
