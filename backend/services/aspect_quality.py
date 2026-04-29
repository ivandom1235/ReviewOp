from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.analytics_common import normalize_text


@dataclass(frozen=True)
class AspectGateDecision:
    accepted: bool
    reason: str | None = None
    quality_score: float = 0.0
    mapping_scope: str = "open_world"


_DOMAIN_ALIASES = {
    "electronics": "laptop",
    "electronic": "laptop",
    "computer": "laptop",
    "computers": "laptop",
    "notebook": "laptop",
    "restaurant": "restaurant",
    "restaurants": "restaurant",
    "dining": "restaurant",
}

_LOW_QUALITY_ASPECTS = {
    "clock",
    "corner",
    "desk",
    "everyone",
    "friend",
    "friends",
    "half",
    "middle",
    "one",
    "one corner",
    "people",
    "person",
    "plenty",
    "project",
    "projects",
    "room",
    "someone",
    "thing",
    "things",
    "time",
    "week",
}

_GENERIC_ASPECTS = {
    "availability",
    "billing",
    "brand",
    "customer support",
    "customer service",
    "delivery",
    "durability",
    "packaging",
    "price",
    "pricing",
    "quality",
    "refund",
    "reliability",
    "shipping",
    "support",
    "value",
    "warranty",
}

_DOMAIN_ASPECTS = {
    "laptop": {
        "audio",
        "battery",
        "battery life",
        "brightness",
        "camera",
        "charging",
        "charger",
        "chassis",
        "cooling",
        "display",
        "fan",
        "graphics",
        "hinge",
        "keyboard",
        "microphone",
        "performance",
        "ports",
        "portability",
        "power",
        "processor",
        "ram",
        "screen",
        "software",
        "speakers",
        "storage",
        "thermals",
        "touchpad",
        "trackpad",
        "usability",
        "webcam",
        "wifi",
        "weight",
    },
    "restaurant": {
        "ambience",
        "atmosphere",
        "bar",
        "cleanliness",
        "dish",
        "drink",
        "food",
        "food quality",
        "host",
        "hygiene",
        "meal",
        "menu",
        "music",
        "patio",
        "portion",
        "reservation",
        "restroom",
        "server",
        "service",
        "service quality",
        "staff",
        "taste",
        "table",
        "waiter",
        "waitress",
        "vibe",
    },
}


def normalized_domain(domain: str | None) -> str | None:
    cleaned = normalize_text(domain or "")
    return _DOMAIN_ALIASES.get(cleaned, cleaned or None)


def normalized_aspect_label(value: str | None) -> str:
    return normalize_text(value or "")


def _tokens(label: str) -> set[str]:
    return set(label.replace("_", " ").split())


def _matches_any(label: str, candidates: set[str]) -> bool:
    label_tokens = _tokens(label)
    for candidate in candidates:
        if label == candidate:
            return True
        candidate_tokens = _tokens(candidate)
        if candidate_tokens and candidate_tokens.issubset(label_tokens):
            return True
    return False


def _domain_match(label: str, domain: str | None) -> bool:
    domain_key = normalized_domain(domain)
    if not domain_key:
        return False
    return _matches_any(label, _DOMAIN_ASPECTS.get(domain_key, set()))


def _generic_match(label: str) -> bool:
    return _matches_any(label, _GENERIC_ASPECTS)


def _other_domain_match(label: str, domain: str | None) -> bool:
    domain_key = normalized_domain(domain)
    if not domain_key:
        return False
    for candidate_domain, candidates in _DOMAIN_ASPECTS.items():
        if candidate_domain == domain_key:
            continue
        if _matches_any(label, candidates):
            return True
    return False


def evaluate_explicit_aspect(aspect: str, domain: str | None = None) -> AspectGateDecision:
    label = normalized_aspect_label(aspect)
    if label in _LOW_QUALITY_ASPECTS:
        return AspectGateDecision(False, "low_quality_aspect", 0.0)
    if _domain_match(label, domain):
        return AspectGateDecision(True, quality_score=0.92, mapping_scope="domain")
    if _generic_match(label):
        return AspectGateDecision(True, quality_score=0.86, mapping_scope="generic")
    if _other_domain_match(label, domain):
        return AspectGateDecision(False, "domain_mismatch", 0.0)
    return AspectGateDecision(True, quality_score=0.72, mapping_scope="open_world")


def implicit_aspect_allowed(aspect: str, domain: str | None = None) -> bool:
    label = normalized_aspect_label(aspect)
    domain_key = normalized_domain(domain)
    if not label or label in _LOW_QUALITY_ASPECTS:
        return False
    if not domain_key:
        return True
    return _domain_match(label, domain_key) or _generic_match(label)


def aspect_from_prediction(row: dict[str, Any]) -> str:
    return str(row.get("aspect_cluster") or row.get("aspect_raw") or row.get("aspect") or "").strip()


def apply_domain_gate_to_implicit_predictions(predictions: list[dict[str, Any]], domain: str | None) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in predictions or []:
        if not isinstance(row, dict):
            continue
        routing = str(row.get("routing") or "known").lower()
        decision = str(row.get("decision") or "").lower()
        if routing == "known" and decision != "abstain" and not bool(row.get("abstain")):
            aspect = aspect_from_prediction(row)
            if not implicit_aspect_allowed(aspect, domain):
                rejected = dict(row)
                rejected["abstain"] = True
                rejected["decision"] = "abstain"
                rejected["routing"] = "boundary"
                rejected["reason"] = "domain_mismatch"
                filtered.append(rejected)
                continue
        filtered.append(dict(row))
    return filtered
