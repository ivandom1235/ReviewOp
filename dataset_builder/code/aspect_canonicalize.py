from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from llm_utils import LLMClient
from prompts import ASPECT_MAP_PROMPT

MASTER_TAXONOMY: List[str] = [
    "battery_life", "charging", "performance", "display", "audio", "camera", "build_quality", "portability",
    "price", "value_for_money", "network_quality", "software", "storage", "keyboard", "touchpad", "heat",
    "design", "durability", "customer_support", "warranty", "delivery_speed", "delivery_reliability",
    "packaging", "order_accuracy", "service_speed", "staff_behavior", "cleanliness", "ambience",
    "food_quality", "menu_variety", "portion_size", "room_quality", "checkin_process", "location",
    "comfort", "wait_time", "refund_process", "availability", "response_time", "general_experience",
]

ASPECT_DEFINITIONS: Dict[str, str] = {
    "service_speed": "time taken to serve, respond, or resolve",
    "staff_behavior": "attitude, politeness, empathy, and conduct of human staff",
    "customer_support": "post-purchase support channels, ticket handling, helpdesk quality",
    "performance": "execution speed, responsiveness, lag/stutter during usage",
    "response_time": "time to first response/reply from system or support",
    "delivery_speed": "how quickly order reaches customer",
    "delivery_reliability": "whether delivery succeeds safely/consistently",
    "order_accuracy": "whether delivered item matches ordered item",
}

ASPECT_ALIASES: Dict[str, str] = {
    "battery": "battery_life", "battery backup": "battery_life", "dies quickly": "battery_life", "drains fast": "battery_life",
    "charger": "charging", "charging": "charging",
    "slow": "performance", "lag": "performance", "lags": "performance", "freeze": "performance",
    "screen": "display", "display": "display", "sound": "audio", "speaker": "audio", "camera": "camera",
    "build": "build_quality", "fragile": "durability", "durable": "durability", "lightweight": "portability", "heavy": "portability",
    "price": "price", "expensive": "price", "worth": "value_for_money",
    "network": "network_quality", "signal": "network_quality", "software": "software", "storage": "storage",
    "keyboard": "keyboard", "touchpad": "touchpad", "heating": "heat", "hot": "heat", "design": "design",
    "support": "customer_support", "warranty": "warranty",
    "delivery": "delivery_reliability", "late": "delivery_speed", "shipping": "delivery_speed", "courier": "delivery_reliability",
    "packaging": "packaging", "wrong item": "order_accuracy", "missing item": "order_accuracy",
    "service": "service_speed", "staff": "staff_behavior", "agent": "customer_support",
    "clean": "cleanliness", "ambience": "ambience", "food": "food_quality", "taste": "food_quality",
    "menu": "menu_variety", "portion": "portion_size", "room": "room_quality", "checkin": "checkin_process",
    "location": "location", "comfort": "comfort", "wait": "wait_time", "refund": "refund_process",
    "availability": "availability", "response": "response_time",
}

COLLAPSE_BLOCKLIST = {"quality", "service", "delivery", "general", "experience"}


# domain+phrase deterministic policy cache
_PHRASE_POLICY_CACHE: Dict[Tuple[str, str], str] = {}


def _normalize(raw: str) -> str:
    return " ".join(str(raw or "").strip().lower().replace("_", " ").split())


def _definition_compatibility(label: str, sentence: str) -> float:
    if label not in ASPECT_DEFINITIONS:
        return 0.5
    d = ASPECT_DEFINITIONS[label].lower()
    s = sentence.lower()
    key_words = [w for w in d.replace(",", " ").split() if len(w) > 4]
    overlap = sum(1 for w in key_words if w in s)
    return 0.55 + min(0.35, overlap * 0.08)


def _disambiguate_generic(raw: str, sentence: str, domain: str) -> str:
    low = f"{raw} {sentence} {domain}".lower()
    if "quality" in raw.lower():
        if any(x in low for x in ["food", "taste", "dish"]):
            return "food_quality"
        if any(x in low for x in ["build", "material", "hinge", "body", "engine"]):
            return "build_quality"
        if any(x in low for x in ["service", "staff", "wait", "support"]):
            return "service_speed"
        return "general_experience"
    if "service" in raw.lower():
        if any(x in low for x in ["rude", "polite", "staff", "waiter", "friendly", "behavior"]):
            return "staff_behavior"
        if any(x in low for x in ["ticket", "helpdesk", "support", "agent", "call center"]):
            return "customer_support"
        if any(x in low for x in ["slow", "fast", "delay", "waiting", "minutes", "queue"]):
            return "service_speed"
        return "service_speed"
    if "delivery" in raw.lower():
        if any(x in low for x in ["late", "delay", "hours", "days", "arrived"]):
            return "delivery_speed"
        if any(x in low for x in ["wrong item", "missing", "damaged", "lost"]):
            return "order_accuracy"
        return "delivery_reliability"
    return ""


def _rule_map(raw: str, evidence_sentence: str, domain: str, use_definitions: bool = True) -> Tuple[str, float]:
    r = _normalize(raw)
    if not r:
        return "", 0.0

    if r in COLLAPSE_BLOCKLIST:
        remap = _disambiguate_generic(r, evidence_sentence, domain)
        score = 0.8
        if use_definitions:
            score = min(0.95, (score + _definition_compatibility(remap, evidence_sentence)) / 2)
        return remap or f"other_{domain or 'general'}", score

    if r in ASPECT_ALIASES:
        can = ASPECT_ALIASES[r]
        score = 0.92
        if use_definitions:
            score = min(0.97, (score + _definition_compatibility(can, evidence_sentence)) / 2)
        return can, score

    for k, v in ASPECT_ALIASES.items():
        if k in r or r in k:
            score = 0.72
            if k in COLLAPSE_BLOCKLIST:
                remap = _disambiguate_generic(k, evidence_sentence, domain)
                if use_definitions:
                    score = min(0.9, (score + _definition_compatibility(remap, evidence_sentence)) / 2)
                return remap or f"other_{domain or 'general'}", score
            if use_definitions:
                score = min(0.9, (score + _definition_compatibility(v, evidence_sentence)) / 2)
            return v, score

    candidate = r.replace(" ", "_")
    if candidate in MASTER_TAXONOMY:
        if candidate == "general_experience":
            return candidate, 0.2
        score = 0.68
        if use_definitions:
            score = min(0.88, (score + _definition_compatibility(candidate, evidence_sentence)) / 2)
        return candidate, score
    return f"other_{domain or 'general'}", 0.15


def _apply_phrase_policy(domain: str, phrase: str, canonical: str, confidence: float) -> str:
    key = (domain or "general", phrase)
    prev = _PHRASE_POLICY_CACHE.get(key)
    if prev is None:
        if canonical == "general_experience" or confidence < 0.35:
            canonical = f"other_{domain or 'general'}"
        _PHRASE_POLICY_CACHE[key] = canonical
        return canonical
    if prev == canonical:
        return canonical
    # strict conflict resolution: keep prior unless new mapping is very high confidence
    if confidence >= 0.92 and canonical != "general_experience":
        _PHRASE_POLICY_CACHE[key] = canonical
        return canonical
    return prev


def apply_anti_collapse(aspects: List[Dict], max_share: float) -> List[Dict]:
    if not aspects:
        return aspects
    c = Counter(a.get("aspect_canonical", "") for a in aspects)
    total = len(aspects)
    for a in aspects:
        label = a.get("aspect_canonical", "")
        if not label:
            continue
        share = c[label] / total
        if share > max_share and label.startswith("other_"):
            a["confidence"] = max(0.0, float(a.get("confidence", 0.0)) - 0.15)
    return aspects


def canonicalize_aspects(
    aspects: List[Dict],
    domain: str,
    llm: LLMClient | None = None,
    max_canonical_share: float = 0.45,
    aspect_definitions_enabled: bool = True,
    llm_min_confidence: float = 0.6,
) -> Tuple[List[Dict], Dict[str, str]]:
    mapping: Dict[str, str] = {}
    out: List[Dict] = []
    for a in aspects:
        phrase = _normalize(a.get("aspect_raw", ""))
        canonical, conf = _rule_map(phrase, a.get("evidence_sentence", ""), domain, use_definitions=aspect_definitions_enabled)
        if not canonical:
            continue
        canonical = _apply_phrase_policy(domain, phrase, canonical, conf)
        item = dict(a)
        item["aspect_canonical"] = canonical
        item["confidence"] = min(0.99, max(float(item.get("confidence", 0.0)), conf))
        mapping[phrase] = canonical
        out.append(item)

    if llm is not None and out:
        ambiguous = [
            a
            for a in out
            if (
                not a.get("aspect_canonical")
                or str(a.get("aspect_canonical", "")).startswith("other_")
                or float(a.get("confidence", 0.0)) < float(llm_min_confidence)
            )
        ]
        raws = sorted({a.get("aspect_raw", "") for a in ambiguous if a.get("aspect_raw")})[:12]
        if not raws:
            out = apply_anti_collapse(out, max_share=max_canonical_share)
            return out, mapping
        prompt = (
            ASPECT_MAP_PROMPT
            + f"\nAllowed taxonomy: {MASTER_TAXONOMY}\n"
            + f"Definitions: {ASPECT_DEFINITIONS}\n"
            + f"Domain: {domain}\nRaw aspects: {raws}"
        )
        data = llm.json_completion(prompt)
        for item in data.get("mappings", []) if isinstance(data, dict) else []:
            raw = _normalize(item.get("raw", ""))
            can = _normalize(item.get("canonical", "")).replace(" ", "_")
            conf = float(item.get("confidence", 0.0) or 0.0)
            if raw and can and (can in MASTER_TAXONOMY or can.startswith("other_")) and conf >= 0.7:
                mapping[raw] = can

        for item in out:
            key = _normalize(item.get("aspect_raw", ""))
            if key in mapping:
                item["aspect_canonical"] = _apply_phrase_policy(domain, key, mapping[key], float(item.get("confidence", 0.0)))

    out = apply_anti_collapse(out, max_share=max_canonical_share)
    return out, mapping
