from __future__ import annotations

import re
from typing import Dict, List, Set

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Domain-agnostic implicit patterns with family tags.
IMPLICIT_PATTERN_BANK = {
    "electronics": {
        "battery_life": [r"charge(?:d)?\s+twice", r"barely\s+lasts", r"dies\s+by", r"drains\s+fast"],
        "performance": [r"freezes", r"lags", r"slow\s+to\s+open", r"stutters"],
        "heat": [r"felt\s+warm\s+even\s+with\s+light\s+use", r"overheats", r"gets\s+hot"],
    },
    "telecom": {
        "network_quality": [r"calls?\s+keep\s+dropping", r"kept\s+reconnecting", r"signal\s+drops", r"weak\s+signal"],
        "response_time": [r"long\s+hold\s+time", r"no\s+response", r"took\s+forever\s+to\s+reply"],
    },
    "ecommerce": {
        "delivery_speed": [r"arrived\s+late", r"came\s+late", r"delayed", r"after\s+\d+\s*(days?|hours?)"],
        "refund_process": [r"refund\s+took\s+forever", r"return\s+was\s+painful", r"return\s+friction"],
        "order_accuracy": [r"wrong\s+item", r"missing\s+item", r"not\s+what\s+i\s+ordered"],
    },
    "mobility": {
        "comfort": [r"ride\s+felt\s+bumpy", r"seat\s+was\s+uncomfortable"],
        "service_speed": [r"driver\s+took\s+too\s+long", r"waited\s+\d+\s*(mins?|minutes?)"],
    },
    "healthcare": {
        "wait_time": [r"waited\s+\d+\s*(mins?|minutes?|hours?)", r"long\s+queue"],
        "service_speed": [r"had\s+to\s+ask\s+twice\s+before\s+anyone\s+came", r"no\s+one\s+came\s+back"],
    },
    "services": {
        "service_speed": [r"took\s+forever", r"long\s+wait", r"slow\s+service"],
        "customer_support": [r"repeated\s+follow\-ups", r"no\s+callback", r"ticket\s+was\s+ignored"],
        "staff_behavior": [r"rude", r"ignored\s+us", r"dismissive", r"unfriendly"],
    },
}

EXPLICIT_HINTS = {
    "battery": "battery_life", "battery life": "battery_life", "charge": "charging", "charging": "charging",
    "performance": "performance", "screen": "display", "display": "display", "panel": "display",
    "delivery": "delivery_reliability", "shipping": "delivery_speed", "courier": "delivery_reliability",
    "staff": "staff_behavior", "service": "service_speed", "service center": "customer_support", "support": "customer_support",
    "food": "food_quality", "taste": "food_quality",
    "price": "price", "network": "network_quality", "support": "customer_support", "room": "room_quality", "wait": "wait_time",
}

CANONICAL_HEADS = {
    "battery_life": ["battery"],
    "network_quality": ["network", "signal", "calls"],
    "service_speed": ["service"],
    "delivery_speed": ["delivery", "shipping", "courier"],
    "performance": ["performance", "speed"],
    "staff_behavior": ["staff", "waiter", "agent"],
    "heat": ["heat", "hot", "warm"],
    "refund_process": ["refund", "return"],
}


def sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def _contrastive_split(sentence: str) -> List[str]:
    return [p.strip() for p in re.split(r"\bbut\b|\bhowever\b|\balthough\b", sentence, flags=re.IGNORECASE) if p.strip()]


def extract_explicit_aspects(text: str, max_candidates: int = 10) -> List[Dict]:
    sents = sentences(text)
    found: List[Dict] = []
    seen: Set[str] = set()

    for sent in sents:
        for chunk in _contrastive_split(sent):
            low = chunk.lower()
            for key, mapped in EXPLICIT_HINTS.items():
                if re.search(rf"\b{re.escape(key)}\b", low):
                    tag = f"{key}|{chunk}"
                    if tag in seen:
                        continue
                    seen.add(tag)
                    found.append({
                        "aspect_raw": key,
                        "aspect_seed": mapped,
                        "evidence_sentence": chunk,
                        "aspect_type": "explicit",
                        "confidence": 0.9 if mapped in {"battery_life", "charging", "display", "service_speed", "delivery_speed", "delivery_reliability", "customer_support"} else 0.8,
                        "domain_family": "generic",
                    })
                    if len(found) >= max_candidates:
                        return found
    return found


def _has_explicit_head(canonical: str, sentence: str) -> bool:
    heads = CANONICAL_HEADS.get(canonical, [canonical.split("_")[0]])
    low = sentence.lower()
    return any(re.search(rf"\b{re.escape(h)}\b", low) for h in heads)


def extract_implicit_aspects(text: str, explicit_aspects: List[Dict]) -> List[Dict]:
    sents = sentences(text)
    out: List[Dict] = []
    explicit_tokens = " ".join(a.get("aspect_raw", "") for a in explicit_aspects).lower()

    for family, bank in IMPLICIT_PATTERN_BANK.items():
        for canonical, patterns in bank.items():
            for sent in sents:
                low = sent.lower()
                match = next((p for p in patterns if re.search(p, low)), None)
                if not match:
                    continue
                if canonical.split("_")[0] in explicit_tokens:
                    continue
                if _has_explicit_head(canonical, sent):
                    continue
                out.append({
                    "aspect_raw": canonical,
                    "aspect_seed": canonical,
                    "evidence_sentence": sent,
                    "aspect_type": "implicit",
                    "implicit_rationale": match,
                    "confidence": 0.84,
                    "domain_family": family,
                })
                break

    generic_implicit_map = {
        "battery_life": ["lasts", "drains", "charge", "battery", "dies", "unplugged"],
        "performance": ["slow", "lag", "freezes", "stutter", "hang", "crash", "crashed", "freezing"],
        "service_speed": ["waited", "long wait", "took forever", "slow"],
        "staff_behavior": ["rude", "ignored", "unfriendly", "dismissive"],
        "delivery_speed": ["late", "delayed", "arrived after", "came late"],
        "customer_support": ["support", "helpdesk", "ticket", "callback", "service center", "customer service"],
        "food_quality": ["tasty", "bland", "fresh", "stale", "good", "bad"],
        "display": ["screen", "bright", "dim", "flicker", "blue screen", "no gui"],
        "network_quality": ["connected properly", "disconnect", "disconnects", "reconnect", "usb devices"],
    }
    sentiment_cues = {"good", "bad", "poor", "great", "terrible", "slow", "rude", "excellent", "awful", "awesome"}
    for sent in sents:
        low = sent.lower()
        if _has_explicit_head("performance", sent) or _has_explicit_head("battery_life", sent) or _has_explicit_head("service_speed", sent):
            continue
        if not any(c in low for c in sentiment_cues):
            continue
        for canonical, cues in generic_implicit_map.items():
            if canonical in explicit_tokens:
                continue
            if any(cue in low for cue in cues):
                out.append({
                    "aspect_raw": canonical,
                    "aspect_seed": canonical,
                    "evidence_sentence": sent,
                    "aspect_type": "implicit",
                    "implicit_rationale": "sentiment_cue_without_direct_aspect",
                    "confidence": 0.72,
                    "domain_family": family,
                })
                break

    return out


def extract_aspects(text: str, max_candidates: int = 10) -> List[Dict]:
    explicit = extract_explicit_aspects(text, max_candidates=max_candidates)
    implicit = extract_implicit_aspects(text, explicit)
    merged: List[Dict] = []
    seen = set()
    for item in explicit + implicit:
        k = (item.get("aspect_raw", ""), item.get("aspect_type", ""), item.get("evidence_sentence", ""))
        if k in seen:
            continue
        seen.add(k)
        merged.append(item)
    return merged[:max_candidates]
