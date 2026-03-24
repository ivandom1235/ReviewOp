"""Shared mapping dictionaries and lightweight ontologies for Universal Taxonomy."""
from __future__ import annotations

from typing import Dict, Iterable

UNIVERSAL_SUPERCLASSES = [
    "product_quality", "performance", "usability", "reliability",
    "aesthetics", "value", "support_quality", "delivery_logistics",
    "safety", "compatibility", "sustainability", "experience"
]

BROAD_CANONICAL_ASPECTS = {
    "product_quality",
    "performance",
    "support_quality",
    "value",
    "usability",
    "experience",
}

SURFACE_STOPWORDS = {
    "food",
    "item",
    "items",
    "product",
    "products",
    "thing",
    "things",
    "service",
    "quality",
    "experience",
    "value",
    "performance",
}

UNIVERSAL_ASPECT_HINTS = {
    "product_quality": ["quality", "build", "material", "fabric", "texture", "finish", "stitching"],
    "performance": ["speed", "slow", "fast", "responsive", "fit", "comfort", "battery", "temperature", "service_speed"],
    "usability": ["easy", "difficult", "intuitive", "navigation", "setup", "controls", "ease_of_use"],
    "reliability": ["durable", "broken", "stable", "lifespan", "color_retention", "lasting"],
    "aesthetics": ["appearance", "look", "style", "design", "presentation", "decor"],
    "value": ["price", "cost", "worth", "deal", "pricing", "value_for_money"],
    "support_quality": ["service", "staff", "support", "return_process", "customer_service", "host", "waiter"],
    "delivery_logistics": ["delivery", "shipping", "arrival", "reservation", "check_in", "return_shipping"],
    "compatibility": ["size", "sizing", "windows", "software", "os_support", "integrations"],
    "experience": ["experience", "atmosphere", "ambience", "comfort", "noise_level"],
}

DOMAIN_DETAIL_HINTS = {
    "food": {
        "taste": ["tasty", "taste", "flavor", "flavour", "bland", "salty", "sweet", "sour", "delicious", "mouthwatering", "tasteless"],
        "freshness": ["fresh", "stale", "freshness", "spoiled", "crispy", "soggy"],
        "portion_size": ["portion", "serving", "small", "tiny", "huge", "large", "generous", "filling"],
        "food_temperature": ["cold", "hot", "warm", "lukewarm", "room temperature"],
        "service_speed": ["quick", "slow", "waited", "minutes", "hour", "forever", "prompt", "late"],
        "staff_attitude": ["friendly", "rude", "helpful", "attentive", "dismissive", "polite"],
        "ambience": ["ambience", "atmosphere", "cozy", "loud", "noisy", "music", "crowded"],
        "price": ["price", "cost", "value", "expensive", "cheap", "overpriced", "deal"],
        "menu_item": ["pizza", "burger", "bagel", "sushi", "dessert", "meal", "dish", "salad", "pasta"],
    },
    "electronics": {
        "battery_life": ["battery", "charge", "charging", "outlet", "lasts", "drains"],
        "screen_quality": ["screen", "display", "resolution", "bright", "dim"],
        "thermal_management": ["hot", "heat", "fan", "warm", "thermal"],
        "processing_power": ["fast", "slow", "lag", "responsive", "snappy"],
        "build_quality": ["build", "material", "sturdy", "solid", "cheaply made"],
        "price": ["price", "cost", "expensive", "cheap", "worth"],
        "device_feature": ["keyboard", "touchpad", "webcam", "speaker", "harddrive", "processor", "memory"],
    },
    "hospitality": {
        "room_cleanliness": ["clean", "dirty", "stains", "dust", "smelled"],
        "bed_comfort": ["bed", "mattress", "pillows", "sleep", "comfortable"],
        "staff_politeness": ["staff", "polite", "rude", "friendly", "helpful"],
        "check_in_speed": ["check in", "checkin", "queue", "line", "waited"],
        "price": ["price", "fees", "expensive", "cheap", "worth"],
    },
}

CANONICAL_ASPECTS = {
    "electronics": {
        "performance": ["speed", "battery_life", "processing_power", "connectivity", "thermal_management"],
        "usability": ["interface", "controls", "setup", "ergonomics"],
        "product_quality": ["build_quality", "materials", "durability", "screen_quality"],
        "support_quality": ["warranty", "repair", "customer_service", "tech_support"],
        "value": ["price", "cost", "worth_it"],
        "delivery_logistics": ["shipping", "packaging", "arrival_condition"],
        "reliability": ["breaks_easily", "lifespan", "sturdiness"]
    },
    "food": {
        "performance": ["service_speed", "food_temperature", "portion_size"],
        "product_quality": ["taste", "freshness", "ingredients_quality", "texture"],
        "aesthetics": ["presentation", "plating", "decor", "restaurant_look"],
        "experience": ["ambience", "atmosphere", "noise_level"],
        "value": ["price", "pricing", "value_for_money"],
        "support_quality": ["staff_attitude", "waiter", "host", "attention_to_detail"],
    },
    "hospitality": {
        "product_quality": ["room_cleanliness", "bed_comfort", "amenities"],
        "experience": ["ambience", "view", "noise_level"],
        "support_quality": ["reception", "housekeeping", "staff_politeness"],
        "value": ["price", "hidden_fees", "resort_fee"],
        "delivery_logistics": ["check_in_speed", "valet", "concierge"],
    },
    "software": {
        "performance": ["load_time", "crash_rate", "lag"],
        "usability": ["uiux", "navigation", "ease_of_use", "learning_curve"],
        "compatibility": ["browser_support", "os_support", "integrations"],
        "support_quality": ["documentation", "ticket_response", "customer_success"],
        "value": ["subscription_cost", "pricing_tier", "roi"],
        "reliability": ["uptime", "data_loss", "bugs"],
    }
}

GENERIC_ASPECTS = {
    "service": ["service", "staff", "support"],
    "price": ["price", "cost", "value"],
    "quality": ["quality", "good", "bad", "defect"],
    "performance": ["slow", "lag", "freezes", "crash"],
}

SENTIMENT_MAP = {
    "positive": "positive", "pos": "positive", "1": "positive",
    "negative": "negative", "neg": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral",
}

IMPLICIT_SYMPTOMS = {
    "battery_life": ["dies by noon", "barely lasts the day", "need to charge twice", "always searching for an outlet", "plugged in all day"],
    "service_speed": ["waited 40 minutes", "took forever", "never came back", "immediately seated", "out within minutes"],
    "thermal_management": ["gets hot", "burning my lap", "warm to the touch", "throttles under load", "fan always running"],
    "price": ["rip off", "waste of money", "overpriced", "great deal", "steal", "too expensive"],
    "durability": ["broke after two uses", "fell apart", "shattered easily", "built like a tank", "flimsy", "stopped working"],
    "customer_service": ["hung up on me", "transferred me three times", "refused to help", "issued a refund immediately"],
    "shipping": ["arrived crushed", "box was opened", "came two weeks late", "next day delivery"],
    "taste": ["bland", "overcooked", "salty", "mouthwatering", "flavorful", "tasteless"],
    "bed_comfort": ["woke up with back pain", "slept like a baby", "mattress was like a rock", "springs poking"],
    "uiux": ["couldn't find the menu", "so intuitive", "took ten clicks to do simple task", "cluttered dashboard"],
    "crash_rate": ["lost all my work", "keeps closing on its own", "blue screen", "stable and reliable"],
    "performance": ["sluggish during use", "snappy and responsive", "painfully slow", "worked without slowing down", "lags"],
    "support_quality": ["felt ignored", "went out of their way to help", "dismissive the whole time", "handled it professionally"],
    "product_quality": ["felt cheaply made", "premium right away", "fell apart quickly", "solid from day one"],
    "experience": ["too loud to enjoy", "cozy the whole evening", "atmosphere felt off", "comfortable the whole time"],
    "value": ["not worth the cost", "worth every penny", "way too expensive for what it was", "excellent for the price"],
    "delivery_logistics": ["showed up much later than promised", "arrived right on time", "kept waiting longer than expected", "everything showed up smoothly"],
    "usability": ["harder than it should be", "easy to figure out", "confusing at first glance", "intuitive from the start"],
    "reliability": ["started acting up again", "kept working without issues", "stopped working out of nowhere", "worked every single time"],
    "freshness": ["tasted stale", "freshly made", "crisp and fresh", "no longer fresh"],
    "portion_size": ["left still hungry", "plenty of food", "tiny serving", "huge portion"],
    "food_temperature": ["served cold", "still piping hot", "lukewarm when it arrived", "not hot anymore"],
    "staff_attitude": ["felt genuinely welcomed", "treated like an inconvenience", "felt ignored", "everyone was friendly"],
    "fit": ["fit like a glove", "too tight around the shoulders", "hung awkwardly", "true to size"],
    "fabric_feel": ["itchy after an hour", "felt soft right away", "scratchy fabric", "rough on the skin"],
    "return_process": ["refund took forever", "return was a hassle", "exchange was easy", "return process was simple"],
}

DOMAIN_HINTS = {
    "electronics": ["laptop", "phone", "battery", "charger", "screen", "device"],
    "food": ["restaurant", "food", "taste", "dish", "waiter", "menu", "dinner"],
    "hospitality": ["hotel", "room", "checkin", "checkout", "stay", "resort"],
    "software": ["app", "software", "ui", "button", "subscription", "download", "tool"],
}

# Final exported labels must come from this bounded canonical vocabulary.
EXPORT_CANONICAL_ASPECTS = {
    "product_quality",
    "performance",
    "usability",
    "reliability",
    "aesthetics",
    "value",
    "support_quality",
    "delivery_logistics",
    "compatibility",
    "experience",
    "taste",
    "freshness",
    "portion_size",
    "food_temperature",
    "service_speed",
    "staff_attitude",
    "ambience",
    "price",
    "battery_life",
    "screen_quality",
    "thermal_management",
    "processing_power",
    "build_quality",
    "device_feature",
    "return_process",
    "fit",
    "fabric_feel",
}

CANONICAL_EXPORT_FALLBACK = {
    "food": "product_quality",
    "hospitality": "experience",
    "electronics": "product_quality",
    "software": "usability",
}

CANONICAL_EXPORT_REDISTRIBUTION = {
    "food": {
        "performance": ["service_speed", "food_temperature"],
        "support_quality": {"staff_attitude", "service_speed"},
        "experience": {"ambience"},
        "value": {"price"},
    },
    "electronics": {
        "product_quality": {"screen_quality", "build_quality", "device_feature"},
        "performance": {"battery_life", "thermal_management", "processing_power"},
        "value": {"price"},
        "support_quality": {"return_process"},
    },
    "hospitality": {
        "support_quality": {"service_speed"},
        "experience": {"ambience"},
        "value": {"price"},
    },
    "software": {
        "performance": {"processing_power"},
        "support_quality": {"return_process"},
        "value": {"price"},
    },
}


def _norm_key(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _iter_alias_pairs() -> Iterable[tuple[str, str]]:
    for canonical in EXPORT_CANONICAL_ASPECTS:
        yield canonical, canonical

    for domain_map in DOMAIN_DETAIL_HINTS.values():
        for detail, hints in domain_map.items():
            if detail in EXPORT_CANONICAL_ASPECTS:
                yield detail, detail
                for hint in hints:
                    yield hint, detail

    for family, aliases in UNIVERSAL_ASPECT_HINTS.items():
        if family in EXPORT_CANONICAL_ASPECTS:
            yield family, family
        for alias in aliases:
            if alias in EXPORT_CANONICAL_ASPECTS:
                yield alias, alias

    for domain_map in CANONICAL_ASPECTS.values():
        for family, aliases in domain_map.items():
            if family in EXPORT_CANONICAL_ASPECTS:
                yield family, family
            for alias in aliases:
                if alias in EXPORT_CANONICAL_ASPECTS:
                    yield alias, alias
                elif family in EXPORT_CANONICAL_ASPECTS:
                    yield alias, family

    for family, aliases in GENERIC_ASPECTS.items():
        canonical = family if family in EXPORT_CANONICAL_ASPECTS else ""
        if not canonical:
            canonical = {
                "service": "support_quality",
                "price": "price",
                "quality": "product_quality",
                "performance": "performance",
            }.get(family, "")
        if canonical:
            yield family, canonical
            for alias in aliases:
                yield alias, canonical

    extra_aliases = {
        "battery": "battery_life",
        "display": "screen_quality",
        "screen": "screen_quality",
        "refund": "return_process",
        "returns": "return_process",
        "returns_process": "return_process",
        "customer_service": "support_quality",
        "staff": "staff_attitude",
        "service": "support_quality",
        "bagel": "taste",
        "bagels": "taste",
        "pizza": "taste",
        "burger": "taste",
        "dessert": "taste",
        "meal": "taste",
        "dish": "taste",
        "food": "taste",
        "greatvalue": "price",
        "tooexpensive": "price",
    }
    for alias, canonical in extra_aliases.items():
        if canonical in EXPORT_CANONICAL_ASPECTS:
            yield alias, canonical


CANONICAL_EXPORT_LOOKUP: Dict[str, str] = {}
for alias, canonical in _iter_alias_pairs():
    alias_key = _norm_key(alias)
    canonical_key = _norm_key(canonical)
    if alias_key and canonical_key in EXPORT_CANONICAL_ASPECTS and alias_key not in CANONICAL_EXPORT_LOOKUP:
        CANONICAL_EXPORT_LOOKUP[alias_key] = canonical_key

