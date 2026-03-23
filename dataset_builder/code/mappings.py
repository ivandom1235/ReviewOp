"""Shared mapping dictionaries and lightweight ontologies for Universal Taxonomy."""
from __future__ import annotations

UNIVERSAL_SUPERCLASSES = [
    "product_quality", "performance", "usability", "reliability",
    "aesthetics", "value", "support_quality", "delivery_logistics",
    "safety", "compatibility", "sustainability", "experience"
]

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
    "price": ["rip off", "waste of money", "overpriced", "great deal", "steal"],
    "durability": ["broke after two uses", "fell apart", "shattered easily", "built like a tank", "flimsy"],
    "customer_service": ["hung up on me", "transferred me three times", "refused to help", "issued a refund immediately"],
    "shipping": ["arrived crushed", "box was opened", "came two weeks late", "next day delivery"],
    "taste": ["bland", "overcooked", "salty", "mouthwatering", "flavorful", "tasteless"],
    "bed_comfort": ["woke up with back pain", "slept like a baby", "mattress was like a rock", "springs poking"],
    "uiux": ["couldn't find the menu", "so intuitive", "took ten clicks to do simple task", "cluttered dashboard"],
    "crash_rate": ["lost all my work", "keeps closing on its own", "blue screen", "stable and reliable"],
}

DOMAIN_HINTS = {
    "electronics": ["laptop", "phone", "battery", "charger", "screen", "device"],
    "food": ["restaurant", "food", "taste", "dish", "waiter", "menu", "dinner"],
    "hospitality": ["hotel", "room", "checkin", "checkout", "stay", "resort"],
    "software": ["app", "software", "ui", "button", "subscription", "download", "tool"],
}

