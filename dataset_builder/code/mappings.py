"""Shared mapping dictionaries and lightweight ontologies."""
from __future__ import annotations

CANONICAL_ASPECTS = {
    "electronics": {
        "battery": ["battery", "battery life", "battery backup", "charge", "charging"],
        "display": ["screen", "display", "panel", "resolution"],
        "performance": ["performance", "lag", "laggy", "slow", "freezes", "hang", "crash"],
        "heating": ["heating", "overheat", "hot", "gets hot"],
        "price": ["price", "cost", "value", "expensive", "cheap"],
        "build_quality": ["build", "material", "durability", "quality"],
    },
    "restaurants": {
        "food_quality": ["food", "taste", "flavor", "dish", "meal"],
        "portion_size": ["portion", "serving", "small portions", "not filling"],
        "service": ["service", "waiter", "staff", "attitude", "behavior"],
        "cleanliness": ["clean", "hygiene", "dirty"],
        "pricing": ["price", "expensive", "cheap", "value"],
        "ambience": ["ambience", "atmosphere", "music", "crowd"],
    },
    "hotels": {
        "room_quality": ["room", "bed", "cleanliness", "washroom"],
        "service": ["staff", "service", "reception"],
        "location": ["location", "distance", "area"],
        "pricing": ["price", "cost", "value"],
    },
    "healthcare": {
        "doctor_quality": ["doctor", "physician", "consultation"],
        "staff_behavior": ["nurse", "staff", "behavior", "attitude"],
        "wait_time": ["wait", "delay", "queue", "long time"],
        "cost": ["cost", "billing", "price", "expense"],
    },
    "ride-sharing": {
        "driver_behavior": ["driver", "rude", "polite"],
        "arrival_time": ["late", "delay", "arrival", "wait"],
        "pricing": ["surge", "price", "fare", "cost"],
        "safety": ["safe", "unsafe", "security"],
    },
    "e-commerce": {
        "delivery": ["delivery", "shipping", "late", "courier"],
        "packaging": ["packaging", "damaged", "box"],
        "product_quality": ["quality", "defect", "broken"],
        "pricing": ["price", "discount", "cost", "value"],
        "support": ["support", "customer care", "refund", "return"],
    },
}

GENERIC_ASPECTS = {
    "service": ["service", "staff", "support"],
    "price": ["price", "cost", "value"],
    "quality": ["quality", "good", "bad", "defect"],
    "performance": ["slow", "lag", "freezes", "crash"],
}

SENTIMENT_MAP = {
    "positive": "positive",
    "pos": "positive",
    "1": "positive",
    "negative": "negative",
    "neg": "negative",
    "-1": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "0": "neutral",
}

IMPLICIT_PATTERNS = {
    "battery": ["dies", "drains", "before evening", "won't last", "charge twice"],
    "performance": ["freezes", "laggy", "slow", "stutter", "hangs"],
    "heating": ["gets hot", "overheats", "warms up quickly"],
    "service": ["ignored us", "never came", "rude"],
    "portion_size": ["not filling", "too little", "small portion"],
    "delivery": ["arrived late", "delayed", "never delivered"],
}

DOMAIN_HINTS = {
    "electronics": ["laptop", "phone", "battery", "charger", "screen", "device"],
    "restaurants": ["restaurant", "food", "taste", "dish", "waiter", "menu"],
    "hotels": ["hotel", "room", "checkin", "checkout", "stay"],
    "healthcare": ["doctor", "hospital", "clinic", "appointment", "medicine"],
    "ride-sharing": ["ride", "driver", "cab", "trip", "uber", "ola"],
    "e-commerce": ["order", "delivery", "shipping", "product", "refund"],
}
