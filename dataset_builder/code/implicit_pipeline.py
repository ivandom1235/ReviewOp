from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Dict, List

from utils import normalize_whitespace, split_sentences, tokenize


POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "clean", "comfortable", "delicious", "easy",
    "excellent", "fantastic", "fast", "friendly", "good", "great", "helpful",
    "intuitive", "nice", "perfect", "quick", "responsive", "smooth", "stunning",
    "tasty", "wonderful", "love",
}

NEGATIVE_WORDS = {
    "awful", "bad", "bland", "broken", "cheap", "confusing", "crash", "crashes",
    "dirty", "expensive", "frustrating", "horrible", "laggy", "late", "noisy",
    "poor", "rude", "slow", "terrible", "uncomfortable", "unhelpful", "worst",
}

GENERIC_ASPECT_WORDS = {
    "item", "thing", "things", "stuff", "part", "parts", "place", "product",
    "service", "quality", "experience", "overall", "food", "review", "reviews",
    "something", "anything", "everything", "someone", "anyone",
    "use", "used", "using", "priced", "pricey",
}

STOP_ASPECTS = {
    "after", "always", "because", "before", "best", "better", "never", "really",
    "time", "thing", "things", "stuff", "place", "product", "service", "quality",
    "experience", "food", "overall", "good", "great", "bad",
}

STOP_TOKENS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had",
    "have", "has", "he", "her", "here", "his", "i", "if", "in", "is", "it", "its",
    "me", "my", "not", "of", "on", "or", "our", "she", "so", "than", "that",
    "the", "their", "them", "there", "these", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "while", "with", "you", "your",
}

NOUNISH_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ship", "age", "ance", "ence", "hood", "ware", "screen", "pad")

LATENT_ASPECT_RULES = [
    ("value", {"price", "prices", "cost", "costs", "expensive", "cheap", "priced", "pricey", "value", "inexpensive", "affordable", "reasonable", "worth", "fair"}),
    ("power", {"battery", "batteries", "battery life", "power", "charger", "charging", "plug", "adapter"}),
    ("display quality", {"screen", "display", "monitor", "brightness", "resolution", "lighting", "bright", "dim"}),
    ("input quality", {"keyboard", "trackpad", "touchpad", "mouse", "input", "touch"}),
    ("performance", {"performance", "processor", "cpu", "speed", "fast", "slow", "lag", "responsive", "program", "software", "windows", "driver", "app", "efficient", "smooth"}),
    ("food quality", {"food", "meal", "meals", "dish", "dishes", "pizza", "sushi", "dumplings", "knishes", "sandwich", "salad", "dessert", "breakfast", "lunch", "dinner", "taste", "tasty", "flavor", "fresh", "soup", "soups", "toppings", "burger", "burgers", "chicken", "beef", "pasta", "noodle", "noodles", "rice", "coffee", "pastries"}),
    ("service quality", {"service", "staff", "waiter", "waitress", "host", "support", "help desk", "helpdesk", "crew", "doctor", "nurse", "pharmacist", "driver", "chauffeur", "conductor", "coach", "stylist", "guide", "teacher", "instructor"}),
    ("dining experience", {"table", "tables", "seating", "seat", "decor", "atmosphere", "ambience", "ambiance", "space", "restaurant", "caf?", "cafe"}),
    ("cleanliness", {"clean", "dirty", "messy", "hygiene", "neat", "tidy"}),
    ("menu variety", {"menu", "menus", "selection", "options", "option", "variety", "available", "in stock"}),
    ("reliability", {"issue", "issues", "problem", "problems", "freeze", "freezes", "freezing", "crash", "crashes", "crashing", "heat", "heating", "motherboard", "broken", "durable", "stable", "damage", "damaged", "tracking"}),
    ("portability", {"compact", "portable", "portability", "size", "small", "thin", "light", "lightweight", "cramped", "crowded"}),
    ("availability", {"reservation", "reservations", "wait", "waiting", "availability", "available", "appointment", "schedule", "booking", "stock", "in stock"}),
    ("support and warranty", {"warranty", "applecare", "support", "replace", "replacement", "refund", "complaint", "claim"}),
    ("software usability", {"chrome", "pages", "keynote", "interface", "usable", "usability", "easy to use", "design tools", "documents", "presentations", "app", "website", "site"}),
    ("compatibility", {"cd drive", "usb", "nvidia", "drivers", "gps"}),
    ("timeliness", {"on time", "late", "delay", "delayed", "delays", "quick", "fast", "prompt", "promptly", "too long", "took too long", "waited too long", "arrived", "arrive"}),
    ("communication", {"explained", "inform", "informed", "announcement", "announcements", "clear", "clearly", "reply", "replied", "notification", "email", "email confirmation", "support", "follow", "follow up"}),
    ("comfort", {"comfortable", "uncomfortable", "crowded", "cramped", "relaxing", "peaceful", "spacious", "noisy", "quiet", "cold", "hot", "stuffy", "smooth", "soft"}),
    ("organization", {"organized", "organised", "neat", "orderly", "clear", "accurate", "tidy", "structured", "well organized"}),
    ("visibility", {"bright", "dim", "lighting", "labels", "read", "readable", "easy to read", "hard to read", "screen", "artwork"}),
    ("accessibility", {"easy to navigate", "navigate", "navigation", "directions", "easy to follow", "aisle", "parking", "pickup", "accessible", "access", "simple"}),
    ("navigation", {"map", "maps", "route", "rerouting", "pickup", "station", "platform", "signage", "signs", "parking lot", "location"}),
    ("process", {"process", "checkout", "booking", "payment", "register", "registration", "loan", "refund", "order", "ordering", "reservation", "pickup", "delivery", "refill", "confirmation", "paperwork", "treatment"}),
    ("safety", {"safe", "unsafe", "security", "secure", "securely", "broken", "damage", "damaged", "elevator", "crash"}),
    ("location", {"location", "branch", "store", "room", "area", "lot", "station", "lobby", "parking"}),
    ("efficiency", {"efficient", "efficiency", "smooth", "simple", "easy", "quick", "fast", "organized"}),
    ("clarity", {"clear", "clearly", "confusing", "unclear", "understand", "understandable", "hard to understand", "hard to read", "label", "labels", "instructions"}),
    ("transparency", {"transparent", "transparency", "clear", "fees", "billing", "paperwork", "policy", "policy"}),
    ("convenience", {"convenient", "easy", "simple", "handy", "helpful", "quick", "fast"}),
    ("security", {"secure", "securely", "safe", "privacy", "password", "login", "authentication"}),
    ("sound quality", {"sound", "audio", "music", "volume", "loud", "quiet", "noise", "noisy", "speaker", "speakers"}),
]

VALID_LATENT_ASPECTS = {label for label, _keywords in LATENT_ASPECT_RULES} | {"general"}

DOMAIN_FAMILY_MAP = {
    "hotel": "hospitality",
    "restaurant": "hospitality",
    "gym": "wellness",
    "hospital": "healthcare",
    "pharmacy": "healthcare",
    "airline": "transport",
    "taxi": "transport",
    "train": "transport",
    "delivery": "logistics",
    "bookstore": "retail",
    "grocery": "retail",
    "bank": "finance",
    "school": "education",
    "museum": "culture",
    "cinema": "media",
    "salon": "beauty",
}

DOMAIN_FAMILY_PRIORS = {
    "hospitality": {"cleanliness", "comfort", "service quality", "value", "timeliness", "availability", "location", "organization"},
    "healthcare": {"communication", "timeliness", "cleanliness", "reliability", "comfort", "availability", "clarity", "support and warranty", "safety"},
    "transport": {"timeliness", "reliability", "safety", "comfort", "navigation", "communication", "accessibility", "availability", "location"},
    "logistics": {"timeliness", "reliability", "communication", "safety", "process", "availability", "location"},
    "retail": {"value", "availability", "cleanliness", "organization", "convenience", "process", "communication", "efficiency"},
    "finance": {"reliability", "security", "transparency", "communication", "process", "timeliness", "value"},
    "education": {"communication", "clarity", "organization", "availability", "comfort", "value", "safety"},
    "culture": {"display quality", "visibility", "comfort", "value", "availability", "communication"},
    "media": {"display quality", "sound quality", "comfort", "navigation", "value", "availability"},
    "beauty": {"comfort", "service quality", "cleanliness", "timeliness", "communication", "value"},
    "wellness": {"cleanliness", "comfort", "service quality", "timeliness", "reliability", "value"},
}

DOMAIN_FAMILY_BOOST = 0.75


def _normalize_aspect(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip("- ")
    if not text:
        return text
    return " ".join(_singularize_token(token) for token in text.split())


def _singularize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes", "ses")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _match_aspect_surface(text: str, aspect: str) -> tuple[str | None, str | None]:
    normalized = _normalize_aspect(aspect)
    if not normalized:
        return None, None

    patterns = [normalized]
    singularized = " ".join(_singularize_token(token) for token in normalized.split())
    if singularized != normalized:
        patterns.append(singularized)

    for pattern in patterns:
        regex = re.compile(rf"\b{re.escape(pattern)}\b", flags=re.IGNORECASE)
        match = regex.search(text)
        if match:
            support_type = "exact" if pattern == normalized else "near_exact"
            return match.group(0), support_type

    return None, None


def _latent_aspect_label(aspect: str, clause: str | None = None) -> str:
    normalized = _normalize_aspect(aspect)
    haystack = normalized
    for label, keywords in LATENT_ASPECT_RULES:
        if any(keyword in haystack for keyword in keywords):
            return label
    clause_haystack = normalize_whitespace(clause or "").lower()
    for label, keywords in LATENT_ASPECT_RULES:
        if any(keyword in clause_haystack for keyword in keywords):
            return label
    return "general"


def _resolve_domain_family(domain: str | None) -> str | None:
    if not domain:
        return None
    return DOMAIN_FAMILY_MAP.get(_normalize_aspect(domain))


def _facet_priority(domain_family: str | None) -> set[str]:
    if not domain_family:
        return set()
    return DOMAIN_FAMILY_PRIORS.get(domain_family, set())


def _score_facet(text: str, facet: str, keywords: set[str], *, domain_family: str | None = None) -> tuple[float, str | None]:
    lowered = normalize_whitespace(text).lower()
    best_surface: str | None = None
    best_surface_len = -1
    score = 0.0
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE)
        match = pattern.search(lowered)
        if not match:
            continue
        surface = match.group(0)
        score += 1.0 + (0.35 if " " in keyword else 0.0)
        if len(surface) > best_surface_len:
            best_surface = surface
            best_surface_len = len(surface)
    if facet in _facet_priority(domain_family):
        score += DOMAIN_FAMILY_BOOST
    if score > 0 and facet == "value" and re.search(r"(?:[$€£]\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:dollars?|bucks?|eur(?:os)?|pounds?)\b)", lowered, flags=re.IGNORECASE):
        score += 1.25
    if score > 0 and facet in {"timeliness", "process", "communication", "clarity", "comfort", "organization", "accessibility", "navigation", "visibility", "sound quality"}:
        if any(phrase in lowered for phrase in ("too long", "on time", "easy to", "hard to", "too crowded", "too noisy", "too loud", "too dim", "too cold", "easy to read", "hard to read", "clear", "clearly", "organized", "organised", "smooth", "simple")):
            score += 0.25
    return score, best_surface


def _is_valid_latent_aspect(value: str) -> bool:
    return _normalize_aspect(value) in VALID_LATENT_ASPECTS


def _find_latent_matches(text: str) -> list[dict[str, Any]]:
    normalized = normalize_whitespace(text)
    lowered = normalized.lower()
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for latent_label, keywords in LATENT_ASPECT_RULES:
        if latent_label == "value":
            currency_match = re.search(r"(?:[$€£]\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:dollars?|bucks?|eur(?:os)?|pounds?)\b)", lowered, flags=re.IGNORECASE)
            if currency_match:
                key = (latent_label, currency_match.group(0).lower(), currency_match.start(), currency_match.end())
                if key not in seen:
                    seen.add(key)
                    matches.append({
                        "latent_aspect": latent_label,
                        "surface_aspect": currency_match.group(0),
                        "normalized_aspect": latent_label,
                        "matched_surface": currency_match.group(0),
                        "support_type": "exact",
                        "start_char": currency_match.start(),
                        "end_char": currency_match.end(),
                    })
        for keyword in sorted(keywords, key=len, reverse=True):
            regex = re.compile(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE)
            match = regex.search(lowered)
            if not match:
                continue
            key = (latent_label, match.group(0).lower(), match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            matches.append({
                "latent_aspect": latent_label,
                "surface_aspect": match.group(0),
                "normalized_aspect": latent_label,
                "matched_surface": match.group(0),
                "support_type": "exact",
                "start_char": match.start(),
                "end_char": match.end(),
            })
            break
    return matches


def _find_latent_matches_v2(text: str, *, domain: str | None = None) -> list[dict[str, Any]]:
    normalized = normalize_whitespace(text)
    lowered = normalized.lower()
    domain_family = _resolve_domain_family(domain)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()

    for latent_label, keywords in LATENT_ASPECT_RULES:
        score, surface = _score_facet(lowered, latent_label, keywords, domain_family=domain_family)
        if latent_label == "value":
            currency_match = re.search(r"(?:[$€£]\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:dollars?|bucks?|eur(?:os)?|pounds?)\b)", lowered, flags=re.IGNORECASE)
            if currency_match:
                score += 1.25
                surface = surface or currency_match.group(0)
        if score <= 0 or not surface:
            continue
        pattern = re.compile(rf"\b{re.escape(surface)}\b", flags=re.IGNORECASE)
        match = pattern.search(lowered)
        if not match:
            continue
        key = (latent_label, match.group(0).lower(), match.start(), match.end())
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "latent_aspect": latent_label,
            "surface_aspect": match.group(0),
            "normalized_aspect": latent_label,
            "matched_surface": match.group(0),
            "support_type": "exact",
            "score": round(score, 4),
            "start_char": match.start(),
            "end_char": match.end(),
        })

    candidates.sort(key=lambda item: (-item["score"], -(item["end_char"] - item["start_char"]), item["start_char"], item["latent_aspect"]))
    matches: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for candidate in candidates:
        start = int(candidate["start_char"])
        end = int(candidate["end_char"])
        if any(not (end <= span_start or start >= span_end) for span_start, span_end in occupied):
            continue
        occupied.append((start, end))
        matches.append(candidate)
    return matches


def _is_meaningful_aspect(value: str) -> bool:
    aspect = _normalize_aspect(value)
    if not aspect or len(aspect) < 3:
        return False
    tokens = [token for token in aspect.split() if token]
    if not tokens:
        return False
    if all(token in STOP_TOKENS for token in tokens):
        return False
    if aspect in STOP_ASPECTS or aspect in GENERIC_ASPECT_WORDS:
        return False
    if any(token in STOP_ASPECTS or token in GENERIC_ASPECT_WORDS for token in tokens):
        return False
    if all(len(token) < 3 for token in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if token in STOP_TOKENS or token in STOP_ASPECTS or token in GENERIC_ASPECT_WORDS:
            return False
        if token in POSITIVE_WORDS or token in NEGATIVE_WORDS:
            return False
        if token.isdigit():
            return False
    if any(token in POSITIVE_WORDS or token in NEGATIVE_WORDS for token in tokens):
        return False
    return True


def _extract_gold_aspects(rows: List[Dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key in ("aspect", "gold_aspect", "target_aspect"):
            value = row.get(key)
            if isinstance(value, str) and _is_meaningful_aspect(value):
                counts[_normalize_aspect(value)] += 1
        if isinstance(row.get("gold_labels"), list):
            for label in row["gold_labels"]:
                if isinstance(label, dict):
                    for key in ("aspect", "text", "implicit_aspect"):
                        value = label.get(key)
                        if isinstance(value, str) and _is_meaningful_aspect(value):
                            counts[_normalize_aspect(value)] += 1
    return [aspect for aspect, _count in counts.most_common()]


def _extract_primary_gold_aspect(row: Dict[str, Any]) -> str | None:
    for key in ("aspect", "gold_aspect", "target_aspect"):
        value = row.get(key)
        if isinstance(value, str) and _is_meaningful_aspect(value):
            return _normalize_aspect(value)
    if isinstance(row.get("gold_labels"), list):
        for label in row["gold_labels"]:
            if isinstance(label, dict):
                for key in ("aspect", "text", "implicit_aspect"):
                    value = label.get(key)
                    if isinstance(value, str) and _is_meaningful_aspect(value):
                        return _normalize_aspect(value)
    return None


def _sentence_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for sentence in split_sentences(text):
        parts = [part.strip(" ,;:-") for part in re.split(r"\bbut\b|\bhowever\b|\byet\b|\bwhile\b|\balthough\b", sentence, flags=re.IGNORECASE) if part.strip()]
        clauses.extend(parts or [sentence])
    return clauses or [normalize_whitespace(text)]


def infer_sentiment(text: str) -> str:
    tokens = tokenize(text)
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def discover_aspects(rows: List[Dict[str, Any]], *, text_column: str, max_aspects: int, implicit_mode: str = "heuristic") -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text:
            continue
        for clause in _sentence_clauses(text):
            for match in _find_latent_matches_v2(clause, domain=str(row.get("domain") or row.get("source_file", ""))):
                counts[str(match["latent_aspect"])] += 1
        if implicit_mode == "benchmark":
            for aspect in _extract_gold_aspects([row]):
                counts[_latent_aspect_label(aspect, text)] += 1
    return [aspect for aspect, _count in counts.most_common(max_aspects)] or ["general"]


def build_implicit_row(
    row: Dict[str, Any],
    *,
    text_column: str,
    candidate_aspects: List[str],
    confidence_threshold: float,
    row_index: int,
    domain: str | None = None,
    implicit_mode: str = "heuristic",
    chunk_offset: int = 0,
) -> Dict[str, Any]:
    text = normalize_whitespace(row.get(text_column, ""))
    clauses = _sentence_clauses(text)
    sentiment = infer_sentiment(text)
    aspects: list[str] = []
    aspect_confidence: dict[str, float] = {}
    aspect_sentiments: dict[str, str] = {}
    spans: list[dict[str, Any]] = []

    for clause in clauses:
        clause_sentiment = infer_sentiment(clause)
        latent_matches = _find_latent_matches_v2(clause, domain=domain or str(row.get("domain", "")))
        if implicit_mode == "benchmark" and not latent_matches:
            benchmark_hints: list[str] = []
            for aspect in (
                _extract_primary_gold_aspect(row),
                *[_normalize_aspect(aspect) for aspect in (row.get("aspect"), row.get("gold_aspect"), row.get("target_aspect")) if isinstance(aspect, str)],
            ):
                if aspect and _is_meaningful_aspect(aspect) and aspect not in benchmark_hints:
                    benchmark_hints.append(aspect)
            if isinstance(row.get("gold_labels"), list):
                for label in row["gold_labels"]:
                    if isinstance(label, dict):
                        for key in ("aspect", "text", "implicit_aspect"):
                            value = label.get(key)
                            if isinstance(value, str):
                                aspect = _normalize_aspect(value)
                                if aspect and _is_meaningful_aspect(aspect) and aspect not in benchmark_hints:
                                    benchmark_hints.append(aspect)
            benchmark_hints.sort(key=lambda value: (-len(value.split()), -len(value), value))
            for aspect in benchmark_hints:
                matched_surface, support_type = _match_aspect_surface(clause, aspect)
                if not matched_surface:
                    continue
                latent_aspect = _latent_aspect_label(aspect, clause)
                if latent_aspect == "general":
                    continue
                latent_matches.append({
                    "latent_aspect": latent_aspect,
                    "surface_aspect": aspect,
                    "normalized_aspect": latent_aspect,
                    "matched_surface": matched_surface,
                    "support_type": support_type or "near_exact",
                    "start_char": max(0, clause.lower().find(matched_surface.lower())),
                    "end_char": max(0, clause.lower().find(matched_surface.lower()) + len(matched_surface)),
                })
        for match in latent_matches:
            latent_aspect = str(match["latent_aspect"])
            if not _is_valid_latent_aspect(latent_aspect):
                continue
            confidence = 0.99 if match.get("support_type") == "exact" else 0.75
            if confidence < confidence_threshold:
                continue
            if latent_aspect not in aspects:
                aspects.append(latent_aspect)
            aspect_confidence[latent_aspect] = round(confidence, 4)
            aspect_sentiments[latent_aspect] = clause_sentiment
            spans.append({
                "surface_aspect": match.get("surface_aspect"),
                "latent_aspect": latent_aspect,
                "normalized_aspect": latent_aspect,
                "matched_surface": match.get("matched_surface"),
                "support_type": match.get("support_type", "exact"),
                "sentiment": clause_sentiment,
                "confidence": round(confidence, 4),
                "start_char": match.get("start_char", 0),
                "end_char": match.get("end_char", 0),
                "clause": clause,
            })
    if not aspects:
        fallback = "general"
        aspects = [fallback]
        aspect_confidence[fallback] = 0.5
        aspect_sentiments[fallback] = sentiment
    weak_support = any(span.get("support_type") != "exact" for span in spans)
    needs_review = aspects == ["general"] or weak_support
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "source_text": text,
        "implicit": {
            "aspects": aspects,
            "dominant_sentiment": sentiment,
            "aspect_sentiments": aspect_sentiments,
            "aspect_confidence": aspect_confidence,
            "spans": spans,
            "sentence_count_processed": len(clauses),
            "row_index": row_index + chunk_offset,
            "needs_review": needs_review,
            "review_reason": "fallback_general" if aspects == ["general"] else ("near_exact_support" if weak_support else None),
        },
    }


def collect_diagnostics(rows: List[Dict[str, Any]], *, text_column: str, candidate_aspects: List[str]) -> Dict[str, Any]:
    aspect_counts: Counter[str] = Counter()
    exact_span_count = 0
    near_exact_span_count = 0
    fallback_only_count = 0
    needs_review_count = 0
    rejected_aspect_count = 0
    clause_count = 0
    for row in rows:
        text = normalize_whitespace(row.get(text_column, ""))
        for clause in _sentence_clauses(text):
            clause_count += 1
        implicit = row.get("implicit", {})
        for aspect in implicit.get("aspects", []):
            if aspect != "general":
                aspect_counts[str(aspect)] += 1
        if implicit.get("aspects") == ["general"]:
            fallback_only_count += 1
        if implicit.get("needs_review"):
            needs_review_count += 1
        for span in implicit.get("spans", []):
            if span.get("support_type") == "exact":
                exact_span_count += 1
            elif span.get("support_type") == "near_exact":
                near_exact_span_count += 1
        for aspect in implicit.get("aspects", []):
            if not _is_valid_latent_aspect(str(aspect)):
                rejected_aspect_count += 1
    total = sum(aspect_counts.values())
    entropy = 0.0
    if total:
        for count in aspect_counts.values():
            share = count / total
            entropy -= share * math.log(share, 2)
    return {
        "candidate_aspects": candidate_aspects,
        "top_implicit_aspects": aspect_counts.most_common(10),
        "meaningful_candidate_aspects": [aspect for aspect in candidate_aspects if _is_valid_latent_aspect(aspect)],
        "generic_candidate_aspects": [aspect for aspect in candidate_aspects if not _is_valid_latent_aspect(aspect)],
        "clause_count": clause_count,
        "span_support": {
            "exact": exact_span_count,
            "near_exact": near_exact_span_count,
        },
        "fallback_only_count": fallback_only_count,
        "needs_review_count": needs_review_count,
        "rejected_aspect_count": rejected_aspect_count,
        "domain_agnostic_skew": {
            "unique_aspects": len(aspect_counts),
            "top_share": round(max(aspect_counts.values()) / total, 4) if total else 0.0,
            "entropy": round(entropy, 4),
        },
    }
