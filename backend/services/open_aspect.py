from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from core.config import settings

ARTICLE_PREFIX_RE = re.compile(r"^(the|a|an|this|that|these|those)\s+", flags=re.I)
TRAILING_PUNCT_RE = re.compile(r"[^\w\s\-]+$")
ALPHA_RE = re.compile(r"[a-zA-Z]")

SPECIAL_SHORT_TOKENS = {"5g", "4g", "gps", "ram", "app"}
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
NOUN_POS = {"NOUN", "PROPN"}
NOUN_HEAD_DEPS = {"nsubj", "dobj", "pobj", "attr", "obj"}
LEFT_MODIFIER_DEPS = {"compound", "amod", "poss", "nummod"}

STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "on", "for", "with", "it",
    "this", "that", "during", "very", "really", "just", "too", "also", "so", "as", "at", "from", "by", "be", "been",
    "being", "i", "we", "you", "they", "he", "she", "them", "us", "my", "our", "your", "their",
    "which", "what", "who", "whom", "whose", "where", "when", "why", "how", "there", "here",
}

# Generic single-token nouns that rarely help as aspects.
GENERIC_SINGLE = {
    "thing", "things", "stuff", "product", "item", "device", "service", "experience", "time", "people", "person",
    "phone", "laptop", "watch", "smartwatch", "hotel", "restaurant", "place",
}

# Allow price/value terms (kept).
PRICE_WORDS = {"price", "cost", "value", "pricing", "refund", "charge", "charges", "fee", "fees"}

TIME_UNITS = {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes"}
APPROX = {"about", "around", "almost", "roughly", "nearly", "approximately"}

# Context/conditions that are almost never "aspects".
CONTEXT_SINGLE = {
    "daylight", "sunlight", "indoors", "outdoors", "outside", "inside", "weather", "room", "area", "place", "location",
    "morning", "evening", "night", "today", "yesterday", "tomorrow",
}

# Context phrases (multi-word) that usually describe conditions, not aspects.
CONTEXT_PHRASES = {"low light", "bright light", "direct sunlight", "high humidity", "peak hours"}

# Attribute heads: phrases like "sharp photos" are attributes of camera, not an aspect.
ATTRIBUTE_HEADS = {
    "photo", "photos", "picture", "pictures", "image", "images", "video", "videos",
    "sound", "sounds", "audio", "brightness", "color", "colors", "resolution",
}

UNIVERSAL_PHRASES = (
    "customer support", "support", "service", "delivery", "shipping",
    "refund", "return", "replacement", "warranty", "wifi", "staff", "room",
    "cleanliness", "food", "taste", "portion", "ambience", "location",
)

PENALIZED_PHRASE_WEIGHTS = {
    "daylight": 0.35,
    "sunlight": 0.35,
    "low light": 0.35,
    "bright light": 0.35,
    "full day": 0.25,
}


@dataclass(frozen=True)
class OpenAspectMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    predicted_count: int
    gold_count: int
    predicted_aspects: List[str]
    gold_aspects: List[str]


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"\s+", " ", text)


def _normalize_phrase(phrase: str) -> str:
    phrase = _clean_text(phrase).lower()
    phrase = TRAILING_PUNCT_RE.sub("", phrase)
    phrase = phrase.strip(" -_")
    return " ".join(phrase.split())


def _contains_mention(text_lower: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text_lower))


def _looks_like_time_quantity(phrase: str) -> bool:
    normalized = _normalize_phrase(phrase)
    tokens = normalized.split()
    if not tokens:
        return True

    if any(token in TIME_UNITS for token in tokens):
        qty_like = 0
        for token in tokens:
            if token.isdigit() or token in APPROX or token in TIME_UNITS or token in NUMBER_WORDS:
                qty_like += 1
        if qty_like >= max(2, len(tokens) - 1):
            return True

    return normalized in {"full day", "a full day"}


def _valid_phrase(phrase: str) -> bool:
    normalized = _normalize_phrase(phrase)
    if not normalized:
        return False

    tokens = normalized.split()
    if not tokens or len(tokens) > 6:
        return False

    if all(token in STOP for token in tokens):
        return False

    if _looks_like_time_quantity(normalized):
        return False

    if normalized in CONTEXT_PHRASES:
        return False

    if len(tokens) == 1:
        token = tokens[0]
        if token in STOP or token in CONTEXT_SINGLE:
            return False
        if token in GENERIC_SINGLE and token not in PRICE_WORDS:
            return False
        if len(token) < 4 and token not in SPECIAL_SHORT_TOKENS:
            return False

    if len(tokens) == 2 and tokens[-1] in ATTRIBUTE_HEADS:
        return False

    return True


@lru_cache(maxsize=1)
def _nlp() -> Language:
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.open_aspect_model_name, local_files_only=True)


def _span_text(span: Span) -> str:
    text = ARTICLE_PREFIX_RE.sub("", span.text.strip()).strip()
    return _normalize_phrase(text)


def _expand_noun_phrase_from_head(head: Token) -> str:
    left_modifiers = sorted((token for token in head.lefts if token.dep_ in LEFT_MODIFIER_DEPS), key=lambda token: token.i)
    parts = [token.text for token in left_modifiers]
    parts.append(head.text)
    return _normalize_phrase(" ".join(parts))


def _apply_canonical_preferences(text_lower: str, add: Callable[[str, float], None]) -> None:
    if "camera" in text_lower:
        add("camera", 0.95)

    battery_pattern = r"\bbattery\s+(life|lasts|lasting|drains|charging)\b"
    if "battery" in text_lower and ("battery life" in text_lower or re.search(battery_pattern, text_lower)):
        add("battery life", 0.90)


def _collect_candidates(doc: Doc) -> Tuple[List[str], Dict[str, float]]:
    candidates: List[str] = []
    scores: Dict[str, float] = {}

    def add(phrase: str, score: float) -> None:
        normalized = _normalize_phrase(phrase)
        if not _valid_phrase(normalized):
            return
        candidates.append(normalized)
        scores[normalized] = max(scores.get(normalized, 0.0), float(score))

    text_lower = doc.text.lower()

    for chunk in doc.noun_chunks:
        add(_span_text(chunk), 0.55)

    for token in doc:
        if token.pos_ in NOUN_POS:
            base = _expand_noun_phrase_from_head(token)
            add(base, 0.50)

            if any(child.dep_ == "amod" and child.pos_ == "ADJ" for child in token.children):
                add(base, 0.90)

            if token.dep_ in NOUN_HEAD_DEPS:
                add(base, 0.65)

        # "X quality" patterns are usually good aspect candidates.
        if token.lower_ == "quality" and token.pos_ == "NOUN":
            add(_expand_noun_phrase_from_head(token), 0.95)

    for phrase in UNIVERSAL_PHRASES:
        if _contains_mention(text_lower, phrase):
            add(phrase, 0.80)

    for phrase in PRICE_WORDS:
        if _contains_mention(text_lower, phrase):
            add(phrase, 0.75)

    _apply_canonical_preferences(text_lower, add)

    candidates = [phrase for phrase in candidates if ALPHA_RE.search(phrase)]
    return candidates, scores


def _candidate_penalty(phrase: str) -> float:
    if phrase in CONTEXT_PHRASES or phrase in CONTEXT_SINGLE:
        return 0.25
    return PENALIZED_PHRASE_WEIGHTS.get(phrase, 1.0)


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _dedup_by_embedding(phrases: List[str], scores: Dict[str, float], sim_thresh: float, max_keep: int) -> List[str]:
    if not phrases:
        return []

    counts = Counter(phrases)
    unique_phrases = list(counts.keys())

    def rank_key(phrase: str) -> Tuple[float, int, int, int, str]:
        tokens = phrase.split()
        base_score = scores.get(phrase, 0.0) * _candidate_penalty(phrase)
        return (-base_score, -counts[phrase], -min(len(tokens), 3), -len(phrase), phrase)

    ranked = sorted(unique_phrases, key=rank_key)
    embeddings = _embedder().encode(ranked, normalize_embeddings=True, show_progress_bar=False)

    kept: List[str] = []
    kept_indices: List[int] = []

    for index, phrase in enumerate(ranked):
        if not kept:
            kept.append(phrase)
            kept_indices.append(index)
            if len(kept) >= max_keep:
                break
            continue

        similarities = cosine_similarity([embeddings[index]], embeddings[kept_indices])[0]
        if float(similarities.max()) < sim_thresh:
            kept.append(phrase)
            kept_indices.append(index)
            if len(kept) >= max_keep:
                break

    return kept


def extract_open_aspects(review_text: str, max_aspects: int = 8) -> List[str]:
    text = _clean_text(review_text)
    if not text:
        return []

    doc = _nlp()(text)
    candidates, scores = _collect_candidates(doc)

    deduped_candidates = list(
        dict.fromkeys(_normalize_phrase(phrase) for phrase in candidates if phrase and _valid_phrase(phrase))
    )
    deduped = _dedup_by_embedding(
        deduped_candidates,
        scores=scores,
        sim_thresh=0.80,
        max_keep=max_aspects + 6,
    )

    final: List[str] = []
    for phrase in sorted(deduped, key=lambda value: (-len(value), value)):
        if any(phrase != existing and phrase in existing for existing in final):
            continue
        final.append(phrase)

    text_lower = text.lower()

    def appearance_index(phrase: str) -> int:
        index = text_lower.find(phrase)
        return index if index != -1 else 10**9

    final.sort(key=appearance_index)
    return final[:max_aspects]


def evaluate_open_aspects(
    review_text: str,
    gold_aspects: List[str],
    max_aspects: int = 8,
) -> OpenAspectMetrics:
    predicted_aspects = extract_open_aspects(review_text, max_aspects=max_aspects)

    predicted_set = set(predicted_aspects)
    gold_set = {
        normalized
        for aspect in gold_aspects
        if (normalized := _normalize_phrase(aspect))
    }

    true_positives = len(predicted_set & gold_set)
    precision = _safe_divide(true_positives, len(predicted_set))
    recall = _safe_divide(true_positives, len(gold_set))
    f1 = _f1_score(precision, recall)

    return OpenAspectMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        predicted_count=len(predicted_set),
        gold_count=len(gold_set),
        predicted_aspects=predicted_aspects,
        gold_aspects=sorted(gold_set),
    )
