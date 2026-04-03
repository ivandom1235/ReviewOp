from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

from utils import normalize_whitespace, tokenize


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

LANGUAGE_HINTS: dict[str, set[str]] = {
    "es": {" el ", " la ", " los ", " las ", " muy ", " pero ", " servicio ", " pantalla ", " comida ", " precio "},
    "fr": {" le ", " la ", " les ", " tres ", " mais ", " service ", " ecran ", " prix ", " nourriture "},
    "de": {" der ", " die ", " das ", " sehr ", " aber ", " dienst ", " bildschirm ", " preis "},
    "pt": {" o ", " a ", " os ", " as ", " muito ", " mas ", " servico ", " tela ", " preco "},
    "it": {" il ", " la ", " gli ", " molto ", " ma ", " servizio ", " schermo ", " prezzo "},
    "nl": {" de ", " het ", " en ", " heel ", " maar ", " service ", " scherm ", " prijs "},
    "hi": {" है ", " और ", " बहुत ", " लेकिन ", " सेवा ", " स्क्रीन ", " कीमत "},
    "ru": {" и ", " но ", " очень ", " экран ", " цена ", " сервис "},
    "ar": {" و ", " لكن ", " جدا ", " خدمة ", " سعر ", " شاشة "},
    "zh": {"服务", "屏幕", "价格", "很", "但是", "食物"},
}


def _normalize_for_hint_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return f" {normalize_whitespace(stripped).lower()} "


def detect_language(text: str) -> str:
    clean = normalize_whitespace(text)
    if not clean:
        return "unknown"
    if DEVANAGARI_RE.search(clean):
        return "hi"
    if CJK_RE.search(clean):
        return "zh"
    if ARABIC_RE.search(clean):
        return "ar"
    if CYRILLIC_RE.search(clean):
        return "ru"

    lowered = _normalize_for_hint_matching(clean)
    scores: Counter[str] = Counter()
    for language, hints in LANGUAGE_HINTS.items():
        for hint in hints:
            if hint in lowered:
                scores[language] += 1

    tokens = tokenize(clean)
    if not tokens:
        return "unknown"
    if scores:
        ordered = scores.most_common(2)
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            return "mixed"
        if ordered[0][1] >= 2:
            return ordered[0][0]

    normalized_tokens = _normalize_for_hint_matching(clean).split()
    if any(token in {"y", "pero", "muy", "precio", "servicio", "pantalla"} for token in normalized_tokens):
        return "es"
    if any(token in {"et", "mais", "tres", "prix", "service", "ecran"} for token in normalized_tokens):
        return "fr"
    return "en"


def language_distribution(texts: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts[detect_language(text)] += 1
    return dict(counts)


def is_implicit_ready(
    text: str,
    *,
    language: str,
    min_tokens: int,
    supported_languages: Iterable[str],
) -> bool:
    if token_count := len(tokenize(text)):
        if token_count < min_tokens:
            return False
    else:
        return False
    supported = {str(item).lower() for item in supported_languages}
    if language == "mixed":
        return True
    if language == "unknown":
        return False
    return language.lower() in supported
