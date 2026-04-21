from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Iterable

try:
    from ..utils.utils import normalize_whitespace, split_sentences, tokenize
except (ImportError, ValueError):  # pragma: no cover
    from utils.utils import normalize_whitespace, split_sentences, tokenize


@dataclass
class CorefResult:
    text: str
    chains: list[dict]


def no_coref(text: str) -> CorefResult:
    return CorefResult(text=text, chains=[])


PRONOUN_RE = re.compile(r"\b(it|they|them|this|that|these|those|he|she|him|her|its|their)\b", re.IGNORECASE)
ANTECEDENT_RE = re.compile(r"\b(?:the|a|an)?\s*([A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,2})\b")
STOP_ANTECEDENT_TOKENS = {
    "and", "or", "but", "so", "because", "with", "without", "very", "really", "too", "not",
    "great", "good", "bad", "nice", "okay", "fine", "all",
}
ANTECEDENT_BOUNDARY_TOKENS = {
    "is", "are", "was", "were", "be", "been", "being", "feels", "felt", "seems",
    "looks", "lasts", "charges", "works",
}


def _clean_antecedent_candidate(candidate: str) -> str | None:
    tokens = tokenize(candidate)
    if not tokens:
        return None
    if any(PRONOUN_RE.fullmatch(token) for token in tokens):
        return None
    while tokens and tokens[0] in {"the", "a", "an"}:
        tokens = tokens[1:]
    if tokens and tokens[0] in STOP_ANTECEDENT_TOKENS:
        return None
    for index, token in enumerate(tokens):
        if token in ANTECEDENT_BOUNDARY_TOKENS:
            tokens = tokens[:index]
            break
    if not tokens or all(token in STOP_ANTECEDENT_TOKENS for token in tokens):
        return None
    if any(token.isdigit() for token in tokens):
        return None
    return normalize_whitespace(" ".join(tokens))


def _pick_antecedent(sentence: str, previous: str | None) -> str | None:
    matches = [match.group(1).strip() for match in ANTECEDENT_RE.finditer(sentence)]
    for candidate in reversed(matches):
        cleaned = _clean_antecedent_candidate(candidate)
        if cleaned:
            return cleaned
    return previous


def _rewrite_sentence(sentence: str, antecedent: str | None, chains: list[dict]) -> str:
    if not antecedent:
        return sentence

    def replace(match: re.Match[str]) -> str:
        pronoun = match.group(0)
        chains.append({"pronoun": pronoun, "antecedent": antecedent})
        return antecedent

    return PRONOUN_RE.sub(replace, sentence)


def heuristic_coref(text: str) -> CorefResult:
    clean = normalize_whitespace(text)
    if not clean:
        return CorefResult(text="", chains=[])

    sentences = split_sentences(clean)
    chains: list[dict] = []
    rewritten: list[str] = []
    antecedent: str | None = None
    for sentence in sentences:
        contains_pronoun = bool(PRONOUN_RE.search(sentence))
        rewritten_sentence = _rewrite_sentence(sentence, antecedent, chains)
        rewritten.append(rewritten_sentence)
        if not contains_pronoun:
            antecedent = _pick_antecedent(sentence, antecedent) or antecedent
    return CorefResult(text=" ".join(rewritten), chains=chains)
