from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Iterable

from utils import normalize_whitespace, split_sentences, tokenize


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
    "great", "good", "bad", "nice", "okay", "okay", "fine",
}


def _pick_antecedent(sentence: str, previous: str | None) -> str | None:
    matches = [match.group(1).strip() for match in ANTECEDENT_RE.finditer(sentence)]
    for candidate in reversed(matches):
        tokens = tokenize(candidate)
        if not tokens:
            continue
        if all(token in STOP_ANTECEDENT_TOKENS for token in tokens):
            continue
        if any(token.isdigit() for token in tokens):
            continue
        return normalize_whitespace(candidate)
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
        current_antecedent = _pick_antecedent(sentence, antecedent)
        rewritten_sentence = _rewrite_sentence(sentence, current_antecedent, chains)
        rewritten.append(rewritten_sentence)
        antecedent = current_antecedent or antecedent
    return CorefResult(text=" ".join(rewritten), chains=chains)
