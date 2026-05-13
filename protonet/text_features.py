from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

VAGUE_PATTERNS = [
    re.compile(r"\bsomething felt off\b", re.I),
    re.compile(r"\bnot what i expected\b", re.I),
    re.compile(r"\bcould have been better\b", re.I),
    re.compile(r"\bnot sure why\b", re.I),
    re.compile(r"\bcannot point to\b", re.I),
    re.compile(r"\bcan't point to\b", re.I),
    re.compile(r"\bexpected more\b", re.I),
    re.compile(r"\boverall disappointing\b", re.I),
]


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def lexical_overlap(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(1, min(len(ta), len(tb)))


def evidence_support(text: str, aspect: str, matched_terms: list[str] | None = None) -> float:
    support = lexical_overlap(text, aspect)
    for term in matched_terms or []:
        if term and term.lower() in (text or "").lower():
            support = max(support, 0.90)
        else:
            support = max(support, lexical_overlap(text, term) * 0.80)
    # phrase contains canonical family name
    aspect_tokens = token_set(aspect.replace("_", " "))
    text_tokens = token_set(text)
    if aspect_tokens and aspect_tokens <= text_tokens:
        support = max(support, 1.0)
    return float(min(1.0, support))


def looks_vague(text: str) -> bool:
    return any(p.search(text or "") for p in VAGUE_PATTERNS)


def text_signature(text: str, max_tokens: int = 40) -> str:
    c = Counter(tokens(text))
    return " ".join(t for t, _ in c.most_common(max_tokens))


def open_world_evidence_quality(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0

    toks = [t for t in text.split() if t.strip()]
    if len(toks) < 3:
        return 0.20

    vague_terms = {"something", "thing", "stuff", "issue", "problem", "bad", "good"}
    vague_count = sum(1 for t in toks if t.lower().strip(".,!?") in vague_terms)

    quality = min(1.0, len(toks) / 12.0)
    quality -= min(0.4, vague_count * 0.15)

    return max(0.0, quality)
