# proto/backend/services/open_aspect.py
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Dict, Tuple

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


STOP = {
    "the","a","an","and","or","but","is","are","was","were","to","of","in","on","for","with","it","this","that",
    "during","very","really","just","too","also","so","as","at","from","by","be","been","being","i","we","you",
    "they","he","she","them","us","my","our","your","their",
    # wh/relative words that leak
    "which","what","who","whom","whose","where","when","why","how","there","here"
}

# generic single-token nouns that rarely help as aspects
GENERIC_SINGLE = {
    "thing","things","stuff","product","item","device","service","experience","time","people","person",
    # container nouns (often too generic alone)
    "phone","laptop","watch","smartwatch","hotel","restaurant","place"
}

# allow price/value terms (kept)
PRICE_WORDS = {"price","cost","value","pricing","refund","charge","charges","fee","fees"}

TIME_UNITS = {"day","days","week","weeks","month","months","year","years","hour","hours","minute","minutes"}
APPROX = {"about","around","almost","roughly","nearly","approximately"}

# context/conditions that are almost never "aspects"
CONTEXT_SINGLE = {
    "daylight","sunlight","indoors","outdoors","outside","inside","weather","room","area","place","location",
    "morning","evening","night","today","yesterday","tomorrow"
}

# context phrases (multi-word) that usually describe conditions, not aspects
CONTEXT_PHRASES = {
    "low light","bright light","direct sunlight","high humidity","peak hours"
}

# attribute heads: phrases like "sharp photos" are attributes of camera, not an aspect
ATTRIBUTE_HEADS = {
    "photo","photos","picture","pictures","image","images","video","videos",
    "sound","sounds","audio","brightness","color","colors","resolution"
}

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_phrase(p: str) -> str:
    p = _clean_text(p).lower()
    p = re.sub(r"[^\w\s\-]+$", "", p)
    p = p.strip(" -_")
    p = " ".join(p.split())
    return p


def _looks_like_time_quantity(phrase: str) -> bool:
    p = _normalize_phrase(phrase)
    toks = p.split()
    if not toks:
        return True

    # time units + mostly qty words -> reject
    if any(t in TIME_UNITS for t in toks):
        qty_like = 0
        for t in toks:
            if t.isdigit():
                qty_like += 1
            elif t in APPROX:
                qty_like += 1
            elif t in TIME_UNITS:
                qty_like += 1
            elif t in {"one","two","three","four","five","six","seven","eight","nine","ten"}:
                qty_like += 1
        if qty_like >= max(2, len(toks) - 1):
            return True

    # patterns like "a full day", "full day"
    if p in {"full day", "a full day"}:
        return True

    return False


def _valid_phrase(p: str) -> bool:
    p = _normalize_phrase(p)
    if not p:
        return False

    toks = p.split()
    if not toks:
        return False
    if len(toks) > 6:
        return False

    if all(t in STOP for t in toks):
        return False

    if _looks_like_time_quantity(p):
        return False

    # reject pure context phrases
    if p in CONTEXT_PHRASES:
        return False

    # single-token rules
    if len(toks) == 1:
        t = toks[0]
        if t in STOP:
            return False
        if t in CONTEXT_SINGLE:
            return False
        if t in GENERIC_SINGLE and t not in PRICE_WORDS:
            return False
        if len(t) < 4 and t not in {"5g","4g","gps","ram","app"}:
            return False

    # reject attribute phrases like "sharp photos" (photos/images/etc. as head)
    if len(toks) == 2 and toks[-1] in ATTRIBUTE_HEADS:
        return False

    return True


@lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def _embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _span_text(span) -> str:
    text = span.text
    text = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", text.strip(), flags=re.I).strip()
    return _normalize_phrase(text)


def _expand_noun_phrase_from_head(head: spacy.tokens.Token) -> str:
    parts = []
    left_mods = [t for t in head.lefts if t.dep_ in {"compound","amod","poss","nummod"}]
    left_mods = sorted(left_mods, key=lambda t: t.i)
    parts.extend([t.text for t in left_mods])
    parts.append(head.text)
    return _normalize_phrase(" ".join(parts))


def _collect_candidates(doc: spacy.tokens.Doc) -> Tuple[List[str], Dict[str, float]]:
    cands: List[str] = []
    scores: Dict[str, float] = {}

    def add(phrase: str, score: float):
        phrase = _normalize_phrase(phrase)
        if not _valid_phrase(phrase):
            return
        cands.append(phrase)
        scores[phrase] = max(scores.get(phrase, 0.0), float(score))

    tl = doc.text.lower()

    # 1) noun chunks
    for chunk in doc.noun_chunks:
        add(_span_text(chunk), 0.55)

    # 2) dependency-based: targets of opinions get higher score
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            base = _expand_noun_phrase_from_head(tok)
            add(base, 0.50)

            # noun with adjectival modifier: likely aspect target
            for child in tok.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    add(base, 0.90)

            # noun as subj/obj/attr: likely important
            if tok.dep_ in {"nsubj","dobj","pobj","attr","obj"}:
                add(base, 0.65)

        # “X quality” patterns
        if tok.lower_ == "quality" and tok.pos_ == "NOUN":
            add(_expand_noun_phrase_from_head(tok), 0.95)

    # 3) universal aspects: support/return/refund/delivery/wifi/etc. (only if mentioned)
    universal_phrases = [
        "customer support", "support", "service", "delivery", "shipping",
        "refund", "return", "replacement", "warranty", "wifi", "staff", "room",
        "cleanliness", "food", "taste", "portion", "ambience", "location"
    ]
    for up in universal_phrases:
        if re.search(rf"\b{re.escape(up)}\b", tl):
            add(up, 0.80)

    # 4) keep price/value terms
    for w in PRICE_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", tl):
            add(w, 0.75)

    # 5) canonicalization preferences (prevents wasting slots)
    # Prefer camera over conditions/attributes
    if "camera" in tl:
        add("camera", 0.95)
        # drop common camera-condition candidates later via penalty (handled in ranking below)
    # Prefer battery life if battery appears with lasts/life/charging
    if "battery" in tl and (("battery life" in tl) or re.search(r"\bbattery\s+(life|lasts|lasting|drains|charging)\b", tl)):
        add("battery life", 0.90)

    # final cleanup (only alpha-containing)
    cands = [p for p in cands if re.search(r"[a-zA-Z]", p)]
    return cands, scores


def _dedup_by_embedding(phrases: List[str], scores: Dict[str, float], sim_thresh: float, max_keep: int) -> List[str]:
    if not phrases:
        return []

    from collections import Counter
    cnt = Counter(phrases)
    uniq = list(cnt.keys())

    def penalty(p: str) -> float:
        # penalize context/condition phrases so they don't consume top-k
        if p in CONTEXT_PHRASES:
            return 0.25
        if p in CONTEXT_SINGLE:
            return 0.25
        if p in {"daylight","sunlight","low light","bright light"}:
            return 0.35
        if p in {"full day"}:
            return 0.25
        return 1.0

    def rank_key(x: str):
        toks = x.split()
        base = scores.get(x, 0.0) * penalty(x)
        return (-base, -cnt[x], -min(len(toks), 3), -len(x), x)

    ranked = sorted(uniq, key=rank_key)

    emb = _embedder().encode(ranked, normalize_embeddings=True, show_progress_bar=False)

    kept: List[str] = []
    kept_idx: List[int] = []

    for i, p in enumerate(ranked):
        if not kept:
            kept.append(p)
            kept_idx.append(i)
            if len(kept) >= max_keep:
                break
            continue

        sims = cosine_similarity([emb[i]], emb[kept_idx])[0]
        if float(sims.max()) < sim_thresh:
            kept.append(p)
            kept_idx.append(i)
            if len(kept) >= max_keep:
                break

    return kept


def extract_open_aspects(review_text: str, max_aspects: int = 8) -> List[str]:
    text = _clean_text(review_text)
    if not text:
        return []

    doc = _nlp()(text)
    cands, scores = _collect_candidates(doc)

    # dedup exact
    cands = list(dict.fromkeys([_normalize_phrase(p) for p in cands if p and _valid_phrase(p)]))

    # embed dedup
    deduped = _dedup_by_embedding(
        cands,
        scores=scores,
        sim_thresh=0.80,
        max_keep=max_aspects + 6,
    )

    # substring cleanup (keep longer)
    final: List[str] = []
    for p in sorted(deduped, key=lambda s: (-len(s), s)):
        if any(p != q and p in q for q in final):
            continue
        final.append(p)

    # order by appearance
    tl = text.lower()
    final.sort(key=lambda s: tl.find(s) if tl.find(s) != -1 else 10**9)

    return final[:max_aspects]