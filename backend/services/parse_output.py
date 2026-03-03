# proto/backend/services/parse_output.py
import json
import re
from typing import List, Tuple

VALID = {"positive", "neutral", "negative"}

SENT_MAP = {
    "pos": "positive",
    "positive": "positive",
    "good": "positive",
    "great": "positive",
    "excellent": "positive",
    "neg": "negative",
    "negative": "negative",
    "bad": "negative",
    "poor": "negative",
    "terrible": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "mixed": "neutral",
}

STOP = {
    "the","a","an","and","or","but","is","are","was","were","to","of","in","on","for","with","it","this","that",
    "during","very","really","just","too","also","so","as","at","from","by","be","been","being"
}

JUNK_SINGLE = {"excellent", "good", "bad", "great", "phone", "product", "service", "delivery"}


def _norm_sent(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    return SENT_MAP.get(s, s)


def _norm_aspect(a: str) -> str:
    a = (a or "").strip()
    a = re.sub(r"\s+", " ", a)
    a = a.strip(" -•*:\t\"'")
    return a[:255]


def _is_good_aspect(aspect: str) -> bool:
    if not aspect:
        return False
    toks = [t for t in aspect.lower().split() if t]
    if len(toks) == 1:
        if toks[0] in STOP or toks[0] in JUNK_SINGLE:
            return False
    non_stop = [t for t in toks if t not in STOP]
    if len(non_stop) == 0:
        return False
    if len(toks) > 8:
        return False
    return True


def _parse_mapping_style(text: str) -> List[Tuple[str, str]]:
    """
    Parses outputs like:
      "speed": "positive"
      "crashes": "negative"
    or:
      "speed": "positive", "crashes": "negative"
    """
    pairs: List[Tuple[str, str]] = []
    seen = set()

    # Find all "key": "value" pairs
    for m in re.finditer(r"\"([^\"]+)\"\s*:\s*\"([^\"]+)\"", text):
        aspect = _norm_aspect(m.group(1))
        sent = _norm_sent(m.group(2))
        if sent not in VALID:
            continue
        if not _is_good_aspect(aspect):
            continue
        key = (aspect.lower(), sent)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((aspect, sent))
    return pairs


def parse_lines(gen_text: str) -> List[Tuple[str, str]]:
    text = (gen_text or "").strip()
    if not text:
        return []

    # If it looks like mapping style, parse it
    if ":" in text:
        mapped = _parse_mapping_style(text)
        if mapped:
            return mapped
        mapped2 = _parse_mapping_style_unquoted(text)
        if mapped2:
            return mapped2

    # Try JSON parse: {"pairs":[{"aspect":"...","sentiment":"..."}]}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("pairs"), list):
            out = []
            seen = set()
            for it in obj["pairs"]:
                if not isinstance(it, dict):
                    continue
                aspect = _norm_aspect(it.get("aspect", ""))
                sent = _norm_sent(it.get("sentiment", ""))
                if sent not in VALID:
                    continue
                if not _is_good_aspect(aspect):
                    continue
                key = (aspect.lower(), sent)
                if key in seen:
                    continue
                seen.add(key)
                out.append((aspect, sent))
            return out
    except Exception:
        pass

    # Line parsing fallback
    pairs: List[Tuple[str, str]] = []
    seen = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"^[\-\*\u2022\d\)\.\s]+", "", line).strip()

        if "|" in line:
            left, right = line.split("|", 1)
        elif ":" in line:
            left, right = line.split(":", 1)
        elif "-" in line:
            left, right = line.split("-", 1)
        else:
            continue

        aspect = _norm_aspect(left)
        sent = _norm_sent(right)

        if sent not in VALID:
            continue
        if not _is_good_aspect(aspect):
            continue

        key = (aspect.lower(), sent)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((aspect, sent))

    return pairs


def heuristic_confidence(sentiment: str) -> float:
    return 0.75 if sentiment in {"positive", "negative"} else 0.65

def _parse_mapping_style_unquoted(text: str) -> List[Tuple[str, str]]:
    """
    Parses outputs like:
      battery: positive, heating: negative
      battery life : positive
    """
    pairs: List[Tuple[str, str]] = []
    seen = set()

    # Split by commas first
    chunks = [c.strip() for c in re.split(r"\s*,\s*", text) if c.strip()]
    for ch in chunks:
        if ":" not in ch:
            continue
        left, right = ch.split(":", 1)
        aspect = _norm_aspect(left)
        sent = _norm_sent(right)
        if sent not in VALID:
            continue
        if not _is_good_aspect(aspect):
            continue
        key = (aspect.lower(), sent)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((aspect, sent))

    return pairs