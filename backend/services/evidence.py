# proto/backend/services/evidence.py
import re
from typing import List, Tuple


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start, end, sentence_text) with character offsets.
    Very lightweight sentence splitter (MVP).
    """
    sents = []
    # Split on punctuation followed by whitespace/newline
    parts = re.split(r"(?<=[.!?])\s+", text)
    cursor = 0
    for p in parts:
        if not p:
            continue
        # find this part in text starting at cursor
        idx = text.find(p, cursor)
        if idx == -1:
            # fallback approximate
            idx = cursor
        start = idx
        end = idx + len(p)
        sents.append((start, end, p.strip()))
        cursor = end
    return sents


def find_evidence_for_aspect(review_text: str, aspect: str) -> Tuple[int, int, str]:
    """
    Choose a best sentence containing the aspect (case-insensitive substring).
    If none found, return first sentence.
    """
    aspect_l = aspect.lower()
    sents = split_sentences(review_text)
    if not sents:
        return (0, min(len(review_text), 1), review_text[:200])

    best = None
    for (s, e, sent) in sents:
        if aspect_l in sent.lower():
            best = (s, e, sent)
            break

    if best is None:
        best = sents[0]

    s, e, sent = best
    snippet = sent
    return (s, e, snippet)