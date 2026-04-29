# proto/backend/services/evidence.py
import re
from typing import List, Tuple


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start, end, sentence_text) with character offsets.
    Very lightweight sentence splitter (MVP).
    """
    sents = []
    # Split on sentence punctuation and contrastive clause separators.
    parts = re.split(r"(?<=[.!?;])\s+|,\s+(?=(?:but|though|however|yet)\b)", text, flags=re.I)
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


def find_evidence_for_explicit_candidate(
    review_text: str,
    aspect: str,
    aliases: tuple[str, ...] = (),
) -> Tuple[int, int, str, float, bool]:
    """
    Strict explicit evidence binding.
    Returns empty evidence when neither aspect nor aliases are present.
    """
    sents = split_sentences(review_text or "")
    if not sents:
        return (0, 0, "", 0.0, False)

    terms = [str(aspect or "").strip().lower()]
    terms.extend(str(a or "").strip().lower() for a in aliases)
    terms = [t for t in terms if t]
    if not terms:
        return (0, 0, "", 0.0, False)

    for s, e, sent in sents:
        sent_l = sent.lower()
        if any(term in sent_l for term in terms):
            score = 1.0 if terms[0] in sent_l else 0.85
            return (s, e, sent, score, True)

    return (0, 0, "", 0.0, False)
