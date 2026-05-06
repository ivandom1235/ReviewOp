from __future__ import annotations

import re

from ..explicit.spacy_pipeline import load_spacy


def _normalize_token(token: str) -> str:
    return re.sub(r"^\W+|\W+$", "", str(token or "").lower())


def _narrow_phrase_window(text: str, cue: str, window_tokens: int = 4) -> str:
    text = str(text or "").strip()
    cue = str(cue or "").strip().lower()
    if not text or not cue:
        return text
    words = text.split()
    cue_tokens = cue.split()
    if not words or not cue_tokens:
        return text
    low = [_normalize_token(word) for word in words]
    cue_tokens = [_normalize_token(token) for token in cue_tokens]
    cue_tokens = [token for token in cue_tokens if token]
    if not cue_tokens:
        return text
    for start_idx in range(0, len(low) - len(cue_tokens) + 1):
        if low[start_idx : start_idx + len(cue_tokens)] == cue_tokens:
            left = max(0, start_idx - max(0, int(window_tokens)))
            right = min(len(words), start_idx + len(cue_tokens) + max(0, int(window_tokens)))
            snippet = " ".join(words[left:right]).strip()
            if snippet and snippet != text:
                return snippet
    return text

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy."""
    import spacy
    nlp = load_spacy()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def select_best_sentence(text: str, cue: str = "") -> str:
    """Select the best sentence containing the cue, or the first one."""
    sentences = split_sentences(text)
    if not sentences:
        return str(text or "").strip()

    cue = cue.lower().strip()
    if cue and cue in str(text or "").lower():
        narrowed = _narrow_phrase_window(text, cue)
        if narrowed != str(text or "").strip():
            return narrowed

    if cue:
        # 1. Exact match
        for sentence in sentences:
            if cue in sentence.lower():
                return sentence
        
        # 2. Token overlap (fallback)
        cue_tokens = set(cue.split())
        best_sent = sentences[0]
        max_overlap = 0
        for sentence in sentences:
            sent_tokens = set(sentence.lower().split())
            overlap = len(cue_tokens & sent_tokens)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sent = sentence
        return best_sent
        
    return sentences[0]

def validate_evidence_span(text: str, span: tuple[int, int]) -> bool:
    """Check if a span is valid within the text length."""
    start, end = span
    return 0 <= start < end <= len(text)
