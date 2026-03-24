from __future__ import annotations

import re
from typing import Dict

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "textcat"])
except Exception:
    nlp = None

def _expand_short_evidence(text: str, sentence: str, sent_start: int) -> str:
    if not text or not sentence or sent_start < 0:
        return sentence
    if len(sentence.split()) > 3:
        return sentence
    start = sent_start
    end = min(len(text), sent_start + len(sentence))

    while start > 0 and text[start - 1] not in ".!?\n":
        start -= 1
    while end < len(text) and text[end] not in ".!?\n":
        end += 1

    clause = text[start:end].strip(" ,;:-")
    if not clause:
        return sentence

    parts = [part.strip(" ,;:-") for part in re.split(r"\b(?:but|and|while|though|although|however)\b|[,;:]", clause, flags=re.IGNORECASE) if part.strip(" ,;:-")]
    low_sentence = sentence.lower()
    for part in parts:
        if low_sentence in part.lower():
            return part
    return clause

def _get_dependency_span(sentence: str, aspect_phrase: str) -> str:
    if not nlp or not aspect_phrase:
        return sentence
    
    doc = nlp(sentence)
    aspect_words = aspect_phrase.lower().split()
    if not aspect_words:
        return sentence
        
    core_word = aspect_words[-1]
    target_token = None
    for token in doc:
        if core_word in token.text.lower():
            target_token = token
            break
            
    if not target_token:
        # Fallback to strict clause boundary
        parts = [p.strip() for p in re.split(r"\b(?:but|and|while|though|although|however)\b|[,;:]", sentence, flags=re.IGNORECASE)]
        for p in parts:
            if aspect_phrase.lower() in p.lower() and len(p.split()) >= 3:
                return p
        return sentence
        
    # Get subtree
    subtree_tokens = list(target_token.subtree)
    if not subtree_tokens:
        return sentence
        
    start_idx = subtree_tokens[0].i
    end_idx = subtree_tokens[-1].i
    
    # Expand slightly if it's too truncated
    if start_idx > 0 and doc[start_idx-1].text.lower() in {"very", "too", "not", "so", "really", "extremely"}:
        start_idx -= 1
        
    span = doc[start_idx:end_idx+1].text.strip(' ,;:-')
    if len(span.split()) >= 3:
        return span
    return sentence

def extract_evidence(text: str, evidence_sentence: str, aspect_raw: str, min_chars: int = 4, max_chars: int = 100) -> Dict:
    sentence = (evidence_sentence or text or "").strip()
    if not sentence:
        return {"evidence_text": "", "evidence_sentence": "", "char_start": None, "char_end": None, "evidence_quality": 0.0, "is_sentence_fallback": True}

    low_text = text.lower()
    low_sent = sentence.lower()
    sent_start = low_text.find(low_sent)
    if sent_start < 0:
        sent_start = 0
    else:
        sentence = _expand_short_evidence(text, sentence, sent_start)
        low_sent = sentence.lower()
        sent_start = low_text.find(low_sent)
        if sent_start < 0:
            sent_start = 0

    raw = str(aspect_raw or "").strip().lower().replace("_", " ")
    phrase = ""
    
    # 1. Try Dependency extraction first if it's a standard sentence
    if len(sentence.split()) > 4:
        dep_span = _get_dependency_span(sentence, raw)
        if dep_span and len(dep_span) >= min_chars and dep_span.lower() != low_sent and dep_span.lower() in low_sent:
            phrase = dep_span
            
    # 2. Regex window fallback if dependency failed
    if not phrase and raw:
        parts = [p.strip() for p in raw.split() if p.strip()]
        if parts:
            pattern = r"([^.?!]{0,30}\b" + r"\b[^.?!]{0,18}\b".join(re.escape(p) for p in parts[:3]) + r"[^.?!]{0,30})"
            m = re.search(pattern, low_sent)
            if m:
                extracted = sentence[m.start(1):m.end(1)].strip(" ,;:-")
                # Clean up extracted boundaries by clause
                clause_parts = [cp.strip() for cp in re.split(r"\b(?:but|however|although|though)\b|;", extracted, flags=re.IGNORECASE) if cp.strip()]
                for cp in clause_parts:
                    if any(p in cp.lower() for p in parts):
                        phrase = cp
                        break
                if not phrase:
                    phrase = extracted

    # 3. Last fallback bounding
    if not phrase:
        idx = low_text.find(low_sent)
        if idx >= 0:
            return {
                "evidence_text": sentence,
                "evidence_sentence": sentence,
                "char_start": idx,
                "char_end": idx + len(sentence),
                "evidence_quality": 0.55,
                "is_sentence_fallback": True,
            }
        return {
            "evidence_text": text[:max_chars],  # NEVER return full review
            "evidence_sentence": sentence,
            "char_start": 0,
            "char_end": min(len(text), max_chars),
            "evidence_quality": 0.15,
            "is_sentence_fallback": True,
        }

    phrase = phrase[:max_chars].strip()
    idx = low_text.find(phrase.lower()) if phrase else -1

    if idx < 0 or len(phrase) < min_chars:
        idx2 = low_text.find(low_sent)
        if idx2 >= 0:
            end = idx2 + len(sentence)
            return {
                "evidence_text": sentence,
                "evidence_sentence": sentence,
                "char_start": idx2,
                "char_end": end,
                "evidence_quality": 0.55,
                "is_sentence_fallback": True,
            }
        return {
            "evidence_text": text[:max_chars], # NEVER return full review
            "evidence_sentence": sentence,
            "char_start": 0,
            "char_end": min(len(text), max_chars),
            "evidence_quality": 0.15,
            "is_sentence_fallback": True,
        }

    return {
        "evidence_text": phrase,
        "evidence_sentence": sentence,
        "char_start": idx,
        "char_end": idx + len(phrase),
        "evidence_quality": 0.95,
        "is_sentence_fallback": False,
    }
