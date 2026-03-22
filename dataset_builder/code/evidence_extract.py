from __future__ import annotations

import re
from typing import Dict


def extract_evidence(text: str, evidence_sentence: str, aspect_raw: str, min_chars: int = 4, max_chars: int = 80) -> Dict:
    sentence = (evidence_sentence or text or "").strip()
    if not sentence:
        return {"evidence_text": "", "evidence_sentence": "", "char_start": None, "char_end": None, "evidence_quality": 0.0, "is_sentence_fallback": True}

    low_text = text.lower()
    low_sent = sentence.lower()
    sent_start = low_text.find(low_sent)
    if sent_start < 0:
        sent_start = 0

    raw = str(aspect_raw or "").strip().lower().replace("_", " ")
    phrase = ""
    if raw:
        parts = [p.strip() for p in raw.split() if p.strip()]
        if parts:
            pattern = r"([^.?!]{0,18}\b" + r"\b[^.?!]{0,18}\b".join(re.escape(p) for p in parts[:3]) + r"[^.?!]{0,18})"
            m = re.search(pattern, low_sent)
            if m:
                phrase = sentence[m.start(1):m.end(1)].strip(" ,;:-")
        if not phrase:
            for token in parts[:3]:
                if len(token) >= 4:
                    idx = low_sent.find(token)
                    if idx >= 0:
                        start = max(0, idx - 18)
                        end = min(len(sentence), idx + len(token) + 18)
                        phrase = sentence[start:end].strip(" ,;:-")
                        break

    if not phrase:
        return {
            "evidence_text": sentence,
            "evidence_sentence": sentence,
            "char_start": sent_start if sent_start >= 0 else None,
            "char_end": (sent_start + len(sentence)) if sent_start >= 0 else None,
            "evidence_quality": 0.35,
            "is_sentence_fallback": True,
        }

    phrase = phrase[:max_chars].strip()
    idx = low_text.find(phrase.lower()) if phrase else -1

    if idx < 0 or len(phrase) < min_chars:
        idx = sent_start if sent_start >= 0 else None
        end = (idx + len(sentence)) if idx is not None else None
        return {
            "evidence_text": sentence,
            "evidence_sentence": sentence,
            "char_start": idx,
            "char_end": end,
            "evidence_quality": 0.55,
            "is_sentence_fallback": True,
        }

    return {
        "evidence_text": phrase,
        "evidence_sentence": sentence,
        "char_start": idx,
        "char_end": idx + len(phrase),
        "evidence_quality": 0.93,
        "is_sentence_fallback": False,
    }
