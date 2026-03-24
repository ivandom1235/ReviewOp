from __future__ import annotations
from typing import List, Dict, Any
import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    pass  # Allow graceful degradation or manual download

OPINION_WORDS = {"good", "bad", "terrible", "great", "excellent", "poor", 
                 "slow", "fast", "beautiful", "ugly", "amazing", "awful", 
                 "horrible", "fantastic", "decent", "mediocre", "stellar", "abysmal"}

def _get_opinion_candidates(doc) -> List[Dict[str, Any]]:
    candidates = []
    seen = set()
    for token in doc:
        if token.lemma_.lower() in OPINION_WORDS or token.text.lower() in OPINION_WORDS:
            # Pattern: ADJ + NOUN (e.g., "good battery")
            if token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PROPN"]:
                noun = token.head.text.lower()
                if noun not in seen:
                    candidates.append({
                        "candidate_text": noun,
                        "evidence_span": token.head.sent.text.strip(),
                        "extraction_method": "opinion_pattern",
                        "confidence": 0.85,
                        "sentiment": "positive" if token.lemma_.lower() in {"good", "great", "excellent", "amazing", "fantastic", "stellar"} else "negative"
                    })
                    seen.add(noun)
            
            # Pattern: NOUN + is + ADJ (e.g., "battery is good")
            elif token.dep_ == "acomp" and token.head.lemma_ == "be":
                for child in token.head.children:
                    if child.dep_ == "nsubj" and child.pos_ in ["NOUN", "PROPN"]:
                        noun = child.text.lower()
                        if noun not in seen:
                            candidates.append({
                                "candidate_text": noun,
                                "evidence_span": child.sent.text.strip(),
                                "extraction_method": "opinion_pattern",
                                "confidence": 0.85,
                                "sentiment": "positive" if token.lemma_.lower() in {"good", "great", "excellent", "amazing", "fantastic", "stellar"} else "negative"
                            })
                            seen.add(noun)
    return candidates

def _get_noun_chunks(doc) -> List[Dict[str, Any]]:
    candidates = []
    for chunk in doc.noun_chunks:
        text = chunk.text.lower().strip()
        # Keep it short, usually an aspect is 1-3 words
        if 0 < len(text.split()) <= 3 and chunk.root.pos_ not in ["PRON", "DET"]:
            candidates.append({
                "candidate_text": text,
                "evidence_span": chunk.sent.text.strip(),
                "extraction_method": "noun_chunk",
                "confidence": 0.60,
                "sentiment": "neutral"
            })
    return candidates

def validate_with_llm(candidate: Dict, review_text: str, llm_client: Any) -> Dict:
    if not llm_client:
        return candidate
        
    prompt = f"""Review: "{review_text}"
Candidate explicit aspect: "{candidate['candidate_text']}"
Is this a genuine product or service aspect mentioned in the review?
Return strictly JSON with keys "is_aspect" (boolean), "sentiment" (string: positive/negative/neutral), "normalized" (string)."""
    
    try:
        resp = llm_client.json_completion(prompt)
        if isinstance(resp, dict) and resp.get("is_aspect"):
            candidate["confidence"] = 0.90
            candidate["sentiment"] = resp.get("sentiment", "neutral")
            candidate["normalized_phrase"] = resp.get("normalized", candidate["candidate_text"])
        else:
            candidate["confidence"] = 0.0  # Rejected by LLM
    except Exception:
        pass
    return candidate

def normalize_aspect_phrase(phrase: str) -> str:
    from mappings import CANONICAL_ASPECTS
    phrase = phrase.lower().strip()
    for domain, aspects in CANONICAL_ASPECTS.items():
        for canonical, aliases in aspects.items():
            if phrase in aliases or phrase == canonical:
                return canonical
    return phrase

def extract_explicit_aspects(text: str, llm_client: Any = None) -> List[Dict[str, Any]]:
    """Stage 4 Pipeline: Candidate Generation + LLM Validation + Normalization"""
    if not text.strip(): return []
    
    try:
        doc = nlp(text)
    except NameError:
        return []

    candidates = _get_opinion_candidates(doc)
    
    seen_texts = {c["candidate_text"] for c in candidates}
    for nc in _get_noun_chunks(doc):
        if nc["candidate_text"] not in seen_texts:
            candidates.append(nc)
            seen_texts.add(nc["candidate_text"])
            
    final_aspects = []
    for c in candidates:
        if c["confidence"] < 0.75 and llm_client:
            c = validate_with_llm(c, text, llm_client)
            
        if c["confidence"] >= 0.70:
            if "normalized_phrase" not in c:
                c["normalized_phrase"] = normalize_aspect_phrase(c["candidate_text"])
                
            final_aspects.append({
                "aspect_phrase": c["candidate_text"],
                "normalized_phrase": c["normalized_phrase"],
                "sentiment": c.get("sentiment", "neutral"),
                "evidence_span": c["evidence_span"],
                "confidence": c["confidence"],
                "extraction_method": c["extraction_method"]
            })
            
    return final_aspects
