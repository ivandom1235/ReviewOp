from __future__ import annotations
from typing import Any
from .spacy_pipeline import parse_review, load_spacy

def extract_noun_chunks(text: str) -> list[dict[str, Any]]:
    """Extract noun chunks from text using spaCy."""
    nlp = load_spacy()
    doc = parse_review(text, nlp)
    
    chunks = []
    for chunk in doc.noun_chunks:
        # Simple cleaning: drop pronouns and very short chunks
        if chunk.root.pos_ == "PRON":
            continue
        
        # Strip leading determiners (e.g. "The quality" -> "quality")
        text_val = chunk.text
        start_char = chunk.start_char
        if chunk[0].pos_ == "DET":
            if len(chunk) > 1:
                start_char = chunk[1].idx
                text_val = chunk[1:].text
            
        chunks.append({
            "text": text_val.strip(),
            "span": (start_char, chunk.end_char),
            "aspect_anchor": chunk.root.text.lower(),
            "anchor_source": "noun_chunk_root",
            "modifier_terms": tuple(t.text.lower() for t in chunk if t.pos_ in ("ADJ", "VERB") and t != chunk.root)
        })
    return chunks

def extract_dependency_phrases(text: str) -> list[dict[str, Any]]:
    """Extract phrases based on dependency modifiers (e.g. ADJ + NOUN)."""
    nlp = load_spacy()
    doc = parse_review(text, nlp)
    
    phrases = []
    for token in doc:
        # Rule 1: Attributive Adjective modifying a Noun (e.g. "amazing quality")
        if token.pos_ in ("NOUN", "PROPN"):
            for child in token.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    # Get the raw text from the document slice
                    start = min(child.idx, token.idx)
                    end = max(child.idx + len(child.text), token.idx + len(token.text))
                    phrase_text = doc.text[start:end]
                    phrases.append({
                        "text": phrase_text,
                        "span": (start, end),
                        "type": "adj_noun",
                        "aspect_anchor": token.text.lower(),
                        "modifier_terms": (child.text.lower(),),
                        "anchor_source": "adj_noun"
                    })
        
        # Rule 2: Predicative Adjective (e.g. "quality is amazing")
        if token.pos_ in ("ADJ", "NOUN") and token.dep_ in ("acomp", "attr"):
            head = token.head
            if head.pos_ in ("AUX", "VERB") or head.lemma_ == "be":
                for child in head.children:
                    if child.dep_ == "nsubj":
                        # Anchor is the subject root
                        anchor_text = child.text.lower()
                        subj_start, subj_end = child.idx, child.idx + len(child.text)
                        
                        for chunk in doc.noun_chunks:
                            if child in chunk:
                                anchor_text = chunk.root.text.lower()
                                subj_start, subj_end = chunk.start_char, chunk.end_char
                                break
                        
                        start = min(subj_start, token.idx)
                        end = max(subj_end, token.idx + len(token.text))
                        phrase_text = doc.text[start:end]
                        phrases.append({
                            "text": phrase_text,
                            "span": (start, end),
                            "type": "nsubj_acomp",
                            "aspect_anchor": anchor_text,
                            "modifier_terms": (token.text.lower(),),
                            "anchor_source": "nsubj_acomp"
                        })

        # Rule 3: Verb with a direct object (e.g. "improved quality")
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "dobj":
                    start = min(token.idx, child.idx)
                    end = max(token.idx + len(token.text), child.idx + len(child.text))
                    phrase_text = doc.text[start:end]
                    phrases.append({
                        "text": phrase_text,
                        "span": (start, end),
                        "type": "verb_dobj",
                        "aspect_anchor": child.text.lower(),
                        "modifier_terms": (token.text.lower(),),
                        "anchor_source": "verb_dobj"
                    })
                    
    return phrases

def extract_modifier_heads(text: str) -> list[dict[str, Any]]:
    """Legacy/Fallback for modifier-head pairs."""
    return extract_dependency_phrases(text)
