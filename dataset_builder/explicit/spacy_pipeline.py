from __future__ import annotations
from functools import lru_cache

import spacy
from spacy.language import Language

@lru_cache(maxsize=4)
def load_spacy(model_name: str = "en_core_web_sm") -> Language:
    try:
        return spacy.load(model_name)
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

def parse_review(text: str, nlp: Language | None = None) -> spacy.tokens.Doc:
    return (nlp or load_spacy())(text or "")
