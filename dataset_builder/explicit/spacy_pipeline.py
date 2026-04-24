from __future__ import annotations
import spacy
from spacy.language import Language
from functools import lru_cache

@lru_cache(maxsize=1)
def load_spacy(model_name: str = "en_core_web_sm") -> Language:
    """Load and return a spaCy model, cached to prevent redundant loading."""
    try:
        return spacy.load(model_name)
    except OSError:
        # Fallback to downloading if not found (though we should have it)
        import os
        os.system(f"python -m spacy download {model_name}")
        return spacy.load(model_name)

def parse_review(text: str, nlp: Language | None = None) -> spacy.tokens.Doc:
    """Parse a review text using the provided nlp model or the default one."""
    if nlp is None:
        nlp = load_spacy()
    return nlp(text)
