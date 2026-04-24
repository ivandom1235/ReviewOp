from ..explicit.spacy_pipeline import load_spacy

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
