from __future__ import annotations


GENERIC = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
    "do", "does", "did", "good", "bad", "great", "poor", "amazing", "very"
}

ACTION_VERBS = {"provide", "get", "give", "take", "make", "use", "want", "need", "visit", "go", "come"}
WEAK_NOUNS = {"visit", "road", "center", "place", "area", "thing", "something", "anything", "nothing", "everything", "way", "lot", "bit"}


def clean_phrase(phrase: str) -> str:
    return " ".join(part for part in str(phrase or "").lower().split() if part not in GENERIC)


def is_noisy_label(label: str) -> bool:
    label = str(label or "").lower().strip()
    if not label:
        return True
    
    parts = label.split()
    # Too long
    if len(parts) > 5:
        return True
    
    # Contains too many verbs/actions
    if any(p in ACTION_VERBS for p in parts):
        return True
        
    # Contains weak nouns
    if any(p in WEAK_NOUNS for p in parts):
        return True
        
    # Too generic / Clause-like
    if label in {"it is", "there is", "this is", "i have", "they have", "we have"}:
        return True
        
    return False


def drop_generic_terms(phrases: list[str]) -> list[str]:
    return [cleaned for phrase in phrases if (cleaned := clean_phrase(phrase))]


def drop_context_only_terms(phrases: list[str]) -> list[str]:
    return [phrase for phrase in phrases if len(phrase) > 2 and not is_noisy_label(phrase)]
