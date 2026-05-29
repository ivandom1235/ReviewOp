from __future__ import annotations


GENERIC = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
    "do", "does", "did", "good", "bad", "great", "poor", "amazing", "very"
}

ACTION_VERBS = {"provide", "get", "give", "take", "make", "use", "want", "need", "visit", "go", "come"}
WEAK_NOUNS = {"visit", "road", "center", "place", "area", "thing", "something", "anything", "nothing", "everything", "way", "lot", "bit", "product", "service", "experience", "stuff", "m"}
QUESTION_PREFIXES = ("what ", "why ", "how ", "who ", "where ", "when ", "which ", "what am ", "what is ", "what are ", "how do ", "how can ", "how could ")
NOISY_PHRASES = (
    "what am i supposed to do",
    "what am i meant to do",
    "provide me the",
    "provide me with the",
)


def clean_phrase(phrase: str) -> str:
    return " ".join(part for part in str(phrase or "").lower().split() if part not in GENERIC)


SENTIMENT_ADJECTIVES = {
    "good", "great", "excellent", "amazing", "nice", "bad", "poor", "terrible", 
    "awful", "perfect", "worst", "best", "wonderful", "fantastic", "decent"
}
BROAD_NOUNS = {
    "place", "thing", "service", "experience", "quality", "job", "work", "program", 
    "feature", "item", "product", "stuff", "everything", "something", "area", "part"
}

def is_noisy_label(label: str) -> bool:
    label = str(label or "").lower().strip()
    if not label:
        return True

    # 1. Single character spans or empty after quote removal
    clean_label = label.replace('"', "").replace("'", "").replace("`", "").strip()
    if len(clean_label) <= 1:
        return True

    # 2. Quoted fragments or fragment-like punctuation
    if label.startswith(("'", '"', "`")) or label.endswith(("'", '"', "`")):
        return True
    
    parts = label.split()
    # 3. Too many parts (likely a clause, not an aspect)
    if len(parts) > 5:
        return True
    
    # 4. Action verbs (likely an instruction or action, not an aspect)
    if any(p in ACTION_VERBS for p in parts):
        return True
        
    # 5. Weak/Generic nouns that lack descriptive power on their own
    if len(parts) == 1 and parts[0] in WEAK_NOUNS:
        return True

    # 6. Sentiment Adjective + Broad Noun (Generic Praise/Complaint)
    # e.g., "great service", "excellent quality", "bad experience"
    if len(parts) == 2:
        if parts[0] in SENTIMENT_ADJECTIVES and parts[1] in BROAD_NOUNS:
            return True
        
    # 7. Clause-like starters
    if label in {"it is", "there is", "this is", "i have", "they have", "we have", "i was", "it was"}:
        return True

    # 8. Question fragments
    if label.startswith(QUESTION_PREFIXES) or label.endswith("?"):
        return True

    # 9. Known noisy phrases
    if any(phrase in label for phrase in NOISY_PHRASES):
        return True
        
    return False

def drop_generic_terms(phrases: list[str]) -> list[str]:
    return [cleaned for phrase in phrases if (cleaned := clean_phrase(phrase))]


def drop_context_only_terms(phrases: list[str]) -> list[str]:
    return [phrase for phrase in phrases if len(phrase) > 2 and not is_noisy_label(phrase)]
