from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any
from .schema import ReviewExample

@dataclass
class DistilledAspectProfile:
    aspect: str
    top_terms: list[str]
    evidence_count: int
    source: str = "train_evidence_only"

# A robust list of standard English stop words to filter out common function words
STOP_WORDS = {
    "the", "and", "a", "of", "to", "is", "in", "it", "that", "i", "you", "he", "she", "they", "we", "us",
    "was", "for", "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from", "or",
    "had", "by", "hot", "but", "some", "what", "there", "we", "can", "out", "other", "were", "all",
    "your", "when", "up", "use", "word", "how", "said", "an", "each", "she", "which", "do", "their",
    "time", "if", "will", "way", "about", "many", "then", "them", "would", "write", "like", "so",
    "these", "her", "long", "make", "thing", "see", "him", "two", "has", "look", "more", "day", "could",
    "go", "come", "did", "my", "sound", "no", "most", "number", "who", "over", "know", "water", "than",
    "call", "first", "people", "may", "down", "side", "been", "now", "find", "any", "new", "only",
    "very", "just", "get", "our", "about", "into", "their", "will", "would", "its", "also", "here",
    "out", "about", "only", "would", "then", "more"
}

def clean_and_tokenize(text: str) -> list[str]:
    if not text:
        return []
    # Tokenize alphanumeric strings of length >= 3, converted to lowercase
    tokens = re.findall(r'\b[a-z0-9]{3,}\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS]

def distill_aspect_profiles(
    train_rows: list[ReviewExample],
    config: Any
) -> dict[str, DistilledAspectProfile]:
    top_k = getattr(config, "distill_profiles_top_k", 5)
    eta = getattr(config, "distill_profiles_eta", 1.00)

    # 1. Map aspects to their evidence snippets and occurrences
    aspects = set()
    for ex in train_rows:
        for g in ex.gold_aspects:
            if g.aspect and g.aspect != "unknown":
                aspects.add(g.aspect)
    
    aspects = sorted(list(aspects))
    
    # Pre-tokenize all rows for global counts
    row_tokens = []
    for ex in train_rows:
        row_tokens.append(set(clean_and_tokenize(ex.text)))

    N = len(train_rows)
    if N == 0:
        return {}

    # Global token document frequency
    df = {}
    for tokens in row_tokens:
        for t in tokens:
            df[t] = df.get(t, 0) + 1

    # Aspect-specific gathering
    aspect_evidence_tokens = {a: [] for a in aspects}
    aspect_row_count = {a: 0 for a in aspects}
    
    # Store which row indexes have which aspects to calculate PMI co-occurrence
    aspect_row_indices = {a: set() for a in aspects}

    for idx, ex in enumerate(train_rows):
        row_aspects = {g.aspect for g in ex.gold_aspects if g.aspect and g.aspect != "unknown"}
        for a in row_aspects:
            aspect_row_indices[a].add(idx)
            aspect_row_count[a] += 1
            # Gather evidence snippet
            ev = next((g.evidence_text for g in ex.gold_aspects if g.aspect == a), "")
            text_to_tokenize = ev or ex.text
            aspect_evidence_tokens[a].extend(clean_and_tokenize(text_to_tokenize))

    profiles = {}
    for a in aspects:
        tokens_in_ev = aspect_evidence_tokens[a]
        evidence_count = aspect_row_count[a]
        if not tokens_in_ev:
            profiles[a] = DistilledAspectProfile(
                aspect=a,
                top_terms=[],
                evidence_count=evidence_count
            )
            continue

        # Count TF in evidence snippets
        tf = {}
        for t in tokens_in_ev:
            tf[t] = tf.get(t, 0) + 1
        
        total_tokens = len(tokens_in_ev)
        
        # Compute TF-IDF + PMI for each token
        scores = {}
        for t, count in tf.items():
            # TF
            tf_val = count / total_tokens
            
            # IDF
            idf_val = math.log(N / (df.get(t, 0) + 1.0)) + 1.0
            tfidf = tf_val * idf_val

            # PMI
            # Count how many times token t is in the aspect rows
            cooc_count = 0
            for idx in aspect_row_indices[a]:
                if t in row_tokens[idx]:
                    cooc_count += 1
            
            pmi_val = 0.0
            if cooc_count > 0:
                p_ta = cooc_count / N
                p_t = df.get(t, 0) / N
                p_a = evidence_count / N
                # Avoid numerical issues
                val = math.log(p_ta / (p_t * p_a))
                # Keep positive PMI only to avoid penalizing rare words with valid context
                pmi_val = max(0.0, val)

            scores[t] = tfidf + eta * pmi_val

        # Sort and select top K
        sorted_terms = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        top_terms = sorted_terms[:top_k]

        profiles[a] = DistilledAspectProfile(
            aspect=a,
            top_terms=top_terms,
            evidence_count=evidence_count
        )

    return profiles
