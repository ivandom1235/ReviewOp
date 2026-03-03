import re
from typing import List

STOP = {
    "the","a","an","and","or","but","is","are","was","were","to","of","in","on","for","with","it","this","that","during"
}

def extract_candidate_aspects(text: str, max_aspects: int = 6) -> List[str]:
    """
    Lightweight heuristic: pick repeated noun-ish tokens/phrases.
    MVP-safe fallback ONLY when seq2seq returns nothing.
    """
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    tokens = [w for w in t.split() if len(w) > 2 and w not in STOP]

    # bigrams as pseudo-aspects
    bigrams = []
    for i in range(len(tokens)-1):
        bg = tokens[i] + " " + tokens[i+1]
        bigrams.append(bg)

    # count freq
    from collections import Counter
    c = Counter(bigrams)
    common = [a for a,_ in c.most_common(max_aspects)]
    # if nothing, fallback to top unigrams
    if not common:
        c2 = Counter(tokens)
        common = [a for a,_ in c2.most_common(max_aspects)]
    return [s[:255] for s in common]