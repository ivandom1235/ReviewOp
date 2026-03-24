from __future__ import annotations

import hashlib
import re
import copy
from typing import Dict, List, Tuple

import spacy
from datasketch import MinHash, MinHashLSH
try:
    from langdetect import detect_langs
except ImportError:
    pass


# Load spacy model for sentence segmentation; download if not present
try:
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser", "tagger", "lemmatizer"])
    nlp.add_pipe("sentencizer")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser", "tagger", "lemmatizer"])
    nlp.add_pipe("sentencizer")

def strip_html(text: str) -> str:
    return re.sub(r'<[^>]*>', '', text)

def remove_pii(text: str) -> str:
    # Basic regex for emails and phones
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    return text

def detect_language(text: str) -> str:
    try:
        langs = detect_langs(text)
        return langs[0].lang if langs else "en"
    except Exception:
        return "en"

def normalize_text(text: str) -> str:

    text = str(text or "")
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = strip_html(text)
    text = remove_pii(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def segment_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def token_count(text: str) -> int:
    return len([t for t in re.split(r"\W+", text.lower()) if t])

def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> List[str]:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
        if len(chunk) < max_tokens:
            break
        i += (max_tokens - overlap)
    return chunks

def _get_minhash(text: str, num_perm: int = 128) -> MinHash:
    tokens = [t for t in re.split(r"\W+", text.lower()) if t]
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode('utf8'))
    return m

def standardize_rating(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        v = float(str(value).strip())
    except Exception:
        return None
    if v <= 0:
        return None
    if v <= 5:
        return int(round(v))
    if v <= 10:
        return int(round(v / 2))
    return 5

def clean_records(
    records: List[Dict],
    min_review_length: int,
    near_dup_threshold: float,
    dedupe_exact: bool = True,
) -> Tuple[List[Dict], Dict[str, int]]:
    cleaned: List[Dict] = []
    removed = {"empty": 0, "too_short": 0, "exact_dup": 0, "near_dup": 0}
    seen_hashes = set()
    
    enable_near_dup = near_dup_threshold < 1.0
    lsh = MinHashLSH(threshold=near_dup_threshold, num_perm=128) if enable_near_dup else None

    # We use a running id_counter for lsh insertion
    id_counter = 0

    for r in records:
        raw = normalize_text(r.get("raw_text", ""))
        text = normalize_text(r.get("clean_text", raw))
        if not text:
            removed["empty"] += 1
            continue
        if token_count(text) < min_review_length:
            removed["too_short"] += 1
            continue
            
        chunks = chunk_text(text, max_tokens=512, overlap=64)
        
        for chunk_idx, chunk_txt in enumerate(chunks):
            chunk_record = copy.deepcopy(r)
            chunk_record["clean_text"] = chunk_txt
            if len(chunks) > 1:
                chunk_record["chunk_of"] = r.get("id", r.get("record_id", "unknown"))
                chunk_record["chunk_idx"] = chunk_idx

            digest = hashlib.sha1(chunk_txt.lower().encode("utf-8")).hexdigest()
            if dedupe_exact and digest in seen_hashes:
                removed["exact_dup"] += 1
                continue
                
            chunk_record["near_duplicate"] = False
            if enable_near_dup:
                m = _get_minhash(chunk_txt)
                result = lsh.query(m)
                if result:
                    removed["near_dup"] += 1
                    chunk_record["near_duplicate"] = True
                else:
                    lsh.insert(str(id_counter), m)
                    id_counter += 1
                    
            if dedupe_exact:
                seen_hashes.add(digest)
                
            chunk_record["sentences"] = segment_sentences(chunk_txt)
            chunk_record["language"] = detect_language(chunk_txt)
            cleaned.append(chunk_record)

    return cleaned, removed

