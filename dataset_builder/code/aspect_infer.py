"""Aspect and sentiment inference for explicit + implicit labels."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from mappings import CANONICAL_ASPECTS, GENERIC_ASPECTS, IMPLICIT_SYMPTOMS, SENTIMENT_MAP
from utils import normalize_text, split_sentences
import re


def normalize_sentiment(raw: Any) -> str:
    value = normalize_text(raw).lower()
    if value in {"", "na", "n/a", "none", "null"}:
        return "neutral"
    return SENTIMENT_MAP.get(value, "neutral") if value else "neutral"


def sanitize_aspect_token(token: str) -> str:
    tok = normalize_text(token).lower()
    tok = re.sub(r"[\"'`]+", "", tok)
    tok = re.sub(r"[/\\]+", "_", tok)
    tok = re.sub(r"[^a-z0-9_\\-\\s]", "", tok)
    tok = re.sub(r"\\s+", "_", tok).strip("_")
    tok = re.sub(r"_+", "_", tok)
    return tok


def _build_lookup(domain: str) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    mapping = dict(GENERIC_ASPECTS)
    if domain in CANONICAL_ASPECTS:
        mapping.update(CANONICAL_ASPECTS[domain])
    for canonical, aliases in mapping.items():
        lookup[canonical] = canonical
        for a in aliases:
            lookup[normalize_text(a).lower()] = canonical
    return lookup


def map_aspect(raw_aspect: str, domain: str, prefer_canonical: bool = True) -> Tuple[str, float, str]:
    raw_norm = sanitize_aspect_token(raw_aspect)
    if not raw_norm:
        return "", 0.0, "none"

    lookup = _build_lookup(domain)
    if raw_norm in lookup:
        return lookup[raw_norm], 0.95, "canonical"

    # fuzzy contains match
    for k, v in lookup.items():
        if k in raw_norm or raw_norm in k:
            return v, 0.75, "alias_fuzzy"

    if prefer_canonical:
        return raw_norm, 0.4, "open_aspect"
    return raw_norm, 0.55, "open_aspect"


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def choose_evidence_sentence(
    review_text: str,
    aspect_text: str,
    evidence_value: Any,
    span_from_value: Any = None,
    span_to_value: Any = None,
) -> str:
    ev = normalize_text(evidence_value)
    if ev and ev in review_text:
        return ev

    start = _to_int_or_none(span_from_value)
    end = _to_int_or_none(span_to_value)
    if start is not None and end is not None and 0 <= start < end <= len(review_text):
        span_text = normalize_text(review_text[start:end])
        if span_text:
            return span_text

    aspect_lc = sanitize_aspect_token(aspect_text).replace("_", " ")
    for sent in split_sentences(review_text):
        if aspect_lc and aspect_lc in sent.lower():
            # Prefer a tighter clause when the aspect is explicitly present.
            parts = [p.strip() for p in re.split(r"[,;:\-]", sent) if p.strip()]
            tight = min(parts, key=len) if parts else sent
            return tight
    sents = split_sentences(review_text)
    return sents[0] if sents else normalize_text(review_text)


import numpy as np
try:
    from rapidfuzz import fuzz
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-mpnet-base-v2")
except ImportError:
    fuzz = None
    encoder = None

def get_symptom_embeddings():
    cache = {}
    if not encoder: return cache
    from mappings import IMPLICIT_SYMPTOMS
    for aspect, symptoms in IMPLICIT_SYMPTOMS.items():
        cache[aspect] = (symptoms, encoder.encode(symptoms))
    return cache

SYMPTOM_EMBEDDINGS = get_symptom_embeddings()

def infer_implicit_aspects(review_text: str, domain: str) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    sentences = split_sentences(review_text)
    
    from mappings import IMPLICIT_SYMPTOMS

    for sent in sentences:
        sent_lc = sent.lower()
        if len(sent_lc.split()) < 3:
            continue
            
        best_aspect = None
        best_conf = 0.0
        best_match = ""
        
        sent_emb = encoder.encode([sent_lc]) if encoder else None
            
        for aspect, patterns in IMPLICIT_SYMPTOMS.items():
            # 1. Fuzzy match
            if fuzz:
                for p in patterns:
                    score = fuzz.partial_ratio(p, sent_lc) / 100.0
                    if score > 0.80 and score > best_conf:
                        best_conf = score
                        best_aspect = aspect
                        best_match = p
                        
            # 2. Semantic match
            if encoder and aspect in SYMPTOM_EMBEDDINGS:
                _, symp_embs = SYMPTOM_EMBEDDINGS[aspect]
                sims = np.dot(sent_emb, symp_embs.T) / (
                    np.linalg.norm(sent_emb) * np.linalg.norm(symp_embs, axis=1)
                )
                max_sim = float(np.max(sims))
                if max_sim > 0.62 and max_sim > best_conf: # Lowered threshold for higher recall
                    best_conf = max_sim
                    best_aspect = aspect
                    best_match = patterns[np.argmax(sims)]
                    
        if best_conf >= 0.60 and best_aspect:
            # Stage 5d Quality check: Removed strict token count to allow punchy symptoms
            labels.append({
                "aspect": best_aspect,
                "sentiment": "negative", # implicit symptom maps usually denote complaints
                "evidence_sentence": sent,
                "type": "implicit",
                "confidence": best_conf,
                "metadata": {"rule": "weak_supervision", "matched_symptom": best_match}
            })
            
    return labels



def collect_labels_for_row(
    row: Dict[str, Any],
    review_text: str,
    domain: str,
    aspect_values: Iterable[str],
    sentiment_value: Any,
    evidence_value: Any,
    span_from_value: Any,
    span_to_value: Any,
    prefer_canonical: bool,
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []

    sentiment = normalize_sentiment(sentiment_value)
    review_lc = normalize_text(review_text).lower()

    for aspect in aspect_values:
        if not normalize_text(aspect):
            continue
        mapped, conf, mode = map_aspect(aspect, domain, prefer_canonical=prefer_canonical)
        if conf < confidence_threshold:
            continue
        evidence = choose_evidence_sentence(
            review_text=review_text,
            aspect_text=aspect,
            evidence_value=evidence_value,
            span_from_value=span_from_value,
            span_to_value=span_to_value,
        )
        labels.append(
            {
                "aspect": mapped,
                "sentiment": sentiment,
                "evidence_sentence": evidence,
                "type": "explicit",
                "confidence": conf,
                "metadata": {"raw_aspect": normalize_text(aspect), "mapping_mode": mode},
            }
        )

    # Always look for implicit aspects via weak supervision to build the dual branch
    implicit_candidates = infer_implicit_aspects(review_text=review_text, domain=domain)
    labels.extend(implicit_candidates)

    filtered: List[Dict[str, Any]] = []
    for lab in labels:
        aspect_lc = sanitize_aspect_token(str(lab.get("aspect", ""))).replace("_", " ")
        sent_lc = normalize_text(lab.get("evidence_sentence", "")).lower()
        if lab.get("type") == "implicit":
            if aspect_lc and aspect_lc in sent_lc:
                continue
            if aspect_lc and aspect_lc in review_lc and len(sent_lc.split()) <= 12:
                # Avoid explicit-like short spans being mislabeled as implicit.
                continue
        filtered.append(lab)

    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for lab in filtered:
        key = (lab["aspect"], lab.get("sentiment", "unknown"), lab.get("evidence_sentence", ""))
        prev = merged.get(key)
        if prev is None or lab["confidence"] > prev["confidence"]:
            merged[key] = lab

    return list(merged.values())


def summarize_aspects(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        for label in row.get("labels", []):
            counts[str(label.get("aspect", "unknown"))] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
