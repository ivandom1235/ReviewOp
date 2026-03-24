"""Aspect and sentiment inference for explicit + implicit labels."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from aspect_extract import extract_explicit_aspects
from evidence_extract import extract_evidence
from mappings import (
    BROAD_CANONICAL_ASPECTS,
    CANONICAL_EXPORT_FALLBACK,
    CANONICAL_EXPORT_LOOKUP,
    CANONICAL_ASPECTS,
    EXPORT_CANONICAL_ASPECTS,
    GENERIC_ASPECTS,
    IMPLICIT_SYMPTOMS,
    SENTIMENT_MAP,
    SURFACE_STOPWORDS,
    UNIVERSAL_ASPECT_HINTS,
)
from senticnet_utils import senticnet_vote
from sentiment_label import infer_aspect_sentiment
from utils import normalize_text, split_sentences
import re


INFERENCE_SETTINGS = {
    "min_implicit_vote_sources": 2,
    "strong_senticnet_threshold": 0.8,
    "conservative_second_aspect_extraction": True,
}


def configure_inference(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        if key in INFERENCE_SETTINGS:
            INFERENCE_SETTINGS[key] = value


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


def _token_set(value: str) -> set[str]:
    return {part for part in sanitize_aspect_token(value).split("_") if part}


def _build_lookup(domain: str, *, include_universal: bool = True) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    mapping = dict(GENERIC_ASPECTS)
    if domain in CANONICAL_ASPECTS:
        mapping.update(CANONICAL_ASPECTS[domain])
    if include_universal:
        mapping.update(UNIVERSAL_ASPECT_HINTS)
    for canonical, aliases in mapping.items():
        lookup[canonical] = canonical
        for a in aliases:
            lookup[normalize_text(a).lower()] = canonical
    return lookup


def infer_aspect_family(raw_aspect: str, domain: str) -> str:
    raw_norm = sanitize_aspect_token(raw_aspect)
    if not raw_norm:
        return ""
    raw_tokens = _token_set(raw_norm)
    for canonical, aliases in UNIVERSAL_ASPECT_HINTS.items():
        if raw_norm == canonical:
            return canonical
        alias_tokens = [_token_set(alias) for alias in aliases]
        if any(raw_tokens and raw_tokens == tokens for tokens in alias_tokens):
            return canonical
    universal_lookup = _build_lookup(domain, include_universal=True)
    if raw_norm in universal_lookup:
        return universal_lookup[raw_norm]
    for alias, canonical in universal_lookup.items():
        alias_tokens = _token_set(alias)
        if raw_tokens and alias_tokens and raw_tokens == alias_tokens:
            return canonical
    return ""


def force_canonical_aspect(raw_aspect: str, domain: str) -> str:
    raw_norm = sanitize_aspect_token(raw_aspect)
    if not raw_norm:
        return ""
    if raw_norm in EXPORT_CANONICAL_ASPECTS:
        return raw_norm
    if raw_norm in CANONICAL_EXPORT_LOOKUP:
        return CANONICAL_EXPORT_LOOKUP[raw_norm]
    family = infer_aspect_family(raw_norm, domain)
    if family in EXPORT_CANONICAL_ASPECTS:
        return family
    if raw_norm.startswith("other_"):
        return CANONICAL_EXPORT_FALLBACK.get(domain, "product_quality")
    return CANONICAL_EXPORT_FALLBACK.get(domain, "product_quality")


def map_aspect(raw_aspect: str, domain: str, prefer_canonical: bool = True) -> Tuple[str, float, str]:
    raw_norm = sanitize_aspect_token(raw_aspect)
    if not raw_norm:
        return "", 0.0, "none"
    raw_tokens = _token_set(raw_norm)

    if raw_norm in EXPORT_CANONICAL_ASPECTS:
        return raw_norm, 0.98, "export_canonical"
    if raw_norm in CANONICAL_EXPORT_LOOKUP:
        return CANONICAL_EXPORT_LOOKUP[raw_norm], 0.93, "export_alias"

    lookup = _build_lookup(domain, include_universal=False)
    if raw_norm in lookup:
        mapped = force_canonical_aspect(lookup[raw_norm], domain)
        return mapped, 0.95, "canonical"

    # fuzzy contains match
    for k, v in lookup.items():
        alias_tokens = _token_set(k)
        if raw_tokens and alias_tokens and (
            raw_tokens == alias_tokens or (len(raw_tokens) > 1 and raw_tokens.issubset(alias_tokens))
        ):
            return force_canonical_aspect(v, domain), 0.75, "alias_fuzzy"

    universal_lookup = _build_lookup(domain, include_universal=True)
    if raw_norm in universal_lookup:
        return force_canonical_aspect(universal_lookup[raw_norm], domain), 0.85, "universal_canonical"
    for k, v in universal_lookup.items():
        alias_tokens = _token_set(k)
        if raw_tokens and alias_tokens and (
            raw_tokens == alias_tokens or (len(raw_tokens) > 1 and raw_tokens.issubset(alias_tokens))
        ):
            return force_canonical_aspect(v, domain), 0.7, "universal_fuzzy"

    if prefer_canonical:
        return force_canonical_aspect(raw_norm, domain), 0.42, "canonical_fallback"
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
        extracted = extract_evidence(review_text, ev, aspect_text)
        return str(extracted.get("evidence_text") or extracted.get("evidence_sentence") or ev).strip() or ev

    start = _to_int_or_none(span_from_value)
    end = _to_int_or_none(span_to_value)
    if start is not None and end is not None and 0 <= start < end <= len(review_text):
        span_text = normalize_text(review_text[start:end])
        if span_text:
            extracted = extract_evidence(review_text, span_text, aspect_text)
            return str(extracted.get("evidence_text") or extracted.get("evidence_sentence") or span_text).strip() or span_text

    aspect_lc = sanitize_aspect_token(aspect_text).replace("_", " ")
    for sent in split_sentences(review_text):
        if aspect_lc and aspect_lc in sent.lower():
            # Prefer a tighter clause when the aspect is explicitly present.
            parts = [p.strip() for p in re.split(r"[,;:\-]", sent) if p.strip()]
            containing = [p for p in parts if aspect_lc in p.lower()]
            tight_pool = containing or parts
            tight = min(tight_pool, key=len) if tight_pool else sent
            extracted = extract_evidence(review_text, tight, aspect_text)
            return str(extracted.get("evidence_text") or extracted.get("evidence_sentence") or tight).strip() or tight
    sents = split_sentences(review_text)
    fallback = sents[0] if sents else normalize_text(review_text)
    extracted = extract_evidence(review_text, fallback, aspect_text)
    return str(extracted.get("evidence_text") or extracted.get("evidence_sentence") or fallback).strip() or fallback


def build_evidence_metadata(review_text: str, evidence_sentence: str, aspect_text: str, *, raw_aspect: str = "") -> Dict[str, Any]:
    aspect_key = raw_aspect or aspect_text
    evidence_info = extract_evidence(review_text, evidence_sentence, aspect_key)
    return {
        "evidence_quality": float(evidence_info.get("evidence_quality", 0.0)),
        "evidence_text": str(evidence_info.get("evidence_text", "")),
        "evidence_is_sentence_fallback": bool(evidence_info.get("is_sentence_fallback", True)),
    }


def _tighten_implicit_evidence(sentence: str, matched_symptom: str) -> str:
    sentence = normalize_text(sentence)
    symptom = normalize_text(matched_symptom)
    if not sentence:
        return sentence
    if not symptom:
        return sentence
    symptom_tokens = {tok for tok in re.findall(r"[a-z0-9_]+", symptom.lower()) if len(tok) > 2}
    clauses = [part.strip() for part in re.split(r"[,;:]|\bbut\b|\bbecause\b|\bwhile\b|\band\b", sentence, flags=re.IGNORECASE) if part.strip()]
    if len(clauses) <= 1:
        return sentence
    scored = []
    for clause in clauses:
        clause_tokens = {tok for tok in re.findall(r"[a-z0-9_]+", clause.lower()) if len(tok) > 2}
        overlap = len(symptom_tokens.intersection(clause_tokens))
        scored.append((overlap, -len(clause), clause))
    best_overlap, _, best_clause = max(scored)
    return best_clause if best_overlap > 0 else sentence


def _prefer_surface_detail(current_surface: str, candidate_surface: str, aspect: str) -> str:
    current = sanitize_aspect_token(current_surface)
    candidate = sanitize_aspect_token(candidate_surface)
    canonical = sanitize_aspect_token(aspect)
    if not candidate or candidate == canonical:
        return current or candidate
    if current in SURFACE_STOPWORDS or current in BROAD_CANONICAL_ASPECTS or current == canonical:
        return candidate
    return current or candidate


import numpy as np
try:
    from rapidfuzz import fuzz
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-mpnet-base-v2")
except Exception:
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

def _aggregate_vote_scores(votes: List[Tuple[str, float]]) -> float:
    if not votes:
        return 0.0
    weight_sum = sum(weight for _, weight in votes)
    cap = max(1.0, len(votes) * 0.85)
    return min(0.99, weight_sum / cap)


def _explicit_transfer_vote(
    sentence: str,
    aspect: str,
    explicit_aspects: List[str],
    domain: str,
) -> float:
    if not explicit_aspects:
        return 0.0
    if aspect in explicit_aspects:
        return 0.18
    lookup = _build_lookup(domain)
    canonical = lookup.get(aspect, aspect)
    if canonical in explicit_aspects:
        return 0.15
    return 0.0


def infer_implicit_aspects(review_text: str, domain: str, explicit_aspects: List[str] | None = None) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    sentences = split_sentences(review_text)
    explicit_aspects = [sanitize_aspect_token(x) for x in (explicit_aspects or []) if sanitize_aspect_token(x)]
    
    from mappings import IMPLICIT_SYMPTOMS

    for sent in sentences:
        sent_lc = sent.lower()
        if len(sent_lc.split()) < 3:
            continue
            
        sent_emb = encoder.encode([sent_lc]) if encoder else None
        candidate_votes: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        candidate_matches: Dict[str, str] = {}
        sentic_vote = senticnet_vote(sent_lc, domain=domain)

        for aspect, patterns in IMPLICIT_SYMPTOMS.items():
            # 1. Fuzzy match
            if fuzz:
                for p in patterns:
                    score = fuzz.partial_ratio(p, sent_lc) / 100.0
                    if score > 0.80:
                        candidate_votes[aspect].append(("fuzzy", score))
                        if aspect not in candidate_matches or score > 0.9:
                            candidate_matches[aspect] = p
                        
            # 2. Semantic match
            if encoder and aspect in SYMPTOM_EMBEDDINGS:
                _, symp_embs = SYMPTOM_EMBEDDINGS[aspect]
                sims = np.dot(sent_emb, symp_embs.T) / (
                    np.linalg.norm(sent_emb) * np.linalg.norm(symp_embs, axis=1)
                )
                max_sim = float(np.max(sims))
                if max_sim > 0.62:
                    candidate_votes[aspect].append(("semantic", max_sim))
                    candidate_matches[aspect] = patterns[int(np.argmax(sims))]

            transfer = _explicit_transfer_vote(sent_lc, aspect, explicit_aspects, domain)
            if transfer > 0:
                candidate_votes[aspect].append(("explicit_transfer", transfer))
            sentic_score = float(sentic_vote.get("aspect_scores", {}).get(aspect, 0.0))
            if sentic_score > 0.0:
                candidate_votes[aspect].append(("sentic_vote", sentic_score))
                if aspect not in candidate_matches and sentic_vote.get("best_concept"):
                    candidate_matches[aspect] = str(sentic_vote.get("best_concept", ""))

        if sentic_vote.get("matched") and sentic_vote.get("best_aspect"):
            candidate_votes[str(sentic_vote["best_aspect"])].append(("sentic_primary", float(sentic_vote.get("aspect_scores", {}).get(sentic_vote["best_aspect"], 0.0))))
            if str(sentic_vote["best_aspect"]) not in candidate_matches:
                candidate_matches[str(sentic_vote["best_aspect"])] = str(sentic_vote.get("best_concept", ""))

        ranked = sorted(
            (
                (_aggregate_vote_scores(votes), aspect, votes)
                for aspect, votes in candidate_votes.items()
            ),
            reverse=True,
        )
        if not ranked:
            continue

        best_conf, best_aspect, votes = ranked[0]
        final_aspect = force_canonical_aspect(best_aspect, domain)
        vote_sources = [name for name, _ in votes]
        strong_sentic = any(name.startswith("sentic") and score >= float(INFERENCE_SETTINGS["strong_senticnet_threshold"]) for name, score in votes)
        required_conf = 0.56 if len(set(vote_sources)) >= int(INFERENCE_SETTINGS["min_implicit_vote_sources"]) else 0.62
        if sentic_vote.get("matched") and str(sentic_vote.get("best_aspect", "")) == best_aspect:
            required_conf -= 0.02
        if best_conf >= required_conf and final_aspect:
            if len(set(vote_sources)) < int(INFERENCE_SETTINGS["min_implicit_vote_sources"]) and not strong_sentic:
                continue
            sentiment_info = infer_aspect_sentiment(
                evidence_sentence=sent,
                raw_sentiment="neutral",
                rating=None,
                multi_aspect=True,
            )
            evidence_seed = _tighten_implicit_evidence(sent, candidate_matches.get(best_aspect, ""))
            evidence_sentence = choose_evidence_sentence(review_text, final_aspect, evidence_seed)
            evidence_meta = build_evidence_metadata(review_text, evidence_sentence, final_aspect, raw_aspect=best_aspect)
            surface_detail = _prefer_surface_detail(final_aspect, candidate_matches.get(best_aspect, ""), final_aspect)
            labels.append({
                "aspect": final_aspect,
                "implicit_aspect": surface_detail,
                "sentiment": str(sentiment_info["sentiment"]),
                "evidence_sentence": evidence_sentence,
                "type": "implicit",
                "confidence": best_conf,
                "metadata": {
                    "rule": "weak_supervision",
                    "matched_symptom": candidate_matches.get(best_aspect, ""),
                    "vote_sources": vote_sources,
                    "aspect_surface": surface_detail,
                    "aspect_family": infer_aspect_family(final_aspect, domain),
                    "senticnet_concept": str(sentic_vote.get("best_concept", "")) if sentic_vote.get("best_aspect") == best_aspect else "",
                    "senticnet_polarity": float(sentic_vote.get("best_polarity", 0.0)) if sentic_vote.get("best_aspect") == best_aspect else 0.0,
                    "canonicalized_from": best_aspect,
                    **evidence_meta,
                }
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

    normalized_aspect_values = [normalize_text(a) for a in aspect_values if normalize_text(a)]
    explicit_candidates = list(normalized_aspect_values)
    extracted_explicit = extract_explicit_aspects(review_text)
    if not explicit_candidates:
        explicit_candidates = [str(item.get("normalized_phrase") or item.get("aspect_phrase") or "").strip() for item in extracted_explicit]

    for aspect in explicit_candidates:
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
        evidence_meta = build_evidence_metadata(review_text, evidence, mapped or aspect, raw_aspect=aspect)
        labels.append(
            {
                "aspect": mapped,
                "implicit_aspect": _prefer_surface_detail(mapped or aspect, aspect, mapped or aspect),
                "sentiment": sentiment,
                "evidence_sentence": evidence,
                "type": "explicit",
                "confidence": conf,
                "metadata": {
                    "raw_aspect": normalize_text(aspect),
                    "mapping_mode": mode,
                    "aspect_surface": _prefer_surface_detail(mapped or aspect, aspect, mapped or aspect),
                    "aspect_family": infer_aspect_family(mapped or aspect, domain),
                    **evidence_meta,
                },
            }
        )

    if INFERENCE_SETTINGS["conservative_second_aspect_extraction"] and labels and len(labels) < 2:
        existing_surfaces = {sanitize_aspect_token(str(label.get("implicit_aspect", ""))) for label in labels}
        for candidate in extracted_explicit:
            candidate_text = str(candidate.get("normalized_phrase") or candidate.get("aspect_phrase") or "").strip()
            candidate_surface = sanitize_aspect_token(candidate_text)
            if not candidate_surface or candidate_surface in existing_surfaces:
                continue
            if float(candidate.get("confidence", 0.0)) < 0.85:
                continue
            candidate_sent = str(candidate.get("sentiment", sentiment)).strip().lower() or sentiment
            candidate_evidence = normalize_text(candidate.get("evidence_span", "")).strip() or review_text
            if candidate_evidence == labels[0].get("evidence_sentence", ""):
                continue
            mapped, conf, mode = map_aspect(candidate_text, domain, prefer_canonical=prefer_canonical)
            if conf < max(confidence_threshold, 0.65):
                continue
            evidence_meta = build_evidence_metadata(review_text, candidate_evidence, mapped or candidate_text, raw_aspect=candidate_text)
            labels.append(
                {
                    "aspect": mapped,
                    "implicit_aspect": _prefer_surface_detail(mapped or candidate_text, candidate_surface or candidate_text, mapped or candidate_text),
                    "sentiment": candidate_sent,
                    "evidence_sentence": candidate_evidence,
                    "type": "explicit",
                    "confidence": max(conf, float(candidate.get("confidence", 0.0)) - 0.08),
                    "metadata": {
                        "raw_aspect": normalize_text(candidate_text),
                        "mapping_mode": f"{mode}_secondary",
                        "aspect_surface": _prefer_surface_detail(mapped or candidate_text, candidate_surface or candidate_text, mapped or candidate_text),
                        "aspect_family": infer_aspect_family(mapped or candidate_text, domain),
                        "secondary_extraction": True,
                        **evidence_meta,
                    },
                }
            )
            break

    # Always look for implicit aspects via weak supervision to build the dual branch
    explicit_mapped = [str(label.get("aspect", "")) for label in labels if label.get("type") == "explicit"]
    implicit_candidates = infer_implicit_aspects(
        review_text=review_text,
        domain=domain,
        explicit_aspects=explicit_mapped,
    )
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
