from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
from typing import Any, Dict, List

from utils import normalize_whitespace, split_sentences, tokenize


POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "clean", "comfortable", "delicious", "easy",
    "excellent", "fantastic", "fast", "friendly", "good", "great", "helpful",
    "intuitive", "nice", "perfect", "quick", "responsive", "smooth", "stunning",
    "tasty", "wonderful", "love", "like",
    "decent", "solid", "reliable", "impressive", "pleasant", "convenient", "durable",
    "sturdy", "soft", "bright", "warm", "quiet", "crisp", "handy", "recommend",
    "pleased", "happy", "cozy", "elegant", "premium", "superb", "flawless",
    "gorgeous", "brilliant", "outstanding", "fine", "neat", "cute", "adorable", "comfy",
}

NEGATIVE_WORDS = {
    "awful", "bad", "bland", "broken", "cheap", "confusing", "crash", "crashes",
    "dirty", "expensive", "frustrating", "horrible", "laggy", "late", "noisy",
    "poor", "rude", "slow", "terrible", "uncomfortable", "unhelpful", "worst",
    "mediocre", "overpriced", "disappointing", "flimsy", "sticky", "stiff", "dim",
    "scratchy", "weak", "lumpy", "rough", "soggy", "stale", "greasy", "salty",
    "tiny", "loud", "defective", "unusable", "useless", "garbage", "trash", "junk",
    "rubbish", "avoid", "waste", "regret", "annoying", "disappointing",
}

STOP_TOKENS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had",
    "have", "has", "he", "her", "here", "his", "i", "if", "in", "is", "it", "its",
    "me", "my", "not", "of", "on", "or", "our", "she", "so", "than", "that",
    "the", "their", "them", "there", "these", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "while", "with", "you", "your",
}

LATENT_ASPECT_RULES = [
    ("value", {"price", "cost", "expensive", "cheap", "value", "affordable", "worth", "fair", "priced",
              "pricey", "pricy", "overpriced", "budget", "deal", "bargain", "money", "saving"}),
    ("power", {"battery", "battery life", "charger", "charging", "power", "plug", "adapter"}),
    ("display quality", {"screen", "display", "brightness", "resolution", "monitor", "dim", "bright"}),
    ("input quality", {"keyboard", "trackpad", "touchpad", "mouse", "input", "touch"}),
    ("performance", {"performance", "speed", "fast", "slow", "lag", "responsive", "software", "app", "smooth",
                     "efficient", "powerful"}),
    ("food quality", {"food", "meal", "taste", "tasty", "flavor", "fresh", "dessert", "coffee", "burger", "pizza",
                      "dish", "ingredient", "cook", "recipe", "spicy", "savory", "portion", "delicious"}),
    ("service quality", {"service", "staff", "support", "waiter", "waitress", "driver", "doctor", "nurse", "crew", "host",
                         "helpful", "attentive", "responsive", "polite", "impatient"}),
    ("cleanliness", {"clean", "dirty", "messy", "hygiene", "neat", "tidy",
                     "spotless", "filthy", "stain", "sanitary", "dust", "gross"}),
    ("timeliness", {"time", "late", "delay", "quick", "fast", "prompt", "arrive",
                    "wait", "waiting", "waited", "rush", "speedy", "overnight", "instant"}),
    ("communication", {"clear", "clearly", "explain", "inform", "reply", "notification", "email", "follow up"}),
    ("comfort", {"comfortable", "uncomfortable", "crowded", "quiet", "noisy", "spacious", "cold", "hot", "stuffy",
                 "cozy", "roomy", "cramped", "pleasant", "relaxing"}),
    ("organization", {"organized", "organised", "neat", "orderly", "structured"}),
    ("navigation", {"map", "route", "signage", "station", "pickup", "location"}),
    ("accessibility", {"easy", "navigate", "directions", "accessible", "access"}),
    ("reliability", {"issue", "problem", "crash", "broken", "stable", "durable",
                     "defect", "defective", "fail", "failure", "malfunction"}),
    ("security", {"secure", "safe", "privacy", "password", "login", "authentication"}),
    ("transparency", {"transparent", "fees", "billing", "policy"}),
    ("sound quality", {"sound", "audio", "music", "volume", "speaker", "noisy"}),
]

VALID_LATENT_ASPECTS = {label for label, _ in LATENT_ASPECT_RULES} | {"general"}


def _canonical_mode(mode: str) -> str:
    mode = str(mode or "").strip().lower()
    return {"heuristic": "zeroshot", "benchmark": "supervised", "zero-shot": "zeroshot"}.get(mode, mode or "zeroshot")


def _normalize_aspect(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip("- ")
    return " ".join(token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token for token in text.split())


def _is_meaningful_aspect(value: str) -> bool:
    aspect = _normalize_aspect(value)
    tokens = [token for token in aspect.split() if token]
    if not aspect or len(aspect) < 3 or not tokens:
        return False
    if all(token in STOP_TOKENS for token in tokens):
        return False
    if len(tokens) == 1 and (tokens[0] in STOP_TOKENS or tokens[0].isdigit()):
        return False
    return True


def _is_valid_latent_aspect(value: str) -> bool:
    return _normalize_aspect(value) in VALID_LATENT_ASPECTS


def _latent_aspect_label(aspect: str, clause: str | None = None) -> str:
    text = _normalize_aspect(aspect)
    haystack = f"{text} {normalize_whitespace(clause or '').lower()}"
    for label, keywords in LATENT_ASPECT_RULES:
        if any(keyword in haystack for keyword in keywords):
            return label
    return "general"


def _stem_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _match_aspect_surface(text: str, aspect: str) -> tuple[str | None, str | None]:
    aspect = _normalize_aspect(aspect)
    if not aspect:
        return None, None
    # Exact match first.
    match = re.search(rf"\b{re.escape(aspect)}\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(0), "exact"
    # Plural-stripped match.
    stemmed = " ".join(_stem_token(token) for token in aspect.split())
    if stemmed != aspect:
        match = re.search(rf"\b{re.escape(stemmed)}\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(0), "near_exact"
    # Individual token match for multi-token aspects.
    aspect_tokens = aspect.split()
    if len(aspect_tokens) > 1:
        for token in aspect_tokens:
            if token in STOP_TOKENS or len(token) < 3:
                continue
            for variant in {token, _stem_token(token)}:
                match = re.search(rf"\b{re.escape(variant)}\b", text, flags=re.IGNORECASE)
                if match:
                    return match.group(0), "near_exact"
    return None, None


_ADVERSATIVE_RE = re.compile(r"\bbut\b|\bhowever\b|\byet\b|\bwhile\b|\balthough\b", re.IGNORECASE)
_COORD_AND_RE = re.compile(r",\s*and\b|\band\b", re.IGNORECASE)
_MIN_CLAUSE_TOKENS_FOR_SPLIT = 10


def _sentence_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for sentence in split_sentences(text):
        # Stage 1: split on adversative conjunctions.
        pieces = [piece.strip(" ,;:-") for piece in _ADVERSATIVE_RE.split(sentence) if piece.strip()]
        if not pieces:
            pieces = [sentence]
        # Stage 2: split long pieces on commas / coordinating 'and'.
        refined: list[str] = []
        for piece in pieces:
            piece_tokens = tokenize(piece)
            if len(piece_tokens) >= _MIN_CLAUSE_TOKENS_FOR_SPLIT:
                sub_pieces = [sub.strip(" ,;:-") for sub in _COORD_AND_RE.split(piece) if sub.strip(" ,;:-")]
                if len(sub_pieces) > 1:
                    refined.extend(sub_pieces)
                else:
                    # Try comma-only split for very long clauses.
                    comma_pieces = [sub.strip(" ,;:-") for sub in piece.split(",") if sub.strip(" ,;:-")]
                    if len(comma_pieces) > 1 and all(len(tokenize(sub)) >= 3 for sub in comma_pieces):
                        refined.extend(comma_pieces)
                    else:
                        refined.append(piece)
            else:
                refined.append(piece)
        clauses.extend(refined)
    return clauses or [normalize_whitespace(text)]


def _infer_fallback_latent_from_clause(clause: str) -> tuple[str | None, int]:
    tokens = set(tokenize(clause))
    if not tokens:
        return None, 0
    ranked: list[tuple[str, int]] = []
    for label, keywords in LATENT_ASPECT_RULES:
        score = 0
        for keyword in keywords:
            keyword_tokens = set(tokenize(keyword))
            if not keyword_tokens:
                continue
            if keyword_tokens.issubset(tokens):
                score += 2
            elif keyword_tokens & tokens:
                score += 1
        if score:
            ranked.append((label, score))
    if not ranked:
        return None, 0
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[0][0], ranked[0][1]


def infer_sentiment(text: str) -> str:
    tokens = tokenize(text)
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _extract_gold_aspects(rows: List[Dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key in ("aspect", "gold_aspect", "target_aspect"):
            value = row.get(key)
            if isinstance(value, str) and _is_meaningful_aspect(value):
                counts[_normalize_aspect(value)] += 1
        if isinstance(row.get("gold_labels"), list):
            for label in row["gold_labels"]:
                if isinstance(label, dict):
                    for key in ("aspect", "text", "implicit_aspect"):
                        value = label.get(key)
                        if isinstance(value, str) and _is_meaningful_aspect(value):
                            counts[_normalize_aspect(value)] += 1
    return [aspect for aspect, _ in counts.most_common()]


def _extract_primary_gold_aspect(row: Dict[str, Any]) -> str | None:
    for key in ("aspect", "gold_aspect", "target_aspect"):
        value = row.get(key)
        if isinstance(value, str) and _is_meaningful_aspect(value):
            return _normalize_aspect(value)
    if isinstance(row.get("gold_labels"), list):
        for label in row["gold_labels"]:
            if isinstance(label, dict):
                for key in ("aspect", "text", "implicit_aspect"):
                    value = label.get(key)
                    if isinstance(value, str) and _is_meaningful_aspect(value):
                        return _normalize_aspect(value)
    return None


def _parse_structured_prediction(raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if raw is None:
        return [], []
    errors: list[str] = []
    payload = raw
    if isinstance(raw, str):
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
        try:
            payload = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            return [], [type(exc).__name__]
    if isinstance(payload, dict):
        items = payload.get("aspects") or payload.get("predictions") or []
    elif isinstance(payload, list):
        items = payload
    else:
        return [], errors
    parsed: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            parsed.append({"aspect": item})
        elif isinstance(item, dict):
            parsed.append(item)
    return parsed, errors


def _build_span(aspect: str, clause: str, *, sentiment: str, confidence: float, source: str) -> dict[str, Any]:
    latent = _latent_aspect_label(aspect, clause)
    matched_surface, support = _match_aspect_surface(clause, aspect)
    matched_surface = matched_surface or aspect
    start = clause.lower().find(matched_surface.lower()) if matched_surface else -1
    return {
        "surface_aspect": aspect,
        "latent_aspect": latent,
        "normalized_aspect": latent,
        "matched_surface": matched_surface,
        "support_type": support or source,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "start_char": max(start, -1),
        "end_char": max(start + len(matched_surface), -1) if matched_surface and start >= 0 else -1,
        "clause": clause,
        "source": source,
    }


def discover_aspects(rows: List[Dict[str, Any]], *, text_column: str, max_aspects: int, implicit_mode: str = "zeroshot") -> list[str]:
    mode = _canonical_mode(implicit_mode)
    counts: Counter[str] = Counter()
    for row in rows:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text:
            continue
        for clause in _sentence_clauses(text):
            for label, keywords in LATENT_ASPECT_RULES:
                haystack = clause.lower()
                if any(keyword in haystack for keyword in keywords):
                    counts[label] += 1
        if mode in {"supervised", "hybrid"}:
            for aspect in _extract_gold_aspects([row]):
                label = _latent_aspect_label(aspect, text)
                if label != "general":
                    counts[label] += 1
    aspects = [aspect for aspect, _ in counts.most_common(max_aspects) if _is_valid_latent_aspect(aspect)]
    return aspects or ["general"]


def build_implicit_row(
    row: Dict[str, Any],
    *,
    text_column: str,
    candidate_aspects: List[str],
    confidence_threshold: float,
    row_index: int,
    domain: str | None = None,
    language: str = "en",
    implicit_mode: str = "zeroshot",
    multilingual_mode: str = "shared_vocab",
    use_coref: bool = False,
    coref_text: str | None = None,
    implicit_ready: bool = True,
    llm_fallback_threshold: float = 0.65,
    enable_llm_fallback: bool = True,
    candidate_aspects_by_language: dict[str, list[str]] | None = None,
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
    strict_domain_conditioning: bool = False,
    domain_conditioning_mode: str = "adaptive_soft",
    domain_prior_boost: float = 0.05,
    domain_prior_penalty: float = 0.08,
    weak_domain_support_row_threshold: int = 80,
    domain_support_rows: int = 0,
    enforce_grounding: bool = True,
) -> Dict[str, Any]:
    mode = _canonical_mode(implicit_mode)
    raw_text = normalize_whitespace(row.get(text_column, ""))
    processed_text = normalize_whitespace(coref_text or raw_text)
    clauses = _sentence_clauses(processed_text)
    sentiment = infer_sentiment(processed_text)
    aspects: list[str] = []
    aspect_confidence: dict[str, float] = {}
    aspect_sentiments: dict[str, str] = {}
    spans: list[dict[str, Any]] = []
    predicted_labels: list[dict[str, Any]] = []
    llm_parse_errors: list[str] = []
    llm_fallback_used = False
    fallback_branch = "none"
    domain_filtered_matches = 0
    domain_prior_boost_count = 0
    domain_prior_penalty_count = 0

    if not implicit_ready:
        return {
            "id": row.get("id"),
            "split": row.get("split"),
            "source_text": raw_text,
            "language": language,
            "track": "skipped",
            "implicit": {
                "mode": mode,
                "multilingual_strategy": multilingual_mode,
                "coreference_enabled": use_coref,
                "coreference_applied": bool(coref_text and coref_text != raw_text),
                "source_text": raw_text,
                "processed_text": processed_text,
                "aspects": [],
                "dominant_sentiment": sentiment,
                "aspect_sentiments": {},
                "aspect_confidence": {},
                "spans": [],
                "predicted_labels": [],
                "sentence_count_processed": len(clauses),
                "row_index": row_index,
                "needs_review": True,
                "review_reason": "implicit_not_ready",
                "implicit_ready": False,
                "language": language,
                "track": "skipped",
                "llm_fallback_used": False,
                "fallback_branch": "implicit_not_ready",
                "llm_parse_error_rate": 0.0,
                "llm_parse_errors": [],
                "skip_reason": "implicit_not_ready",
            },
        }

    language_candidates = (candidate_aspects_by_language or {}).get(language, [])
    domain_candidates = (candidate_aspects_by_domain or {}).get(str(domain or "unknown"), [])
    allowed_latents = {
        _latent_aspect_label(candidate, processed_text)
        for candidate in domain_candidates
    }
    allowed_latents = {aspect for aspect in allowed_latents if aspect != "general" and _is_valid_latent_aspect(aspect)}
    effective_mode = str(domain_conditioning_mode or "").strip().lower()
    if effective_mode not in {"adaptive_soft", "strict_hard", "off"}:
        effective_mode = "strict_hard" if strict_domain_conditioning else "adaptive_soft"
    elif strict_domain_conditioning and effective_mode == "adaptive_soft":
        effective_mode = "strict_hard"
    weak_domain_support = int(domain_support_rows or 0) < int(weak_domain_support_row_threshold)
    row_candidates = list(
        dict.fromkeys(
            [
                *candidate_aspects,
                *language_candidates,
                *domain_candidates,
                *(_extract_gold_aspects([row]) if mode in {"supervised", "hybrid"} else []),
            ]
        )
    )

    for clause in clauses:
        clause_sentiment = infer_sentiment(clause)
        clause_matches: list[dict[str, Any]] = []
        for label, keywords in LATENT_ASPECT_RULES:
            for keyword in sorted(keywords, key=len, reverse=True):
                if keyword in clause.lower():
                    matched_surface, match_type = _match_aspect_surface(clause, keyword)
                    surface = matched_surface or keyword
                    clause_matches.append({"latent": label, "aspect": surface, "support_type": match_type or "exact", "surface": surface})
                    break

        if mode in {"supervised", "hybrid"}:
            for aspect in [*_extract_gold_aspects([row]), *_extract_gold_aspects([{"gold_labels": row.get("gold_labels", [])}])]:
                if not aspect:
                    continue
                label = _latent_aspect_label(aspect, clause)
                if label == "general":
                    continue
                matched_surface, support = _match_aspect_surface(clause, aspect)
                clause_matches.append({"latent": label, "aspect": matched_surface or aspect, "support_type": support or "gold", "surface": matched_surface or aspect})

        if mode == "zeroshot" and not clause_matches:
            for aspect in row_candidates:
                matched_surface, support = _match_aspect_surface(clause, aspect)
                if matched_surface:
                    clause_matches.append({"latent": _latent_aspect_label(aspect, clause), "aspect": matched_surface, "support_type": support or "near_exact", "surface": matched_surface})

        if not clause_matches and enable_llm_fallback:
            parsed_labels, parse_errors = _parse_structured_prediction(row.get("llm_prediction") or row.get("llm_fallback_text"))
            if parsed_labels:
                llm_fallback_used = True
                fallback_branch = "llm_parse"
                llm_parse_errors.extend(parse_errors)
                for item in parsed_labels:
                    aspect = _normalize_aspect(str(item.get("aspect") or item.get("text") or ""))
                    if aspect:
                        clause_matches.append({"latent": _latent_aspect_label(aspect, clause), "aspect": aspect, "support_type": "llm_parse", "surface": aspect, "llm_confidence": float(item.get("confidence", llm_fallback_threshold))})
            else:
                llm_parse_errors.extend(parse_errors)

        if not clause_matches:
            inferred_latent, support_score = _infer_fallback_latent_from_clause(clause)
            sentiment_signal = sum(1 for token in tokenize(clause) if token in POSITIVE_WORDS or token in NEGATIVE_WORDS)
            strong_support = support_score >= 1
            if inferred_latent and inferred_latent != "general" and strong_support and sentiment_signal >= 1:
                llm_fallback_used = True
                if fallback_branch == "none":
                    fallback_branch = "rule_fallback"
                clause_matches.append({
                    "latent": inferred_latent,
                    "aspect": inferred_latent,
                    "support_type": "rule_fallback",
                    "surface": inferred_latent,
                    "llm_confidence": max(confidence_threshold, 0.56),
                })

        for match in clause_matches:
            latent = _normalize_aspect(match["latent"])
            if not _is_valid_latent_aspect(latent):
                continue
            if effective_mode == "strict_hard" and allowed_latents and latent not in allowed_latents:
                domain_filtered_matches += 1
                continue
            confidence = float(match.get("llm_confidence", 1.0 if match.get("support_type") in {"exact", "gold", "llm_parse"} else 0.85))
            if effective_mode == "adaptive_soft" and allowed_latents:
                if latent in allowed_latents:
                    confidence += max(0.0, float(domain_prior_boost))
                    domain_prior_boost_count += 1
                else:
                    penalty = max(0.0, float(domain_prior_penalty))
                    if weak_domain_support:
                        penalty = penalty * 0.35
                    confidence -= penalty
                    domain_prior_penalty_count += 1
            if confidence < confidence_threshold:
                continue
            if latent not in aspects:
                aspects.append(latent)
            aspect_confidence[latent] = round(confidence, 4)
            aspect_sentiments[latent] = clause_sentiment
            spans.append(_build_span(match["aspect"], clause, sentiment=clause_sentiment, confidence=confidence, source=str(match.get("support_type", "exact"))))
            predicted_labels.append({"aspect": latent, "surface_aspect": match["aspect"], "sentiment": clause_sentiment, "confidence": round(confidence, 4)})

    if mode in {"supervised", "hybrid"}:
        for aspect in _extract_gold_aspects([row]):
            latent = _latent_aspect_label(aspect, processed_text)
            if latent == "general":
                continue
            if latent not in aspects:
                aspects.append(latent)
            if latent not in aspect_confidence:
                aspect_confidence[latent] = 1.0
                aspect_sentiments[latent] = sentiment
            if not any(span["latent_aspect"] == latent for span in spans):
                spans.append(_build_span(aspect, processed_text, sentiment=sentiment, confidence=1.0, source="gold"))
            predicted_labels.append({"aspect": latent, "surface_aspect": aspect, "sentiment": sentiment, "confidence": 1.0})

    if not aspects:
        aspects = ["general"]
        aspect_confidence["general"] = 0.5
        aspect_sentiments["general"] = sentiment
        if fallback_branch == "none":
            fallback_branch = "fallback_general"

    strong_support_types = {"exact", "gold"}
    weak_support = (
        ((aspects != ["general"] and not spans) if enforce_grounding else False)
        or any(span.get("support_type") not in strong_support_types for span in spans)
    )
    parse_error_present = bool(llm_parse_errors and not llm_fallback_used)
    needs_review = aspects == ["general"] or weak_support or parse_error_present
    if aspects == ["general"]:
        review_reason = "fallback_general"
    elif parse_error_present:
        review_reason = "llm_parse_error"
    elif weak_support:
        review_reason = "weak_support"
    else:
        review_reason = None
    llm_parse_error_rate = round(len(llm_parse_errors) / max(1, len(clauses)), 4) if llm_parse_errors else 0.0

    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "source_text": raw_text,
        "language": language,
        "track": mode,
        "implicit": {
            "mode": mode,
            "multilingual_strategy": multilingual_mode,
            "coreference_enabled": use_coref,
            "coreference_applied": bool(coref_text and coref_text != raw_text),
            "source_text": raw_text,
            "processed_text": processed_text,
            "aspects": aspects,
            "dominant_sentiment": sentiment,
            "aspect_sentiments": aspect_sentiments,
            "aspect_confidence": aspect_confidence,
            "spans": spans,
            "predicted_labels": predicted_labels,
            "sentence_count_processed": len(clauses),
            "row_index": row_index,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "implicit_ready": True,
            "language": language,
            "track": mode,
            "llm_fallback_used": llm_fallback_used,
            "fallback_branch": fallback_branch,
            "llm_parse_error_rate": llm_parse_error_rate,
            "llm_parse_errors": llm_parse_errors,
            "domain_filtered_matches": domain_filtered_matches,
            "domain_conditioning_mode": effective_mode,
            "weak_domain_support": weak_domain_support,
            "domain_prior_boost_count": domain_prior_boost_count,
            "domain_prior_penalty_count": domain_prior_penalty_count,
            "skip_reason": None,
        },
    }


def collect_diagnostics(rows: List[Dict[str, Any]], *, text_column: str, candidate_aspects: List[str]) -> Dict[str, Any]:
    aspect_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()
    exact_span_count = 0
    near_exact_span_count = 0
    fallback_only_count = 0
    needs_review_count = 0
    rejected_aspect_count = 0
    clause_count = 0
    llm_fallback_count = 0
    llm_parse_error_count = 0
    review_reason_counts: Counter[str] = Counter()
    fallback_branch_counts: Counter[str] = Counter()
    for row in rows:
        text = normalize_whitespace(row.get(text_column, ""))
        clause_count += len(_sentence_clauses(text))
        implicit = row.get("implicit", {})
        language_counts[str(row.get("language", implicit.get("language", "unknown")))] += 1
        track_counts[str(row.get("track", implicit.get("track", "unknown")))] += 1
        if implicit.get("aspects") == ["general"]:
            fallback_only_count += 1
        if implicit.get("needs_review"):
            needs_review_count += 1
        review_reason_counts[str(implicit.get("review_reason") or "none")] += 1
        fallback_branch_counts[str(implicit.get("fallback_branch") or "none")] += 1
        if implicit.get("llm_fallback_used"):
            llm_fallback_count += 1
        llm_parse_error_count += len(implicit.get("llm_parse_errors", []))
        for aspect in implicit.get("aspects", []):
            if aspect != "general":
                aspect_counts[str(aspect)] += 1
            if not _is_valid_latent_aspect(str(aspect)):
                rejected_aspect_count += 1
        for span in implicit.get("spans", []):
            if span.get("support_type") == "exact":
                exact_span_count += 1
            elif span.get("support_type") == "near_exact":
                near_exact_span_count += 1
    total = sum(aspect_counts.values())
    entropy = 0.0
    if total:
        for count in aspect_counts.values():
            share = count / total
            entropy -= share * math.log(share, 2)
    return {
        "candidate_aspects": candidate_aspects,
        "top_implicit_aspects": aspect_counts.most_common(10),
        "meaningful_candidate_aspects": [aspect for aspect in candidate_aspects if _is_valid_latent_aspect(aspect)],
        "generic_candidate_aspects": [aspect for aspect in candidate_aspects if not _is_valid_latent_aspect(aspect)],
        "clause_count": clause_count,
        "span_support": {"exact": exact_span_count, "near_exact": near_exact_span_count},
        "fallback_only_count": fallback_only_count,
        "needs_review_count": needs_review_count,
        "rejected_aspect_count": rejected_aspect_count,
        "domain_agnostic_skew": {
            "unique_aspects": len(aspect_counts),
            "top_share": round(max(aspect_counts.values()) / total, 4) if total else 0.0,
            "entropy": round(entropy, 4),
        },
        "language_distribution": dict(language_counts),
        "track_distribution": dict(track_counts),
        "llm_fallback_count": llm_fallback_count,
        "llm_parse_error_count": llm_parse_error_count,
        "review_reason_counts": dict(review_reason_counts),
        "fallback_branch_counts": dict(fallback_branch_counts),
    }
