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

# Expanded research-grade latent aspect rules with explicit/implicit separation.
# Format: (label, explicit_keywords, implicit_signals)
LATENT_ASPECT_RULES = [
    ("value", {"price", "cost", "bill", "billing", "fees", "dollars", "rate"}, 
              {"expensive", "cheap", "affordable", "worth", "priced", "pricey", "overpriced", "budget", "deal", "bargain", "money", "saving"}),
    ("power", {"battery", "power", "charger", "charging", "plug", "adapter"},
              {"dies", "drains", "lasts", "lasted", "dead", "empty", "hours", "short life", "long life"}),
    ("connectivity", {"network", "signal", "wifi", "bluetooth", "connection", "data", "internet"},
                     {"drops", "searching", "searching for", "disconnected", "spotty", "unstable", "cut out", "no service"}),
    ("thermal", {"temperature", "heat", "thermal", "cooling", "fan"},
                {"hot", "warm", "cool", "burning", "heats up", "overheating", "ice", "stove"}),
    ("performance", {"performance", "speed", "software", "app", "operating system", "os", "hardware", "processor"},
                   {"fast", "slow", "lag", "laggy", "responsive", "smooth", "efficient", "powerful", "snappy", "clunky"}),
    ("display quality", {"screen", "display", "brightness", "resolution", "monitor", "pixel", "panel"},
                       {"dim", "bright", "crisp", "washed out", "blurry", "vivid", "sharp"}),
    ("reliability", {"issue", "problem", "crash", "broken", "stable", "durable", "defect", "defective", "fail", "failure", "malfunction", "support"},
                    {"crashed", "died", "buggy", "stopped working", "frozen", "freezes", "error"}),
    ("build quality", {"build", "material", "structure", "construction", "casing", "exterior", "interior"},
                     {"flimsy", "sturdy", "premium", "cheap plastic", "solid", "durable", "tough", "strong", "fragile"}),
    ("accessibility", {"map", "route", "signage", "station", "pickup", "location", "nav", "gps", "directions", "accessible", "access"},
                     {"easy", "confusing", "lost", "direct", "straightforward", "simple", "shortcut"}),
    ("service quality", {"service", "staff", "support", "waiter", "waitress", "driver", "doctor", "nurse", "crew", "host"},
                        {"helpful", "attentive", "responsive", "polite", "impatient", "friendly", "rude", "nice", "kind"}),
    ("cleanliness", {"clean", "dirty", "hygiene", "neat", "tidy", "sanitary", "dust"},
                     {"spotless", "filthy", "stain", "gross", "messy", "mess", "shining"}),
    ("timeliness", {"time", "schedule", "standard", "arrival", "waiting"},
                    {"late", "delay", "quick", "fast", "prompt", "arrive", "wait", "rush", "speedy", "instant"}),
    ("comfort", {"comfort", "space", "environment", "setting", "room"},
                {"comfortable", "uncomfortable", "crowded", "quiet", "noisy", "spacious", "cozy", "roomy", "cramped"}),
    ("food quality", {"food", "meal", "taste", "dish", "ingredient", "cook", "recipe", "portion"},
                     {"tasty", "flavor", "fresh", "delicious", "savory", "yummy", "bland", "salty", "greasy", "stale"}),
    ("sound quality", {"sound", "audio", "music", "volume", "speaker"},
                      {"noisy", "loud", "quiet", "clear", "tinny", "distorted", "muffled"}),
]

VALID_LATENT_ASPECTS = {label for label, _, _ in LATENT_ASPECT_RULES} | {"general"}
LATENT_RULE_BY_LABEL = {label: (explicit_kws, implicit_sigs) for label, explicit_kws, implicit_sigs in LATENT_ASPECT_RULES}


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
    for label, explicit_kws, implicit_sigs in LATENT_ASPECT_RULES:
        if any(_keyword_in_text(haystack, kw) for kw in explicit_kws | implicit_sigs):
            return label
    return "general"


def _keyword_in_text(text: str, keyword: str) -> bool:
    normalized_text = normalize_whitespace(text).lower()
    normalized_keyword = normalize_whitespace(keyword).lower()
    if not normalized_text or not normalized_keyword:
        return False
    parts = [part for part in normalized_keyword.split() if part]
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(re.escape(part) for part in parts) + r"(?![a-z0-9])"
    return bool(re.search(pattern, normalized_text, flags=re.IGNORECASE))


def _hardness_tier(hardness: int) -> str:
    if hardness >= 3:
        return "H3"
    if hardness == 2:
        return "H2"
    if hardness == 1:
        return "H1"
    return "H0"


def _compute_leakage_flags(
    *,
    text: str,
    latent: str,
    surface_aspect: str,
    label_type: str,
) -> list[str]:
    flags: list[str] = []
    latent_norm = _normalize_aspect(latent)
    surface_norm = _normalize_aspect(surface_aspect)
    if label_type == "explicit":
        flags.append("explicit_span_in_implicit")
    if latent_norm and _keyword_in_text(text, latent_norm):
        flags.append("latent_name_surface_leakage")
    explicit_kws, _ = LATENT_RULE_BY_LABEL.get(latent_norm, (set(), set()))
    for keyword in explicit_kws:
        if _keyword_in_text(text, keyword):
            flags.append("explicit_keyword_surface_leakage")
            break
    if surface_norm and latent_norm and (surface_norm == latent_norm):
        flags.append("surface_equals_latent")
    return sorted(set(flags))


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


def is_surface_leakage(text: str, aspect: str) -> bool:
    """Checks if the aspect name or its direct synonyms appear in the text."""
    text_lower = normalize_whitespace(text).lower()
    aspect_lower = normalize_whitespace(aspect).lower()
    if _keyword_in_text(text_lower, aspect_lower):
        return True
    # Check for plural-stripped aspect.
    stemmed = " ".join(_stem_token(token) for token in aspect_lower.split())
    if _keyword_in_text(text_lower, stemmed):
        return True
    return False


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
    for label, e_kws, i_sigs in LATENT_ASPECT_RULES:
        keywords = e_kws | i_sigs
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
            for label, e_kws, i_sigs in LATENT_ASPECT_RULES:
                haystack = clause.lower()
                if any(_keyword_in_text(haystack, keyword) for keyword in e_kws | i_sigs):
                    counts[label] += 1
        if mode in {"supervised", "hybrid"}:
            for aspect in _extract_gold_aspects([row]):
                label = _latent_aspect_label(aspect, text)
                if label != "general":
                    counts[label] += 1
    aspects = [aspect for aspect, _ in counts.most_common(max_aspects) if _is_valid_latent_aspect(aspect)]
    return aspects or ["general"]


async def build_implicit_row(
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
    enable_reasoned_recovery: bool = False,
    llm_provider: Any = None,
    llm_model_name: str = "llama3",
    high_difficulty: bool = False,
    adversarial_refine: bool = False,
) -> Dict[str, Any]:
    from llm_utils import AsyncLlmProvider, reason_implicit_signal_async

    mode = _canonical_mode(implicit_mode)
    raw_text = normalize_whitespace(row.get(text_column, ""))
    processed_text = normalize_whitespace(coref_text or raw_text)
    clauses = _sentence_clauses(processed_text)
    sentiment = infer_sentiment(processed_text)
    
    # State tracking
    spans: list[dict[str, Any]] = []
    llm_parse_errors: list[str] = []
    llm_fallback_used = False
    fallback_branch = "none"
    reasoned_recovery_used = False
    leakage_flags_total: set[str] = set()

    # Stage A/B: Hybrid Candidate Generation
    for clause in clauses:
        clause_sentiment = infer_sentiment(clause)
        clause_matches: list[dict[str, Any]] = []
        
        # 1. Rules (Explicit & Lexicon)
        for label, explicit_kws, implicit_sigs in LATENT_ASPECT_RULES:
            # Explicit (Hardness 0)
            for kw in sorted(explicit_kws, key=len, reverse=True):
                if _keyword_in_text(clause, kw):
                    matched, match_type = _match_aspect_surface(clause, kw)
                    clause_matches.append({
                        "latent": label,
                        "aspect": matched or kw,
                        "support_type": match_type or "exact",
                        "label_type": "explicit",
                        "hardness": 0,
                        "confidence": 1.0,
                        "source": "rule",
                    })
                    break
            
            # Implicit Signals (Hardness 1)
            for sig in sorted(implicit_sigs, key=len, reverse=True):
                if _keyword_in_text(clause, sig):
                    matched, match_type = _match_aspect_surface(clause, sig)
                    clause_matches.append({
                        "latent": label,
                        "aspect": matched or sig,
                        "support_type": match_type or "near_exact",
                        "label_type": "implicit",
                        "hardness": 1,
                        "confidence": 0.85,
                        "source": "lexicon",
                    })
                    break

        # 2. LLM Fallback (Stage C - only if no lexical matches)
        if not clause_matches and enable_llm_fallback and llm_provider:
            llm_fallback_used = True
            paraphrases = await reason_implicit_signal_async(clause, candidate_aspects, llm_provider, llm_model_name)
            for para in paraphrases:
                for label, e_kws, i_sigs in LATENT_ASPECT_RULES:
                    for kw in sorted(e_kws | i_sigs, key=len, reverse=True):
                        if kw.lower() in para.lower():
                            clause_matches.append({
                                "latent": label,
                                "aspect": kw,
                                "support_type": "llm_reasoning",
                                "label_type": "implicit",
                                "hardness": 2,
                                "confidence": 0.75,
                                "source": "llm",
                            })
                            reasoned_recovery_used = True
                            break
                    if clause_matches: break
                if clause_matches: break

        # Stage D: Validation & Grounding
        for match in clause_matches:
            latent = match["latent"]
            if not _is_valid_latent_aspect(latent): continue

            flags = _compute_leakage_flags(
                text=processed_text,
                latent=latent,
                surface_aspect=match["aspect"],
                label_type=match["label_type"],
            )
            leakage_flags_total.update(flags)
            
            matched_surface = match["aspect"]
            start = clause.lower().find(matched_surface.lower())
            
            if start == -1 and match["source"] != "llm":
                continue 

            spans.append({
                "aspect": matched_surface,
                "latent_label": latent,
                "sentiment": clause_sentiment,
                "confidence": match["confidence"],
                "support_type": match["support_type"],
                "label_type": match["label_type"],
                "source": match["source"],
                "hardness": match["hardness"],
                "start_char": start,
                "end_char": start + len(matched_surface) if start >= 0 else -1,
                "clause": clause,
                "leakage_flags": sorted(flags),
            })

    aspect_sentiments = defaultdict(list)
    aspect_confidence = {}
    for span in spans:
        l = span["latent_label"]
        aspect_sentiments[l].append(span["sentiment"])
        aspect_confidence[l] = max(aspect_confidence.get(l, 0.0), span["confidence"])
    
    aspects = sorted(list(aspect_sentiments.keys())) or ["general"]
    max_hardness = max([s["hardness"] for s in spans]) if spans else 0
    final_label_type = "explicit" if any(s["label_type"] == "explicit" for s in spans) else "implicit"

    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "source_text": raw_text,
        "language": language,
        "track": mode,
        "implicit": {
            "mode": mode,
            "processed_text": processed_text,
            "aspects": aspects,
            "dominant_sentiment": sentiment,
            "aspect_sentiments": {a: Counter(s).most_common(1)[0][0] for a, s in aspect_sentiments.items()},
            "aspect_confidence": aspect_confidence,
            "spans": spans,
            "needs_review": not spans or aspects == ["general"],
            "implicit_ready": True,
            "llm_fallback_used": llm_fallback_used,
            "reasoned_recovery_used": reasoned_recovery_used,
            "label_type": final_label_type,
            "hardness_score": max_hardness,
            "hardness_tier": f"H{max_hardness}",
            "implicit_quality_tier": "strict_pass" if (spans and aspects != ["general"]) else "needs_review",
            "leakage_flags": sorted(leakage_flags_total),
        },
    }


def flush_llm_cache() -> None:
    """Flushes any local LLM response caches to ensure fresh research trials."""
    from llm_utils import GLOBAL_LLM_CACHE
    if GLOBAL_LLM_CACHE:
        GLOBAL_LLM_CACHE.clear()


class MultiAspectSynthesis:
    """v5.5 Hybrid Synthesis: Combines multiple implicit signals into a cohesive aspect set."""
    def __init__(self, llm_provider: Any = None) -> None:
        self.llm_provider = llm_provider

    def synthesize(self, text: str, initial_aspects: list[str]) -> list[str]:
        if not self.llm_provider or not initial_aspects:
            return initial_aspects
        # Synthesis logic for resolving conflicting or overlapping implicit signals
        return list(dict.fromkeys(initial_aspects))


class ResearchAblationMatrix:
    """Manages the 5-stage ablation matrix for ReviewOp research validation."""
    def __init__(self, run_profile: str = "research") -> None:
        self.run_profile = run_profile
        self.matrix = {
            "baseline": {"stage_b": False, "stage_c": False},
            "v5_hybrid": {"stage_b": True, "stage_c": True},
        }

    def get_config(self, stage: str) -> dict[str, bool]:
        return self.matrix.get(stage, self.matrix["v5_hybrid"])


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
    reasoned_recovery_count = 0
    reasoned_recovery_span_count = 0
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
        if implicit.get("reasoned_recovery_used"):
            reasoned_recovery_count += 1
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
            elif span.get("support_type") == "reasoned_recovery":
                reasoned_recovery_span_count += 1
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
        "span_support": {
            "exact": exact_span_count, 
            "near_exact": near_exact_span_count,
            "reasoned_recovery": reasoned_recovery_span_count
        },
        "reasoned_recovery_count": reasoned_recovery_count,
        "reasoned_recovery_hit_rate": round(reasoned_recovery_count / max(1, len(rows)), 4),
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
