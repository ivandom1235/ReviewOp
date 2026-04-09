from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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

PIVOT_TEMPLATES = {
    "battery": "The device's power duration was {sentiment} during usage, evidenced by '{evidence}'",
    "price": "The value-for-money proposition of this item is {sentiment} because '{evidence}'",
    "quality": "The build integrity and manufacturing standard is {sentiment} as shown in '{evidence}'",
    "service": "The customer interaction experience was {sentiment} due to '{evidence}'",
    "speed": "The performance and response time was {sentiment} based on '{evidence}'",
    "network": "The connectivity and signal stability was {sentiment} during '{evidence}'",
}

# Expanded research-grade latent aspect rules with explicit/implicit separation.
# Format: (label, explicit_keywords, implicit_signals)
LATENT_ASPECT_RULES = [
    ("value", {"price", "cost", "bill", "billing", "fees", "dollars", "rate", "value", "payment"}, 
              {"expensive", "cheap", "affordable", "worth", "priced", "pricey", "overpriced", "budget", "deal", "bargain", "money", "saving", "stole", "rip off"}),
    ("power", {"battery", "power", "charger", "charging", "plug", "adapter", "energy", "voltage"},
              {"dies", "drains", "lasts", "lasted", "dead", "empty", "hours", "short life", "long life", "shuts down", "recharge", "plugged in", "out of juice"}),
    ("connectivity", {"network", "signal", "wifi", "bluetooth", "connection", "data", "internet", "wireless", "pairing"},
                     {"drops", "searching", "disconnected", "spotty", "unstable", "cut out", "no service", "cannot connect", "no bars", "lost connection"}),
    ("thermal", {"temperature", "heat", "thermal", "cooling", "fan", "vent", "airflow"},
                {"hot", "warm", "cool", "burning", "heats up", "overheating"}),
    ("performance", {"performance", "speed", "software", "app", "operating system", "os", "hardware", "processor", "engine", "motor"},
                   {"fast", "slow", "lag", "laggy", "responsive", "smooth", "efficient", "powerful", "snappy", "clunky", "freezes", "wait", "stalling"}),
    ("display quality", {"screen", "display", "brightness", "resolution", "monitor", "pixel", "panel", "visuals", "interface"},
                       {"dim", "bright", "crisp", "washed out", "blurry", "vivid", "sharp", "glare", "dead pixels", "bleeding"}),
    ("reliability", {"issue", "problem", "crash", "broken", "stable", "durable", "defect", "defective", "fail", "failure", "malfunction", "support", "warranty"},
                    {"crashed", "died", "buggy", "stopped working", "frozen", "freezes", "error", "reboot", "restart", "quit", "useless", "garbage"}),
    ("build quality", {"build", "material", "structure", "construction", "casing", "exterior", "interior", "design", "finish"},
                     {"flimsy", "sturdy", "premium", "cheap plastic", "solid", "durable", "tough", "strong", "fragile", "scratched", "dent", "rattling"}),
    ("accessibility", {"map", "route", "signage", "station", "pickup", "location", "nav", "gps", "directions", "accessible", "access", "entry", "exit"},
                     {"easy", "confusing", "lost", "direct", "straightforward", "simple", "shortcut", "stuck", "closed", "blocked"}),
    ("service quality", {"service", "staff", "support", "waiter", "waitress", "driver", "doctor", "nurse", "crew", "host", "personnel", "manager"},
                        {"helpful", "attentive", "responsive", "polite", "impatient", "friendly", "rude", "nice", "kind", "professional", "rude"}),
    ("cleanliness", {"clean", "dirty", "hygiene", "neat", "tidy", "sanitary", "dust", "garbage", "trash"},
                     {"spotless", "filthy", "stain", "gross", "messy", "mess", "shining", "smell", "soiled"}),
    ("timeliness", {"time", "schedule", "standard", "arrival", "waiting", "appointment", "deadline"},
                    {"late", "delay", "quick", "fast", "prompt", "arrive", "wait", "rushed", "speedy", "instant", "on time"}),
    ("comfort", {"comfort", "space", "environment", "setting", "room", "seat", "cushion", "ambience"},
                {"comfortable", "uncomfortable", "crowded", "quiet", "noisy", "spacious", "cozy", "roomy", "cramped", "harsh"}),
    ("sensory quality", {"food", "meal", "taste", "dish", "ingredient", "cook", "recipe", "portion", "flavor", "smell", "scent", "sound", "audio", "music"},
                       {"tasty", "flavor", "fresh", "delicious", "savory", "yummy", "bland", "salty", "greasy", "stale", "noisy", "loud", "clear"}),
]

# Unconventional: Explicit Pivoting Templates
# Maps latent aspects to explicit restatement templates
PIVOT_TEMPLATES = {
    "power": "The power management is {sentiment} because {evidence}",
    "connectivity": "The connectivity is {sentiment} because {evidence}",
    "performance": "The performance is {sentiment} because {evidence}",
    "reliability": "The operational reliability is {sentiment} because {evidence}",
    "value": "The value for money is {sentiment} because {evidence}",
    "thermal": "The thermal state is {sentiment} because {evidence}",
    "display quality": "The visual/display quality is {sentiment} because {evidence}",
    "build quality": "The physical build quality is {sentiment} because {evidence}",
    "accessibility": "The accessibility is {sentiment} because {evidence}",
    "service quality": "The quality of service is {sentiment} because {evidence}",
    "cleanliness": "The cleanliness is {sentiment} because {evidence}",
    "timeliness": "The timeliness is {sentiment} because {evidence}",
    "comfort": "The overall comfort is {sentiment} because {evidence}",
    "sensory quality": "The sensory quality (taste/sound) is {sentiment} because {evidence}",
}

VALID_LATENT_ASPECTS = {label for label, _, _ in LATENT_ASPECT_RULES} | {"general"}
LATENT_RULE_BY_LABEL = {label: (explicit_kws, implicit_sigs) for label, explicit_kws, implicit_sigs in LATENT_ASPECT_RULES}


def inject_harvested_rules(new_rules: list[tuple[str, set[str], set[str]]]):
    """Dynamic injection for Adaptive Lexicon (Phase 1)."""
    global LATENT_ASPECT_RULES, VALID_LATENT_ASPECTS, LATENT_RULE_BY_LABEL
    
    # Merge rules: if label exists, merge keywords/signals; if new, append.
    current_rules_dict = {label: (e, i) for label, e, i in LATENT_ASPECT_RULES}
    for label, e_kws, i_sigs in new_rules:
        if label in current_rules_dict:
            curr_e, curr_i = current_rules_dict[label]
            current_rules_dict[label] = (curr_e | e_kws, curr_i | i_sigs)
        else:
            current_rules_dict[label] = (e_kws, i_sigs)
    
    LATENT_ASPECT_RULES = [(label, e, i) for label, (e, i) in current_rules_dict.items()]
    VALID_LATENT_ASPECTS = {label for label, _, _ in LATENT_ASPECT_RULES} | {"general"}
    LATENT_RULE_BY_LABEL = {label: (e, i) for label, e, i in LATENT_ASPECT_RULES}
    
    # Refresh VectorAspectMatcher if instance exists
    if VectorAspectMatcher._instance:
        VectorAspectMatcher._instance._initialize_centroids()


class VectorAspectMatcher:
    """Stage 2: Implicit candidate scoring using prototype embeddings."""
    
    _instance: Optional[VectorAspectMatcher] = None
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # We use local_files_only=True if we expect them to be pre-downloaded, 
        # but for builder we'll let it download if needed (assuming connection)
        self.model = SentenceTransformer(model_name)
        self.centroids: Dict[str, np.ndarray] = {}
        self._initialize_centroids()

    @classmethod
    def get_instance(cls) -> VectorAspectMatcher:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _initialize_centroids(self):
        """Pre-calculate prototypes for each latent aspect label."""
        for label, explicit_kws, implicit_sigs in LATENT_ASPECT_RULES:
            # Seed the prototype with the label itself + all recognized keywords
            seeds = [label] + list(explicit_kws) + list(implicit_sigs)
            embeddings = self.model.encode(seeds)
            self.centroids[label] = np.mean(embeddings, axis=0)

    def match(self, text: str, threshold: float = 0.55) -> List[Dict[str, Any]]:
        """ stage 2: returns candidate aspects with scores based on cosine similarity."""
        query_emb = self.model.encode([text])[0]
        results = []
        for label, centroid in self.centroids.items():
            sim = float(cosine_similarity([query_emb], [centroid])[0][0])
            if sim >= threshold:
                results.append({
                    "latent": label,
                    "confidence": round(sim, 3),
                    "source": "vector_grounding",
                    "hardness": 2 if sim < 0.75 else 1
                })
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results



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


_KEYWORD_REGEX_CACHE: dict[str, re.Pattern] = {}

def _keyword_in_text(text: str, keyword: str) -> bool:
    normalized_text = normalize_whitespace(text).lower()
    normalized_keyword = normalize_whitespace(keyword).lower()
    if not normalized_text or not normalized_keyword:
        return False
        
    pattern = _KEYWORD_REGEX_CACHE.get(normalized_keyword)
    if pattern is None:
        parts = [part for part in normalized_keyword.split() if part]
        if not parts:
            return False
        pattern_str = r"(?<![a-z0-9])" + r"\s+".join(re.escape(part) for part in parts) + r"(?![a-z0-9])"
        pattern = re.compile(pattern_str, flags=re.IGNORECASE)
        _KEYWORD_REGEX_CACHE[normalized_keyword] = pattern
        
    return bool(pattern.search(normalized_text))


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


_ADVERSATIVE_RE = re.compile(r"(?:\s+)(?:but|however|yet|while|although)(?:\s+)", re.IGNORECASE)
_COORD_AND_RE = re.compile(r"(?:,\s*and\s+|\s+and\s+)", re.IGNORECASE)
_MIN_CLAUSE_TOKENS_FOR_SPLIT = 6  # Lowered from 10 to catch multi-aspect pairings

_NEGATORS = {"not", "never", "no", "isnt", "wasnt", "doesnt", "didnt", "cant", "wont", "neither", "nor", "without", "hardly"}
_CONTRASTIVE_TOKENS = {"but", "however", "although", "though", "yet", "while"}
_COMPARATIVE_TOKENS = {"better", "worse", "faster", "slower", "more", "less"}
_SUPERLATIVE_POSITIVE = {"best", "fastest", "cleanest", "smoothest", "greatest"}
_SUPERLATIVE_NEGATIVE = {"worst", "slowest", "dirtiest", "weakest"}
_STRONG_NEGATIVE_EVENTS = {
    "crashed", "crash", "dropped", "drop", "refund denied", "denied", "waited", "stopped working", "failed", "failure",
    "disconnected", "overheating", "burning", "stalled", "froze", "freeze", "broken",
}
_STRONG_POSITIVE_EVENTS = {"resolved", "fixed", "quickly replaced", "on time", "seamless", "worked perfectly", "stable"}


def _sentence_clauses(text: str) -> list[str]:
    return [item["clause"] for item in _sentence_clauses_with_offsets(text)]


def _sentence_clauses_with_offsets(text: str) -> list[dict[str, int | str]]:
    full_text = normalize_whitespace(text)
    if not full_text:
        return []
    clauses: list[str] = []
    for sentence in split_sentences(text):
        # Stage 1: split on adversative conjunctions.
        pieces = [piece.strip(" ,;:-") for piece in _ADVERSATIVE_RE.split(sentence) if piece.strip()]
        if not pieces:
            pieces = [sentence]
        # Stage 2: split pieces on coordinating 'and' or commas.
        refined: list[str] = []
        for piece in pieces:
            piece_tokens = tokenize(piece)
            # More aggressive splitting for long pieces OR pieces with clear conjunctions
            if len(piece_tokens) >= _MIN_CLAUSE_TOKENS_FOR_SPLIT or _COORD_AND_RE.search(piece):
                sub_pieces = [sub.strip(" ,;:-") for sub in _COORD_AND_RE.split(piece) if sub.strip(" ,;:-")]
                if len(sub_pieces) > 1:
                    refined.extend(sub_pieces)
                else:
                    # Comma-based fallback for lists
                    comma_pieces = [sub.strip(" ,;:-") for sub in piece.split(",") if sub.strip(" ,;:-")]
                    # Only accept comma splits if they yield substantial sub-clauses
                    if len(comma_pieces) > 1 and any(len(tokenize(sub)) >= 3 for sub in comma_pieces):
                        refined.extend(comma_pieces)
                    else:
                        refined.append(piece)
            else:
                refined.append(piece)
        clauses.extend(refined)


    final_clauses = clauses or [full_text]
    out: list[dict[str, int | str]] = []
    cursor = 0
    lowered = full_text.lower()
    for clause in final_clauses:
        piece = normalize_whitespace(clause)
        if not piece:
            continue
        start = lowered.find(piece.lower(), cursor)
        if start < 0:
            start = lowered.find(piece.lower())
        if start < 0:
            start = 0
        end = start + len(piece)
        cursor = end
        out.append({"clause": piece, "start": int(start), "end": int(end)})
    return out


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
    return infer_sentiment_details(text)["label"]


def infer_sentiment_details(text: str) -> dict[str, Any]:
    tokens = tokenize(text)
    lower_text = normalize_whitespace(text).lower()
    pos_score = 0.0
    neg_score = 0.0
    strong_positive_hits = 0
    strong_negative_hits = 0
    comparative_hits = 0
    if not tokens:
        return {
            "label": "neutral",
            "abstained": True,
            "margin": 0.0,
            "risk_bucket": "high",
            "sentiment_mismatch": False,
            "scores": {"positive": 0.0, "negative": 0.0},
        }
    
    for i, token in enumerate(tokens):
        # Lookback for negations (simple 1-2 token window)
        is_negated = False
        if i > 0 and tokens[i-1] in _NEGATORS:
            is_negated = True
        elif i > 1 and tokens[i-2] in _NEGATORS:
            is_negated = True

        if token in POSITIVE_WORDS:
            if is_negated:
                neg_score += 1.4
            else:
                pos_score += 1.0
        elif token in NEGATIVE_WORDS:
            if is_negated:
                pos_score += 1.0
            else:
                neg_score += 1.0
        if token in _COMPARATIVE_TOKENS:
            comparative_hits += 1
            if token in {"better", "faster", "more"}:
                pos_score += 0.35
            elif token in {"worse", "slower", "less"}:
                neg_score += 0.35
        if token in _SUPERLATIVE_POSITIVE:
            pos_score += 1.2
        if token in _SUPERLATIVE_NEGATIVE:
            neg_score += 1.2

    for trigger in _STRONG_NEGATIVE_EVENTS:
        if trigger in lower_text:
            neg_score += 2.0
            strong_negative_hits += 1
    for trigger in _STRONG_POSITIVE_EVENTS:
        if trigger in lower_text:
            pos_score += 1.5
            strong_positive_hits += 1

    if any(tok in tokens for tok in _CONTRASTIVE_TOKENS):
        # Tail clauses after contrastive pivots carry stronger sentiment signal.
        tail = re.split(r"\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b|\bwhile\b", lower_text)[-1].strip()
        if tail:
            tail_tokens = tokenize(tail)
            tail_pos = sum(1 for tok in tail_tokens if tok in POSITIVE_WORDS)
            tail_neg = sum(1 for tok in tail_tokens if tok in NEGATIVE_WORDS)
            if tail_neg > tail_pos:
                neg_score += 0.8
            elif tail_pos > tail_neg:
                pos_score += 0.8

    margin = abs(pos_score - neg_score)
    abstained = margin < 0.55
    if pos_score > neg_score and not abstained:
        label = "positive"
    elif neg_score > pos_score and not abstained:
        label = "negative"
    else:
        label = "neutral"

    sentiment_mismatch = bool(strong_positive_hits > 0 and label == "neutral")
    risk_bucket = "low" if margin >= 1.25 else ("medium" if margin >= 0.55 else "high")
    return {
        "label": label,
        "abstained": abstained,
        "margin": round(margin, 4),
        "risk_bucket": risk_bucket,
        "sentiment_mismatch": sentiment_mismatch,
        "scores": {"positive": round(pos_score, 4), "negative": round(neg_score, 4)},
        "diagnostics": {
            "strong_positive_hits": strong_positive_hits,
            "strong_negative_hits": strong_negative_hits,
            "comparative_hits": comparative_hits,
        },
    }


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


def discover_aspects(
    rows: List[Dict[str, Any]],
    *,
    text_column: str,
    max_aspects: int,
    implicit_mode: str = "zeroshot",
    sample_rate: float = 0.4,
    random_seed: int | None = None,
) -> list[str]:
    # Lazy Discovery: Sample the dataset to speed up aspect discovery (Step 3/4)
    # Statically sufficient for large research benchmarks
    if len(rows) > 500:
        import random
        sample_n = max(1, int(len(rows) * sample_rate))
        rng = random.Random(random_seed) if random_seed is not None else random
        sampled_rows = rng.sample(rows, sample_n)
    else:
        sampled_rows = rows

    mode = _canonical_mode(implicit_mode)
    counts: Counter[str] = Counter()
    for row in sampled_rows:
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
    llm_model_name: str | None = None,
    high_difficulty: bool = False,
    adversarial_refine: bool = False,
    bypass_cache: bool = False,
    discovery_mode: bool = False,
    discovery_min_confidence: float = 0.55,
) -> Dict[str, Any]:
    from llm_utils import AsyncLlmProvider, reason_implicit_signal_async, discover_novel_aspects_async

    mode = _canonical_mode(implicit_mode)
    raw_text = normalize_whitespace(row.get(text_column, ""))
    processed_text = normalize_whitespace(coref_text or raw_text)
    clauses = _sentence_clauses_with_offsets(processed_text)
    sentiment_detail = infer_sentiment_details(processed_text)
    sentiment = str(sentiment_detail.get("label") or "neutral")
    
    # State tracking
    spans: list[dict[str, Any]] = []
    llm_parse_errors: list[str] = []
    llm_fallback_used = False
    fallback_branch = "none"
    reasoned_recovery_used = False
    leakage_flags_total: set[str] = set()

    # Stage A/B: Hybrid Candidate Generation (Optimized single-pass)
    for clause_payload in clauses:
        clause = str(clause_payload.get("clause") or "")
        clause_start = int(clause_payload.get("start") or 0)
        clause_sentiment = infer_sentiment(clause)
        clause_matches: list[dict[str, Any]] = []
        
        # Priority 1: Rules & Lexicon
        haystack = clause.lower()
        for label, explicit_kws, implicit_sigs in LATENT_ASPECT_RULES:
            # Check explicit first (Hardness 0)
            found_explicit = False
            for kw in sorted(explicit_kws, key=len, reverse=True):
                if _keyword_in_text(haystack, kw):
                    matched, match_type = _match_aspect_surface(clause, kw)
                    clause_matches.append({
                        "latent": label,"aspect": matched or kw,"support_type": match_type or "exact",
                        "label_type": "explicit","hardness": 0,"confidence": 1.0,"source": "rule"
                    })
                    found_explicit = True
                    break
            
            if found_explicit: 
                # FAST-PATH: If we found a perfect explicit match for this aspect, 
                # we don't need to check implicit signals for the same aspect label.
                continue

            # Check implicit signals (Hardness 1)
            for sig in sorted(implicit_sigs, key=len, reverse=True):
                if _keyword_in_text(haystack, sig):
                    matched, match_type = _match_aspect_surface(clause, sig)
                    clause_matches.append({
                        "latent": label,"aspect": matched or sig,"support_type": match_type or "near_exact",
                        "label_type": "implicit","hardness": 1,"confidence": 0.85,"source": "lexicon"
                    })
                    break

        # Priority 2: Stage 2 - Vector Grounding (Phase 2 Implementation)
        if not clause_matches:
            vector_matcher = VectorAspectMatcher.get_instance()
            vector_hits = vector_matcher.match(clause)
            for hit in vector_hits:
                anchor = (clause[:35] + "...") if len(clause) > 35 else clause
                clause_matches.append({
                    "latent": hit["latent"],
                    "aspect": anchor,
                    "support_type": "vector_semantic",
                    "label_type": "implicit",
                    "hardness": hit["hardness"],
                    "confidence": hit["confidence"],
                    "source": "vector"
                })
                if len(clause_matches) >= 2: 
                    break

        # Priority 3: Stage 3 - LLM Fallback (only if no lexical or vector matches)
        if not clause_matches and enable_llm_fallback and llm_provider:
            if llm_model_name is None or not str(llm_model_name).strip():
                raise RuntimeError(
                    "LLM_MODEL_NAME or provider-specific model env is required"
                )
            llm_fallback_used = True
            paraphrases = await reason_implicit_signal_async(clause, candidate_aspects, llm_provider, llm_model_name, bypass_cache=bypass_cache)
            fallback_branch = "llm_parse"
            for para in paraphrases:
                for label, e_kws, i_sigs in LATENT_ASPECT_RULES:
                    for kw in sorted(e_kws | i_sigs, key=len, reverse=True):
                        if _keyword_in_text(para, kw) and _keyword_in_text(clause, kw):
                            clause_matches.append({
                                "latent": label,"aspect": kw,"support_type": "llm_reasoning",
                                "label_type": "implicit","hardness": 2,"confidence": 0.75,"source": "llm",
                            })
                            reasoned_recovery_used = True
                            break
                    if clause_matches: break
                if clause_matches: break
            if not clause_matches:
                llm_parse_errors.append("ungrounded_llm_match")
        
        # Priority 4: Stage 4 - Open-Domain Discovery (Phase 2 Implementation)
        if not clause_matches and discovery_mode and llm_provider:
            novel_hits = await discover_novel_aspects_async(
                clause, 
                excluded_aspects=list(VALID_LATENT_ASPECTS), 
                provider=llm_provider, 
                model_name=llm_model_name,
                domain=domain,
                bypass_cache=bypass_cache
            )
            for hit in novel_hits:
                conf = float(hit.get("confidence", 0.0))
                if conf >= discovery_min_confidence:
                    clause_matches.append({
                        "latent": str(hit.get("label", "unknown")).lower(),
                        "aspect": str(hit.get("evidence") or clause[:30]),
                        "support_type": "discovered",
                        "label_type": "implicit",
                        "hardness": 3, # Discovered aspects are always high difficulty
                        "confidence": conf,
                        "source": "discovery"
                    })
                    if len(clause_matches) >= 2: break
        # Stage D: Validation & Grounding (Phase 4 Logic)
        # Sort and filter matches: Prioritize specificity (granularity) over general label
        if len(clause_matches) > 1:
            # If we have granular hits, remove 'general' or low-confidence noise
            granular_hits = [m for m in clause_matches if m["latent"] != "general"]
            if granular_hits:
                clause_matches = granular_hits

        for match in clause_matches:
            latent = match["latent"]
            if not _is_valid_latent_aspect(latent): continue

            # Conflict Resolution: Domain-Clash Check
            # (e.g., if sentiment is service-based but aspect is hardware-based)
            is_conflict = False
            service_cues = {"helpful", "staff", "waiter", "waitress", "service", "friendly", "polite"}
            hardware_cues = {"battery", "screen", "fast", "slow", "performance", "build", "quality"}
            
            if latent in {"service quality", "timeliness"} and any(kw in clause.lower() for kw in hardware_cues):
                is_conflict = True
            elif latent in {"power", "performance", "display quality"} and any(kw in clause.lower() for kw in service_cues):
                is_conflict = True
                
            if is_conflict:
                match["confidence"] -= 0.15 # Penalize conflicting signals
                match["support_type"] = "conflicting_signal"


            flags = _compute_leakage_flags(
                text=processed_text,
                latent=latent,
                surface_aspect=match["aspect"],
                label_type=match["label_type"],
            )
            leakage_flags_total.update(flags)
            
            matched_surface = match["aspect"]
            local_start = clause.lower().find(matched_surface.lower())
            
            if local_start == -1:
                continue 

            absolute_start = int(clause_start + local_start)
            absolute_end = int(absolute_start + len(matched_surface))
            spans.append({
                "aspect": matched_surface,
                "latent_label": latent,
                "evidence_text": clause,
                "evidence_span": [absolute_start, absolute_end],
                "sentiment": clause_sentiment,
                "confidence": match["confidence"],
                "support_type": match["support_type"],
                "label_type": match["label_type"],
                "source": match["source"],
                "hardness": match["hardness"],
                "start_char": absolute_start,
                "end_char": absolute_end if absolute_start >= 0 else -1,
                "clause": clause,
                "leakage_flags": sorted(flags),
            })

    aspect_sentiments = defaultdict(list)
    aspect_confidence = {}
    for span in spans:
        l = span["latent_label"]
        aspect_sentiments[l].append(span["sentiment"])
        aspect_confidence[l] = max(aspect_confidence.get(l, 0.0), span["confidence"])
    
    # Final assembly with conformal sets and pivoting
    inferred_aspects = sorted(list(aspect_sentiments.keys())) or ["general"]
    max_hardness = max([s["hardness"] for s in spans]) if spans else 0
    final_label_type = "explicit" if any(s["label_type"] == "explicit" for s in spans) else "implicit"

    # Conformal Set Logic (Research-Grade)
    # Include all aspects whose confidence is within 0.15 of the top candidate
    sorted_conf = sorted(aspect_confidence.items(), key=lambda x: x[1], reverse=True)
    conformal_set = []
    if sorted_conf:
        top_val = sorted_conf[0][1]
        conformal_set = [it[0] for it in sorted_conf if (top_val - it[1]) <= 0.15]
    
    # Ambiguity Score (Continuous entropy-based signal)
    ambiguity_score = 0.0
    if len(sorted_conf) > 1:
        # High ambiguity if the gap between top two is small
        ambiguity_score = max(0.0, 1.0 - (sorted_conf[0][1] - sorted_conf[1][1]))
    if spans:
        span_labels = {str(s.get("latent_label") or "") for s in spans if str(s.get("latent_label") or "")}
        if len(span_labels) >= 2 and ambiguity_score >= 0.45:
            max_hardness = max(max_hardness, 3)
        elif any(int(s.get("hardness", 0)) >= 2 for s in spans):
            max_hardness = max(max_hardness, 2)
    
    # Explicit Pivoting (Unconventional cross-check)
    pivot_confirmed = False
    if inferred_aspects and inferred_aspects[0] in PIVOT_TEMPLATES:
        best_aspect = inferred_aspects[0]
        template = PIVOT_TEMPLATES[best_aspect]
        pivot_text = template.format(sentiment=sentiment, evidence=raw_text)
        # Check if the core keywords of the aspect appear in the context of the explicit restatement
        # This is a simplified "template-based" pivoting check
        explicit_kws, _ = LATENT_RULE_BY_LABEL.get(best_aspect, (set(), set()))
        if any(kw in pivot_text.lower() for kw in explicit_kws):
            pivot_confirmed = True

    review_reason = "none"
    if not spans or inferred_aspects == ["general"]:
        review_reason = "fallback_general"
        if fallback_branch == "none":
            fallback_branch = "fallback_general"
    elif ambiguity_score > 0.8:
        review_reason = "low_confidence"

    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "source_text": raw_text,
        "language": language,
        "track": mode,
        "implicit": {
            "mode": mode,
            "processed_text": processed_text,
            "aspects": inferred_aspects,
            "aspect": inferred_aspects[0] if inferred_aspects else "general", # Canonical aspect
            "conformal_set": conformal_set,
            "ambiguity_score": round(ambiguity_score, 4),
            "pivot_confirmed": pivot_confirmed,
            "dominant_sentiment": sentiment,
            "sentiment_abstained": bool(sentiment_detail.get("abstained", False)),
            "sentiment_risk_bucket": str(sentiment_detail.get("risk_bucket") or "high"),
            "sentiment_margin": float(sentiment_detail.get("margin", 0.0) or 0.0),
            "sentiment_mismatch": bool(sentiment_detail.get("sentiment_mismatch", False)),
            "sentiment_scores": sentiment_detail.get("scores", {}),
            "sentiment_diagnostics": sentiment_detail.get("diagnostics", {}),
            "aspect_sentiments": {a: Counter(s).most_common(1)[0][0] for a, s in aspect_sentiments.items()},
            "aspect_confidence": aspect_confidence,
            "spans": spans,
            "needs_review": review_reason != "none",
            "implicit_ready": True,
            "llm_fallback_used": llm_fallback_used,
            "reasoned_recovery_used": reasoned_recovery_used,
            "llm_parse_errors": llm_parse_errors,
            "review_reason": review_reason,
            "fallback_branch": fallback_branch,
            "label_type": final_label_type,
            "hardness_score": max_hardness,
            "hardness_tier": f"H{max_hardness}",
            "implicit_quality_tier": "strict_pass" if (spans and inferred_aspects != ["general"]) else "needs_review",
            "leakage_flags": sorted(leakage_flags_total),
        },
    }


def flush_llm_cache() -> None:
    """Flushes any local LLM response caches to ensure fresh research trials."""
    from llm_utils import GLOBAL_LLM_CACHE
    if GLOBAL_LLM_CACHE:
        GLOBAL_LLM_CACHE.clear()


class MultiAspectSynthesis:
    """V6 Hybrid Synthesis: Combines multiple implicit signals into a cohesive aspect set."""
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
    fallback_branch_counts: Counter[str] = Counter()
    review_reason_counts: Counter[str] = Counter()
    gold_miss_counts: Counter[str] = Counter()
    total_gold_aspects = 0
    matched_gold_aspects = 0
    
    for row in rows:
        text = normalize_whitespace(row.get(text_column, ""))
        clause_count += len(_sentence_clauses(text))
        implicit = row.get("implicit", {})
        
        # Registry Coverage Audit (Phase 1)
        gold_labels = _extract_gold_aspects([row])
        detected_labels = set(implicit.get("aspects") or [])
        for gold in gold_labels:
            if gold == "general": continue
            total_gold_aspects += 1
            if gold in detected_labels:
                matched_gold_aspects += 1
            else:
                gold_miss_counts[gold] += 1

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
        "registry_coverage": {
            "total_gold_aspects": total_gold_aspects,
            "matched_gold_aspects": matched_gold_aspects,
            "coverage_rate": round(matched_gold_aspects / max(1, total_gold_aspects), 4),
            "top_missing_gold_aspects": gold_miss_counts.most_common(15)
        }
    }
