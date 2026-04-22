from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from utils.utils import normalize_whitespace, split_sentences, tokenize
except ImportError:
    from .utils.utils import normalize_whitespace, split_sentences, tokenize

from row_contracts import Prepared, Interpretation, GroundedInterpretation, Grounded, ImplicitScored

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
    global LATENT_ASPECT_RULES, VALID_LATENT_ASPECTS, LATENT_RULE_BY_LABEL
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
    if VectorAspectMatcher._instance:
        VectorAspectMatcher._instance._initialize_centroids()

class VectorAspectMatcher:
    _instance: Optional[VectorAspectMatcher] = None
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.centroids: Dict[str, np.ndarray] = {}
        self._initialize_centroids()
    @classmethod
    def get_instance(cls) -> VectorAspectMatcher:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    def _initialize_centroids(self):
        for label, explicit_kws, implicit_sigs in LATENT_ASPECT_RULES:
            seeds = [label] + list(explicit_kws) + list(implicit_sigs)
            embeddings = self.model.encode(seeds)
            self.centroids[label] = np.mean(embeddings, axis=0)
    def match(self, text: str, threshold: float = 0.55) -> List[Dict[str, Any]]:
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
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

def _canonical_mode(mode: str) -> str:
    mode = str(mode or "").strip().lower()
    return {"heuristic": "zeroshot", "benchmark": "supervised", "zero-shot": "zeroshot"}.get(mode, mode or "zeroshot")

def _canonicalize_label(value: Any) -> str | None:
    """Map a raw string to the official ontology labels via direct match or synonyms."""
    if not isinstance(value, str): return None
    clean = value.strip().lower()
    if clean in VALID_LATENT_ASPECTS: return clean
    
    # Synonym mapping
    syns = {
        "price": "value", "cost": "value", "expensive": "value", "cheap": "value",
        "battery": "power", "charging": "power", "power": "power",
        "internet": "connectivity", "wifi": "connectivity", "network": "connectivity", "signal": "connectivity",
        "speed": "performance", "lag": "performance", "performance": "performance",
        "screen": "display quality", "brightness": "display quality",
        "service": "service quality", "staff": "service quality", "support": "service quality",
        "dirty": "cleanliness", "hygiene": "cleanliness",
        "fast": "timeliness", "late": "timeliness", "delay": "timeliness",
        "taste": "sensory quality", "flavor": "sensory quality", "sound": "sensory quality", "audio": "sensory quality"
    }
    if clean in syns: return syns[clean]
    
    # Fuzzy match
    for label in VALID_LATENT_ASPECTS:
        if label != "general" and (label in clean or clean in label):
            return label
    return None

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

def _compute_leakage_flags(*, text: str, latent: str, surface_aspect: str, label_type: str) -> list[str]:
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
    if len(token) <= 3: return token
    if token.endswith("ies") and len(token) > 4: return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4: return token[:-2]
    if token.endswith("ing") and len(token) > 5: return token[:-3]
    if token.endswith("ed") and len(token) > 4: return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3: return token[:-1]
    return token

def _match_aspect_surface(text: str, aspect: str) -> tuple[str | None, str | None]:
    aspect = _normalize_aspect(aspect)
    if not aspect: return None, None
    match = re.search(rf"\b{re.escape(aspect)}\b", text, flags=re.IGNORECASE)
    if match: return match.group(0), "exact"
    stemmed = " ".join(_stem_token(token) for token in aspect.split())
    if stemmed != aspect:
        match = re.search(rf"\b{re.escape(stemmed)}\b", text, flags=re.IGNORECASE)
        if match: return match.group(0), "near_exact"
    aspect_tokens = aspect.split()
    if len(aspect_tokens) > 1:
        for token in aspect_tokens:
            if token in STOP_TOKENS or len(token) < 3: continue
            for variant in {token, _stem_token(token)}:
                match = re.search(rf"\b{re.escape(variant)}\b", text, flags=re.IGNORECASE)
                if match: return match.group(0), "near_exact"
    return None, None

_ADVERSATIVE_RE = re.compile(r"(?:\s+)(?:but|however|yet|while|although)(?:\s+)", re.IGNORECASE)
_COORD_AND_RE = re.compile(r"(?:,\s*and\s+|\s+and\s+)", re.IGNORECASE)
_MIN_CLAUSE_TOKENS_FOR_SPLIT = 6
_NEGATORS = {"not", "never", "no", "isnt", "wasnt", "doesnt", "didnt", "cant", "wont", "neither", "nor", "without", "hardly"}
_CONTRASTIVE_TOKENS = {"but", "however", "although", "though", "yet", "while"}
_COMPARATIVE_TOKENS = {"better", "worse", "faster", "slower", "more", "less"}
_SUPERLATIVE_POSITIVE = {"best", "fastest", "cleanest", "smoothest", "greatest"}
_SUPERLATIVE_NEGATIVE = {"worst", "slowest", "dirtiest", "weakest"}
_STRONG_NEGATIVE_EVENTS = {"crashed", "crash", "dropped", "drop", "refund denied", "denied", "waited", "stopped working", "failed", "failure", "disconnected", "overheating", "burning", "stalled", "froze", "freeze", "broken"}
_STRONG_POSITIVE_EVENTS = {"resolved", "fixed", "quickly replaced", "on time", "seamless", "worked perfectly", "stable"}

def _sentence_clauses(text: str) -> list[str]:
    return [item["clause"] for item in _sentence_clauses_with_offsets(text)]

def _sentence_clauses_with_offsets(text: str) -> list[dict[str, int | str]]:
    full_text = normalize_whitespace(text)
    if not full_text: return []
    clauses: list[str] = []
    for sentence in split_sentences(text):
        pieces = [piece.strip(" ,;:-") for piece in _ADVERSATIVE_RE.split(sentence) if piece.strip()]
        if not pieces: pieces = [sentence]
        refined: list[str] = []
        for piece in pieces:
            piece_tokens = tokenize(piece)
            if len(piece_tokens) >= _MIN_CLAUSE_TOKENS_FOR_SPLIT or _COORD_AND_RE.search(piece):
                sub_pieces = [sub.strip(" ,;:-") for sub in _COORD_AND_RE.split(piece) if sub.strip(" ,;:-")]
                if len(sub_pieces) > 1: refined.extend(sub_pieces)
                else:
                    comma_pieces = [sub.strip(" ,;:-") for sub in piece.split(",") if sub.strip(" ,;:-")]
                    if len(comma_pieces) > 1 and any(len(tokenize(sub)) >= 3 for sub in comma_pieces): refined.extend(comma_pieces)
                    else: refined.append(piece)
            else: refined.append(piece)
        clauses.extend(refined)
    final_clauses = clauses or [full_text]
    out: list[dict[str, int | str]] = []
    cursor = 0
    lowered = full_text.lower()
    for clause in final_clauses:
        piece = normalize_whitespace(clause)
        if not piece: continue
        start = lowered.find(piece.lower(), cursor)
        if start < 0: start = lowered.find(piece.lower())
        if start < 0: start = 0
        end = start + len(piece)
        cursor = end
        out.append({"clause": piece, "start": int(start), "end": int(end)})
    return out

def infer_sentiment(text: str) -> str:
    return infer_sentiment_details(text)["label"]

def infer_sentiment_details(text: str) -> dict[str, Any]:
    tokens = tokenize(text)
    lower_text = normalize_whitespace(text).lower()
    pos_score, neg_score = 0.0, 0.0
    strong_positive_hits, strong_negative_hits, comparative_hits = 0, 0, 0
    if not tokens: return {"label": "neutral", "abstained": True, "margin": 0.0, "risk_bucket": "high", "sentiment_mismatch": False, "scores": {"positive": 0.0, "negative": 0.0}}
    for i, token in enumerate(tokens):
        is_negated = False
        if i > 0 and tokens[i-1] in _NEGATORS: is_negated = True
        elif i > 1 and tokens[i-2] in _NEGATORS: is_negated = True
        if token in POSITIVE_WORDS:
            if is_negated: neg_score += 1.4
            else: pos_score += 1.0
        elif token in NEGATIVE_WORDS:
            if is_negated: pos_score += 1.0
            else: neg_score += 1.0
        if token in _COMPARATIVE_TOKENS:
            comparative_hits += 1
            if token in {"better", "faster", "more"}: pos_score += 0.35
            elif token in {"worse", "slower", "less"}: neg_score += 0.35
        if token in _SUPERLATIVE_POSITIVE: pos_score += 1.2
        if token in _SUPERLATIVE_NEGATIVE: neg_score += 1.2
    for trigger in _STRONG_NEGATIVE_EVENTS:
        if trigger in lower_text: neg_score += 2.0; strong_negative_hits += 1
    for trigger in _STRONG_POSITIVE_EVENTS:
        if trigger in lower_text: pos_score += 1.5; strong_positive_hits += 1
    if any(tok in tokens for tok in _CONTRASTIVE_TOKENS):
        tail = re.split(r"\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b|\bwhile\b", lower_text)[-1].strip()
        if tail:
            tail_tokens = tokenize(tail)
            tail_pos = sum(1 for tok in tail_tokens if tok in POSITIVE_WORDS)
            tail_neg = sum(1 for tok in tail_tokens if tok in NEGATIVE_WORDS)
            if tail_neg > tail_pos: neg_score += 0.8
            elif tail_pos > tail_neg: pos_score += 0.8
    margin = abs(pos_score - neg_score)
    abstained = margin < 0.55
    label = "positive" if pos_score > neg_score and not abstained else ("negative" if neg_score > pos_score and not abstained else "neutral")
    risk_bucket = "low" if margin >= 1.25 else ("medium" if margin >= 0.55 else "high")
    return {"label": label, "abstained": abstained, "margin": round(margin, 4), "risk_bucket": risk_bucket, "sentiment_mismatch": bool(strong_positive_hits > 0 and label == "neutral"), "scores": {"positive": round(pos_score, 4), "negative": round(neg_score, 4)}, "diagnostics": {"strong_positive_hits": strong_positive_hits, "strong_negative_hits": strong_negative_hits, "comparative_hits": comparative_hits}}

def _extract_gold_aspects(rows: List[Dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key in ("aspect", "gold_aspect", "target_aspect"):
            value = row.get(key)
            if isinstance(value, str) and _is_meaningful_aspect(value): counts[_normalize_aspect(value)] += 1
        if isinstance(row.get("gold_labels"), list):
            for label in row["gold_labels"]:
                if isinstance(label, dict):
                    for key in ("aspect", "text", "implicit_aspect"):
                        value = label.get(key)
                        if isinstance(value, str) and _is_meaningful_aspect(value): counts[_normalize_aspect(value)] += 1
    return [aspect for aspect, _ in counts.most_common()]

def discover_aspects(rows: List[Dict[str, Any]], *, text_column: str, max_aspects: int, implicit_mode: str = "zeroshot", sample_rate: float = 0.4, random_seed: int | None = None) -> list[str]:
    if len(rows) > 500:
        import random
        sample_n = max(1, int(len(rows) * sample_rate))
        rng = random.Random(random_seed) if random_seed is not None else random
        sampled_rows = rng.sample(rows, sample_n)
    else: sampled_rows = rows
    mode = _canonical_mode(implicit_mode)
    counts: Counter[str] = Counter()
    for row in sampled_rows:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text: continue
        for clause in _sentence_clauses(text):
            for label, e_kws, i_sigs in LATENT_ASPECT_RULES:
                haystack = clause.lower()
                if any(_keyword_in_text(haystack, keyword) for keyword in e_kws | i_sigs): counts[label] += 1
        if mode in {"supervised", "hybrid"}:
            for aspect in _extract_gold_aspects([row]):
                label = _latent_aspect_label(aspect, text)
                if label != "general": counts[label] += 1
    aspects = [aspect for aspect, _ in counts.most_common(max_aspects) if _is_valid_latent_aspect(aspect)]
    return aspects or ["general"]

async def build_implicit_row(
    row: Prepared | dict[str, Any],
    *,
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
    llm_provider: Any = None,
    llm_model_name: str | None = None,
    high_difficulty: bool = False,
    adversarial_refine: bool = False,
    bypass_cache: bool = False,
    discovery_mode: bool = False,
    discovery_min_confidence: float = 0.55,
) -> Grounded | ImplicitScored:
    try:
        from .llm_utils import reason_implicit_signal_async, discover_novel_aspects_async, AsyncRunPodProvider
    except ImportError:
        from llm_utils import reason_implicit_signal_async, discover_novel_aspects_async, AsyncRunPodProvider

    mode = _canonical_mode(implicit_mode)
    raw_text = row.get("review_text") or row.get("text") or "" if isinstance(row, dict) else row.review_text
    processed_text = normalize_whitespace(coref_text or raw_text)
    clauses = _sentence_clauses_with_offsets(processed_text)
    sentiment_detail = infer_sentiment_details(processed_text)
    sentiment = str(sentiment_detail.get("label") or "neutral")
    spans: list[dict[str, Any]] = []
    llm_fallback_used = False
    reasoned_recovery_used = False
    leakage_flags_total: set[str] = set()
    effective_llm_model_name = str(llm_model_name or "")

    for clause_payload in clauses:
        clause = str(clause_payload.get("clause") or "")
        clause_start = int(clause_payload.get("start") or 0)
        clause_sentiment = infer_sentiment(clause)
        clause_matches: list[dict[str, Any]] = []
        haystack = clause.lower()
        for label, explicit_kws, implicit_sigs in LATENT_ASPECT_RULES:
            found_explicit = False
            for kw in sorted(explicit_kws, key=len, reverse=True):
                if _keyword_in_text(haystack, kw):
                    matched, match_type = _match_aspect_surface(clause, kw)
                    clause_matches.append({"latent": label, "aspect": matched or kw, "support_type": match_type or "exact", "label_type": "explicit", "hardness": 0, "confidence": 1.0, "source": "rule"})
                    found_explicit = True; break
            if found_explicit: continue
            for sig in sorted(implicit_sigs, key=len, reverse=True):
                if _keyword_in_text(haystack, sig):
                    matched, match_type = _match_aspect_surface(clause, sig)
                    clause_matches.append({"latent": label, "aspect": matched or sig, "support_type": match_type or "near_exact", "label_type": "implicit", "hardness": 1, "confidence": 0.85, "source": "lexicon"})
                    break
        if not clause_matches:
            vector_matcher = VectorAspectMatcher.get_instance()
            for hit in vector_matcher.match(clause):
                clause_matches.append({"latent": hit["latent"], "aspect": (clause[:35] + "...") if len(clause)>35 else clause, "support_type": "vector_semantic", "label_type": "implicit", "hardness": hit["hardness"], "confidence": hit["confidence"], "source": "vector"})
                if len(clause_matches) >= 2: break
        if not clause_matches and enable_llm_fallback and llm_provider:
            llm_fallback_used = True
            predicted = await reason_implicit_signal_async(clause, candidate_aspects, llm_provider, effective_llm_model_name, bypass_cache=bypass_cache)
            for pred in predicted:
                pred_label = _canonicalize_label(pred)
                if pred_label:
                    matched_surface = next((kw for kw in sorted(LATENT_RULE_BY_LABEL.get(pred_label, (set(), set()))[0] | LATENT_RULE_BY_LABEL.get(pred_label, (set(), set()))[1], key=len, reverse=True) if _keyword_in_text(clause, kw)), None)
                    clause_matches.append({"latent": pred_label, "aspect": matched_surface or clause[:30], "support_type": "llm_reasoning", "label_type": "implicit", "hardness": 2, "confidence": 0.75, "source": "llm"})
                    reasoned_recovery_used = True
        
        if not clause_matches and discovery_mode and llm_provider:
            novel_hits = await discover_novel_aspects_async(clause, excluded_aspects=list(VALID_LATENT_ASPECTS), provider=llm_provider, model_name=effective_llm_model_name, domain=domain, bypass_cache=bypass_cache)
            for hit in novel_hits:
                conf = float(hit.get("confidence", 0.0))
                if conf >= discovery_min_confidence:
                    raw_label = str(hit.get("label", "unknown")).lower()
                    # Try to canonicalise even discovered aspects
                    canon_label = _canonicalize_label(raw_label)
                    clause_matches.append({
                        "latent": canon_label or raw_label, 
                        "aspect": str(hit.get("evidence") or clause[:30]), 
                        "support_type": "discovered", 
                        "label_type": "implicit", 
                        "hardness": 3, 
                        "confidence": conf, 
                        "source": "discovery"
                    })
                    if len(clause_matches) >= 2: break

        for match in clause_matches:
            latent = match["latent"]
            if not _is_valid_latent_aspect(latent): continue
            flags = _compute_leakage_flags(text=processed_text, latent=latent, surface_aspect=match["aspect"], label_type=match["label_type"])
            leakage_flags_total.update(flags)
            matched_surface = match["aspect"]
            local_start = clause.lower().find(matched_surface.lower())
            if local_start == -1: continue
            absolute_start = int(clause_start + local_start)
            absolute_end = int(absolute_start + len(matched_surface))
            if absolute_start < 0 or absolute_end <= absolute_start or absolute_end > len(processed_text):
                continue
            spans.append({"aspect": matched_surface, "latent_label": latent, "evidence_text": clause, "evidence_span": [absolute_start, absolute_end], "sentiment": clause_sentiment, "confidence": match["confidence"], "support_type": match["support_type"], "label_type": match["label_type"], "source": match["source"], "hardness": match["hardness"], "leakage_flags": sorted(flags), "span_quality": "exact_sentence_match"})

    # Purity Filter: If an aspect was detected as both explicit and implicit in the SAME row,
    # and the explicit evidence is strong, we prefer explicit and may suppress the 'implicit' tag 
    # to maintain high implicit purity stats.
    final_spans = []
    explicit_latents = {s["latent_label"] for s in spans if s["label_type"] == "explicit"}
    for span in spans:
        if span["label_type"] == "implicit" and span["latent_label"] in explicit_latents:
            # If we found it explicitly, it's not a 'pure' implicit signal in this row
            continue
        final_spans.append(span)
    spans = final_spans

    aspect_sentiments = defaultdict(list)
    aspect_confidence = {}
    for span in spans:
        l = span["latent_label"]
        aspect_sentiments[l].append(span["sentiment"])
        aspect_confidence[l] = max(aspect_confidence.get(l, 0.0), span["confidence"])
    
    interpretations = []
    for span in spans:
        if span.get("aspect") and span.get("evidence_span"):
            interpretations.append(GroundedInterpretation(
                aspect=span["latent_label"], 
                sentiment=span["sentiment"],
                evidence_text=str(span["aspect"]), 
                evidence_span=span["evidence_span"], 
                interpretation_type="implicit" if span["label_type"] == "implicit" else "explicit", 
                confidence=span["confidence"]
            ))
        else:
            interpretations.append(Interpretation(
                aspect=span["latent_label"], 
                sentiment=span["sentiment"],
                interpretation_type="implicit" if span["label_type"] == "implicit" else "explicit", 
                confidence=span["confidence"]
            ))
    
    row_id = row.get("row_id") or row.get("id") or "unknown" if isinstance(row, dict) else row.row_id
    domain_val = row.get("domain") if isinstance(row, dict) else row.domain
    group_id = row.get("group_id") if isinstance(row, dict) else row.group_id

    # V7 Model instance
    if all(isinstance(i, GroundedInterpretation) for i in interpretations) and interpretations:
        v7_result = Grounded(row_id=row_id, review_text=raw_text, domain=domain_val, group_id=group_id, interpretations=interpretations)
    else:
        v7_result = ImplicitScored(row_id=row_id, review_text=raw_text, domain=domain_val, group_id=group_id, interpretations=interpretations)

    # Aggregate hardness
    max_h = max([s.get("hardness", 0) for s in spans]) if spans else 0
    hardness_tier = f"H{max_h}"

    # Convert to V6-compatible dict for orchestration script
    return {
        "implicit": {
            "aspects": [i.aspect for i in interpretations],
            "sentiments": [i.sentiment for i in interpretations],
            "spans": spans,
            "hardness_tier": hardness_tier,
            "v7_model": v7_result.model_dump(),
        },
        "track": {
            "source": "v7_hybrid",
            "confidence": max([i.confidence for i in interpretations]) if interpretations else 0.0,
            "llm_fallback_used": llm_fallback_used,
            "reasoned_recovery_used": reasoned_recovery_used,
        }
    }

def collect_diagnostics(rows: List[Dict[str, Any]], *, text_column: str, candidate_aspects: List[str]) -> Dict[str, Any]:
    aspect_counts, language_counts, track_counts = Counter(), Counter(), Counter()
    exact_span_count, near_exact_span_count, fallback_only_count, total_gold_aspects, matched_gold_aspects = 0, 0, 0, 0, 0
    for row in rows:
        text = normalize_whitespace(row.get(text_column, ""))
        implicit = row.get("implicit", {})
        gold_labels = _extract_gold_aspects([row])
        detected_labels = set(implicit.get("aspects") or [])
        for gold in gold_labels:
            if gold == "general": continue
            total_gold_aspects += 1
            if gold in detected_labels: matched_gold_aspects += 1
        language_counts[str(row.get("language", "unknown"))] += 1
        if implicit.get("aspects") == ["general"]: fallback_only_count += 1
        for aspect in implicit.get("aspects", []):
            if aspect != "general": aspect_counts[str(aspect)] += 1
        for span in implicit.get("spans", []):
            if span.get("support_type") == "exact": exact_span_count += 1
            elif span.get("support_type") == "near_exact": near_exact_span_count += 1
    return {"top_implicit_aspects": aspect_counts.most_common(10), "clause_count": 0, "span_support": {"exact": exact_span_count, "near_exact": near_exact_span_count}, "fallback_only_count": fallback_only_count, "registry_coverage": {"coverage_rate": round(matched_gold_aspects / max(1, total_gold_aspects), 4)}}
