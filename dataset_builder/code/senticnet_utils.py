from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCE_PATH = PROJECT_ROOT / "resources" / "senticnet_seed.json"


_STATE: Dict[str, Any] = {
    "enabled": True,
    "resource_path": DEFAULT_RESOURCE_PATH,
    "entries": [],
    "loaded": False,
    "source": "uninitialized",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _token_variants(token: str) -> set[str]:
    token = token.strip().lower()
    if not token:
        return set()
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    token = token.replace("-", "_")
    variants.add(token)
    return {item for item in variants if item}


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in re.findall(r"[a-z0-9_]+", _normalize(text).replace("-", "_")):
        if len(raw_token) <= 2:
            continue
        tokens.update(item for item in _token_variants(raw_token) if len(item) > 2)
    return tokens


def _seed_entries() -> List[Dict[str, Any]]:
    return [
        {"concept": "too expensive", "aspect": "value", "polarity": -0.82, "aliases": ["overpriced", "waste of money", "not worth the cost", "too expensive"]},
        {"concept": "great value", "aspect": "value", "polarity": 0.78, "aliases": ["worth every penny", "great deal", "excellent for the price"]},
        {"concept": "late delivery", "aspect": "delivery_logistics", "polarity": -0.76, "aliases": ["late delivery", "showed up much later", "kept waiting", "arrived late"]},
        {"concept": "smooth delivery", "aspect": "delivery_logistics", "polarity": 0.7, "aliases": ["arrived on time", "showed up on time", "everything arrived smoothly"]},
        {"concept": "slow service", "aspect": "service_speed", "polarity": -0.78, "aliases": ["waited forever", "took forever", "waited 40 minutes", "slow service"]},
        {"concept": "attentive staff", "aspect": "staff_attitude", "polarity": 0.73, "aliases": ["attentive", "helpful staff", "friendly staff", "polite staff"]},
        {"concept": "dismissive staff", "aspect": "staff_attitude", "polarity": -0.74, "aliases": ["dismissive", "rude staff", "felt ignored", "unhelpful staff"]},
        {"concept": "poor durability", "aspect": "reliability", "polarity": -0.8, "aliases": ["fell apart", "broke after", "stopped working", "acting up again"]},
        {"concept": "stable performance", "aspect": "reliability", "polarity": 0.71, "aliases": ["kept working", "worked every time", "stable and reliable"]},
        {"concept": "battery drains quickly", "aspect": "battery_life", "polarity": -0.8, "aliases": ["dies by noon", "need to charge twice", "battery drains quickly"]},
        {"concept": "good battery life", "aspect": "battery_life", "polarity": 0.72, "aliases": ["lasts all day", "battery lasted all day"]},
        {"concept": "poor taste", "aspect": "taste", "polarity": -0.77, "aliases": ["bland", "tasteless", "salty", "overcooked"]},
        {"concept": "great taste", "aspect": "taste", "polarity": 0.79, "aliases": ["delicious", "flavorful", "mouthwatering", "outstanding taste"]},
        {"concept": "fresh ingredients", "aspect": "freshness", "polarity": 0.74, "aliases": ["fresh", "crisp", "fresh ingredients"]},
        {"concept": "stale food", "aspect": "freshness", "polarity": -0.73, "aliases": ["stale", "soggy", "spoiled"]},
        {"concept": "small portion", "aspect": "portion_size", "polarity": -0.65, "aliases": ["tiny portion", "small portion", "not filling"]},
        {"concept": "large portion", "aspect": "portion_size", "polarity": 0.63, "aliases": ["large portion", "generous portion", "filling"]},
        {"concept": "poor usability", "aspect": "usability", "polarity": -0.72, "aliases": ["confusing", "hard to use", "harder than it should be"]},
        {"concept": "easy to use", "aspect": "usability", "polarity": 0.7, "aliases": ["easy to use", "intuitive", "easy to figure out"]},
        {"concept": "slow performance", "aspect": "performance", "polarity": -0.74, "aliases": ["sluggish", "painfully slow", "laggy", "slow"]},
        {"concept": "fast performance", "aspect": "performance", "polarity": 0.71, "aliases": ["snappy", "responsive", "fast"]},
        {"concept": "pleasant ambience", "aspect": "ambience", "polarity": 0.66, "aliases": ["cozy", "pleasant atmosphere", "comfortable atmosphere"]},
        {"concept": "noisy ambience", "aspect": "ambience", "polarity": -0.68, "aliases": ["too loud", "noisy", "crowded"]},
        {"concept": "screen quality", "aspect": "screen_quality", "polarity": 0.55, "aliases": ["bright display", "sharp screen", "clear display"]},
        {"concept": "technical support", "aspect": "support_quality", "polarity": -0.45, "aliases": ["tech support", "technical support", "customer support"]},
        {"concept": "great fit", "aspect": "fit", "polarity": 0.68, "aliases": ["fits perfectly", "fit like a glove", "true to size", "comfortable fit"]},
        {"concept": "poor fit", "aspect": "fit", "polarity": -0.71, "aliases": ["too tight", "too loose", "awkward fit", "uncomfortable fit"]},
        {"concept": "poor fabric feel", "aspect": "fabric_feel", "polarity": -0.67, "aliases": ["itchy after an hour", "scratchy fabric", "rough material", "uncomfortable fabric"]},
        {"concept": "good fabric feel", "aspect": "fabric_feel", "polarity": 0.64, "aliases": ["soft fabric", "comfortable material", "feels soft", "smooth fabric"]},
        {"concept": "poor return process", "aspect": "return_process", "polarity": -0.65, "aliases": ["return was a hassle", "return process was painful", "refund took forever", "refund process", "exchange was difficult"]},
        {"concept": "easy return process", "aspect": "return_process", "polarity": 0.61, "aliases": ["easy return", "refund was easy", "exchange was smooth", "return process was simple"]},
        {"concept": "food freshness", "aspect": "freshness", "polarity": 0.66, "aliases": ["fresh ingredients", "tasted fresh", "freshly made", "crisp and fresh"]},
        {"concept": "cold food", "aspect": "food_temperature", "polarity": -0.63, "aliases": ["served cold", "lukewarm", "not hot", "cold when it arrived"]},
        {"concept": "small portion", "aspect": "portion_size", "polarity": -0.65, "aliases": ["tiny serving", "small for the price", "not enough food", "left still hungry"]},
        {"concept": "great portion", "aspect": "portion_size", "polarity": 0.62, "aliases": ["huge serving", "plenty of food", "very filling", "generous portion"]},
    ]


def _coerce_entries(raw_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in raw_entries:
        concept = _normalize(item.get("concept", ""))
        aspect = _normalize(item.get("aspect", "")).replace(" ", "_")
        if not concept or not aspect:
            continue
        aliases = [_normalize(alias) for alias in item.get("aliases", []) if _normalize(alias)]
        if concept not in aliases:
            aliases.insert(0, concept)
        entries.append(
            {
                "concept": concept,
                "aspect": aspect,
                "polarity": float(item.get("polarity", 0.0)),
                "aliases": aliases,
                "tokens": set().union(*(_tokenize(alias) for alias in aliases)),
            }
        )
    return entries


def configure_senticnet(*, enabled: bool = True, resource_path: str | Path | None = None) -> None:
    path = Path(resource_path) if resource_path else DEFAULT_RESOURCE_PATH
    _STATE["enabled"] = enabled
    _STATE["resource_path"] = path
    _STATE["entries"] = []
    _STATE["loaded"] = False
    _STATE["source"] = "disabled" if not enabled else "pending"
    if not enabled:
        return
    raw_entries: List[Dict[str, Any]]
    if path.exists():
        raw_entries = json.loads(path.read_text(encoding="utf-8"))
        _STATE["source"] = str(path)
    else:
        raw_entries = _seed_entries()
        _STATE["source"] = "built_in_seed"
    _STATE["entries"] = _coerce_entries(raw_entries)
    _STATE["loaded"] = True


def senticnet_status() -> Dict[str, Any]:
    if not _STATE["loaded"] and _STATE["enabled"]:
        configure_senticnet(enabled=True, resource_path=_STATE["resource_path"])
    return {
        "enabled": bool(_STATE["enabled"]),
        "loaded": bool(_STATE["loaded"]),
        "source": _STATE["source"],
        "entry_count": len(_STATE["entries"]),
    }


def senticnet_vote(text: str, domain: str = "generic") -> Dict[str, Any]:
    del domain
    if not _STATE["loaded"] and _STATE["enabled"]:
        configure_senticnet(enabled=True, resource_path=_STATE["resource_path"])
    if not _STATE["enabled"] or not _STATE["entries"]:
        return {
            "enabled": False,
            "matched": False,
            "best_aspect": "",
            "best_concept": "",
            "best_polarity": 0.0,
            "aspect_scores": {},
            "matched_terms": [],
        }

    normalized = _normalize(text)
    tokens = _tokenize(normalized)
    aspect_scores: Dict[str, float] = {}
    matches: List[Dict[str, Any]] = []
    for entry in _STATE["entries"]:
        score = 0.0
        matched_terms: List[str] = []
        for alias in entry["aliases"]:
            alias_tokens = _tokenize(alias)
            if alias and alias in normalized:
                score = max(score, 0.72 + min(0.18, len(alias_tokens) * 0.04) + min(0.06, len(alias) * 0.003))
                matched_terms.append(alias)
            elif alias_tokens and tokens and alias_tokens.issubset(tokens):
                score = max(score, 0.62 + min(0.16, len(alias_tokens) * 0.04) + min(0.04, len(alias) * 0.002))
                matched_terms.append(alias)
            elif alias_tokens and tokens:
                overlap = len(alias_tokens & tokens)
                coverage = overlap / max(1, len(alias_tokens))
                if overlap >= 2 and coverage >= 0.5:
                    score = max(score, 0.48 + min(0.2, coverage * 0.22))
                    matched_terms.extend(sorted(alias_tokens & tokens))
        if score <= 0:
            continue
        aspect_scores[entry["aspect"]] = max(aspect_scores.get(entry["aspect"], 0.0), score)
        matches.append(
            {
                "aspect": entry["aspect"],
                "concept": entry["concept"],
                "polarity": entry["polarity"],
                "score": round(score, 4),
                "matched_terms": sorted(set(matched_terms)),
            }
        )

    if {"refund", "return", "exchange"} & tokens and {"process", "refund", "return", "hassle"} & tokens:
        heuristic_score = 0.84 if "process" in tokens else 0.78
        aspect_scores["return_process"] = max(aspect_scores.get("return_process", 0.0), heuristic_score)
        matches.append(
            {
                "aspect": "return_process",
                "concept": "return process",
                "polarity": -0.62 if {"hassle", "forever", "difficult"} & tokens else 0.28,
                "score": round(heuristic_score, 4),
                "matched_terms": sorted(({"refund", "return", "exchange", "process", "hassle"} & tokens)),
            }
        )

    matches.sort(key=lambda item: (item["score"], abs(item["polarity"])), reverse=True)
    best = matches[0] if matches else None
    return {
        "enabled": True,
        "matched": bool(best),
        "best_aspect": best["aspect"] if best else "",
        "best_concept": best["concept"] if best else "",
        "best_polarity": float(best["polarity"]) if best else 0.0,
        "aspect_scores": {k: round(v, 4) for k, v in aspect_scores.items()},
        "matched_terms": best["matched_terms"] if best else [],
        "matches": matches[:5],
    }
