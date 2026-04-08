from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from utils import normalize_whitespace

ASPECT_REGISTRY_VERSION = "v1"

RESTAURANT_DOMAINS = {"restaurant", "restaurants", "dining", "food"}
RESTAURANT_CANONICAL_ASPECTS = {"food_quality", "service_speed", "ambience", "portion_size"}

_RESTAURANT_LATENT_MAP = {
    "sensory quality": "food_quality",
    "food": "food_quality",
    "service quality": "service_speed",
    "service": "service_speed",
    "timeliness": "service_speed",
    "comfort": "ambience",
    "cleanliness": "ambience",
    "value": "portion_size",
    "portion": "portion_size",
    "portion size": "portion_size",
}


def _norm(text: Any) -> str:
    return normalize_whitespace(str(text or "")).strip().lower()


def _to_canonical_token(text: str) -> str:
    token = _norm(text).replace("-", " ").replace("/", " ")
    token = "_".join(part for part in token.split(" ") if part)
    return token


def is_restaurant_domain(domain: str) -> bool:
    return _norm(domain) in RESTAURANT_DOMAINS


def canonicalize_domain_aspect(*, domain: str, aspect_label: str, surface_rationale_tag: str = "") -> str | None:
    latent = _norm(aspect_label)
    surface = _norm(surface_rationale_tag)

    if not latent and not surface:
        return None

    if is_restaurant_domain(domain):
        mapped = _RESTAURANT_LATENT_MAP.get(latent)
        if mapped:
            return mapped
        if "portion" in surface or "size" in surface:
            return "portion_size"
        if "service" in surface or "wait" in surface or "slow" in surface or "quick" in surface:
            return "service_speed"
        if any(tok in surface for tok in ("food", "taste", "flavor", "dish", "meal")):
            return "food_quality"
        if any(tok in surface for tok in ("ambience", "atmosphere", "noise", "music", "clean")):
            return "ambience"
        return None

    canonical = _to_canonical_token(latent or surface)
    return canonical or None


def _entry_template(*, canonical_label: str, alias: str, sentiment: str, run_ts: str) -> dict[str, Any]:
    return {
        "canonical_label": canonical_label,
        "aliases": [alias] if alias else [],
        "support_count": 0,
        "stability_score": 0.0,
        "sentiment_profile": {sentiment: 1} if sentiment else {},
        "first_seen": run_ts,
        "last_seen": run_ts,
        "status": "candidate",
        "seen_runs": 1,
        "missing_runs": 0,
        "version": ASPECT_REGISTRY_VERSION,
    }


def build_run_registry(*, rows: list[dict[str, Any]], run_id: str, run_ts: str) -> dict[str, Any]:
    per_domain: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        domain = _norm(row.get("domain") or "unknown")
        implicit = row.get("implicit", {}) or {}
        spans = list(implicit.get("spans") or [])
        if not spans:
            spans = [{
                "latent_label": aspect,
                "aspect": aspect,
                "sentiment": str(implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral"))),
            } for aspect in list(implicit.get("aspects") or []) if str(aspect) != "general"]

        for span in spans:
            latent = str(span.get("latent_label") or span.get("aspect") or "")
            surface = str(span.get("aspect") or span.get("surface_rationale_tag") or span.get("evidence_text") or "")
            sentiment = _norm(span.get("sentiment") or implicit.get("dominant_sentiment") or "neutral") or "neutral"
            canonical = canonicalize_domain_aspect(domain=domain, aspect_label=latent, surface_rationale_tag=surface)
            if not canonical:
                continue

            bucket = per_domain[domain]
            if canonical not in bucket:
                bucket[canonical] = _entry_template(canonical_label=canonical, alias=_norm(surface), sentiment=sentiment, run_ts=run_ts)
            entry = bucket[canonical]
            entry["support_count"] = int(entry.get("support_count", 0)) + 1
            if surface:
                aliases = set(entry.get("aliases") or [])
                aliases.add(_norm(surface))
                entry["aliases"] = sorted(alias for alias in aliases if alias)
            profile: dict[str, int] = dict(entry.get("sentiment_profile") or {})
            profile[sentiment] = int(profile.get(sentiment, 0)) + 1
            entry["sentiment_profile"] = profile
            entry["last_seen"] = run_ts

    for domain_entries in per_domain.values():
        for entry in domain_entries.values():
            support = int(entry.get("support_count", 0))
            entry["stability_score"] = round(min(1.0, support / 50.0), 4)

    return {
        "run_id": run_id,
        "generated_at": run_ts,
        "registry_version": ASPECT_REGISTRY_VERSION,
        "domains": {dom: {key: value for key, value in entries.items()} for dom, entries in per_domain.items()},
    }


def update_promoted_registry(
    *,
    previous: dict[str, Any] | None,
    run_registry: dict[str, Any],
    promote_min_runs: int = 3,
    promote_min_support: int = 25,
    promote_min_stability: float = 0.7,
    deprecate_missing_runs: int = 5,
    deprecate_max_stability: float = 0.3,
) -> dict[str, Any]:
    base = deepcopy(previous) if isinstance(previous, dict) else {
        "registry_version": ASPECT_REGISTRY_VERSION,
        "domains": {},
        "history": {"runs_seen": 0},
    }
    base.setdefault("domains", {})
    base.setdefault("history", {})
    base["history"]["runs_seen"] = int(base["history"].get("runs_seen", 0)) + 1
    run_ts = str(run_registry.get("generated_at") or "")

    run_domains = run_registry.get("domains", {})
    for domain, entries in base["domains"].items():
        run_entries = run_domains.get(domain, {})
        for canonical, entry in entries.items():
            if canonical not in run_entries:
                entry["missing_runs"] = int(entry.get("missing_runs", 0)) + 1

    for domain, entries in run_domains.items():
        domain_bucket = base["domains"].setdefault(domain, {})
        for canonical, run_entry in entries.items():
            if canonical not in domain_bucket:
                domain_bucket[canonical] = deepcopy(run_entry)
                domain_bucket[canonical]["seen_runs"] = 1
                domain_bucket[canonical]["missing_runs"] = 0
            else:
                merged = domain_bucket[canonical]
                merged["support_count"] = int(merged.get("support_count", 0)) + int(run_entry.get("support_count", 0))
                merged["last_seen"] = run_ts or merged.get("last_seen")
                merged["missing_runs"] = 0
                merged["seen_runs"] = int(merged.get("seen_runs", 0)) + 1
                aliases = set(merged.get("aliases") or []) | set(run_entry.get("aliases") or [])
                merged["aliases"] = sorted(alias for alias in aliases if alias)
                profile = Counter(merged.get("sentiment_profile") or {})
                profile.update(Counter(run_entry.get("sentiment_profile") or {}))
                merged["sentiment_profile"] = dict(profile)
                merged["stability_score"] = round(min(1.0, int(merged.get("support_count", 0)) / 50.0), 4)

            entry = domain_bucket[canonical]
            seen_runs = int(entry.get("seen_runs", 0))
            support = int(entry.get("support_count", 0))
            stability = float(entry.get("stability_score", 0.0))
            missing_runs = int(entry.get("missing_runs", 0))
            if missing_runs >= deprecate_missing_runs or stability < deprecate_max_stability:
                if missing_runs >= deprecate_missing_runs or seen_runs >= promote_min_runs:
                    entry["status"] = "deprecated"
                else:
                    entry["status"] = "candidate"
            elif seen_runs >= promote_min_runs and support >= promote_min_support and stability >= promote_min_stability:
                entry["status"] = "promoted"
            else:
                entry["status"] = "candidate"
            entry["version"] = ASPECT_REGISTRY_VERSION

    base["registry_version"] = ASPECT_REGISTRY_VERSION
    base["updated_at"] = run_ts
    return base


def resolve_registry_version(registry: dict[str, Any] | None) -> str:
    if not isinstance(registry, dict):
        return ASPECT_REGISTRY_VERSION
    return str(registry.get("registry_version") or ASPECT_REGISTRY_VERSION)


def resolve_domain_canonical_aspect(
    *,
    registry: dict[str, Any] | None,
    domain: str,
    latent_aspect: str,
    surface_rationale_tag: str,
    enforce_registry_membership: bool = False,
) -> str | None:
    candidate = canonicalize_domain_aspect(domain=domain, aspect_label=latent_aspect, surface_rationale_tag=surface_rationale_tag)
    if not candidate:
        return None

    if not enforce_registry_membership:
        return candidate

    if not isinstance(registry, dict):
        return None
    domain_bucket = (registry.get("domains") or {}).get(_norm(domain), {})
    if candidate in domain_bucket:
        return candidate
    return None


def restaurant_ontology_compatible(*, domain: str, canonical_aspect: str) -> bool:
    if not is_restaurant_domain(domain):
        return True
    return canonical_aspect in RESTAURANT_CANONICAL_ASPECTS
