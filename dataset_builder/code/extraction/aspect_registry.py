from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

try:
    from .utils import normalize_whitespace
except (ImportError, ValueError):  # pragma: no cover
    from utils.utils import normalize_whitespace

ASPECT_REGISTRY_VERSION = "v1"

RESTAURANT_DOMAINS = {"restaurant", "restaurants", "dining", "food"}
RESTAURANT_CANONICAL_ASPECTS = {
    "food_quality", "service_speed", "ambience", "portion_size", 
    "price", "cleanliness", "wait_time", "location"
}

_GENERIC_CANONICAL_LABELS = {
    "good",
    "quality",
    "misc",
    "other",
    "stuff",
    "thing",
    "general",
}

_RESTAURANT_LATENT_MAP = {
    "sensory quality": "food_quality",
    "food": "food_quality",
    "taste": "food_quality",
    "meal": "food_quality",
    "meals": "food_quality",
    "menu": "food_quality",
    "service quality": "service_speed",
    "service": "service_speed",
    "staff": "service_speed",
    "waiter": "service_speed",
    "waitress": "service_speed",
    "wait staff": "service_speed",
    "timeliness": "wait_time",
    "wait time": "wait_time",
    "speed": "service_speed",
    "comfort": "ambience",
    "environment": "ambience",
    "decor": "ambience",
    "cleanliness": "cleanliness",
    "hygiene": "cleanliness",
    "value": "price",
    "price": "price",
    "cost": "price",
    "portion": "portion_size",
    "portion size": "portion_size",
    "location": "location",
    "atmosphere": "ambience",
}

_ELECTRONICS_LATENT_MAP = {
    "battery": "battery_life",
    "battery backup": "battery_life",
    "battery timing": "battery_life",
    "power": "battery_life",
    "heat": "thermal",
    "overheating": "thermal",
    "gets hot": "thermal",
    "lag": "performance",
    "slow performance": "performance",
    "hangs": "performance",
    "display": "display",
    "screen": "display",
    "screen quality": "display",
    "brightness": "display",
    "keyboard": "keyboard",
    "typing feel": "keyboard",
    "speaker": "audio",
    "sound": "audio",
    "audio": "audio",
}

_ELECTRONICS_SURFACE_MAP = {
    "battery backup": "battery_life",
    "battery timing": "battery_life",
    "battery life": "battery_life",
    "overheating": "thermal",
    "gets hot": "thermal",
    "slow performance": "performance",
    "screen quality": "display",
    "typing feel": "keyboard",
}

_BROAD_GENERIC_LABELS = {
    "service",
    "quality",
    "food",
    "experience",
    "thing",
}


def _norm(text: Any) -> str:
    return normalize_whitespace(str(text or "")).strip().lower()


def _to_canonical_token(text: str) -> str:
    token = _norm(text).replace("-", " ").replace("/", " ")
    token = "_".join(part for part in token.split(" ") if part)
    return token


def _is_registry_worthy_canonical_aspect(*, domain: str, canonical_aspect: str) -> bool:
    canonical = _norm(canonical_aspect)
    if not canonical or canonical in _GENERIC_CANONICAL_LABELS:
        return False
    if is_restaurant_domain(domain):
        return canonical in RESTAURANT_CANONICAL_ASPECTS
    return True

from pathlib import Path
import json


class LearnedOntologyManager:
    """Phase 3: Persistent Domain-Specific Discovery Loop."""
    
    _instances: dict[str, LearnedOntologyManager] = {}

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._learned_data: dict[str, dict[str, Any]] = {}

    @classmethod
    def get_instance(cls, state_dir: Path) -> LearnedOntologyManager:
        sd_str = str(state_dir)
        if sd_str not in cls._instances:
            cls._instances[sd_str] = LearnedOntologyManager(state_dir)
        return cls._instances[sd_str]

    def _get_path(self, domain: str) -> Path:
        return self.state_dir / f"learned_ontology_{domain.lower()}.json"

    def load(self, domain: str):
        path = self._get_path(domain)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._learned_data[domain] = json.load(f)
            except:
                self._learned_data[domain] = {}
        else:
            self._learned_data[domain] = {}
        return self._learned_data[domain]

    def save(self, domain: str):
        if domain not in self._learned_data: return
        path = self._get_path(domain)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._learned_data[domain], f, indent=2, ensure_ascii=False)

    def record_observation(self, domain: str, aspect: str, confidence: float, stability_threshold: int = 5):
        """Records a discovery event and returns True if aspect should be promoted."""
        data = self.load(domain)
        aspect = aspect.lower().strip()
        if not aspect: return False
        
        entry = data.setdefault(aspect, {
            "occurrences": 0,
            "max_confidence": 0.0,
            "promoted": False,
            "aliases": []
        })
        
        entry["occurrences"] += 1
        entry["max_confidence"] = max(entry["max_confidence"], confidence)
        
        should_save = True
        promoted_now = False
        if not entry["promoted"] and entry["occurrences"] >= stability_threshold:
            entry["promoted"] = True
            promoted_now = True
        
        if should_save:
            self.save(domain)
        return promoted_now

    def get_promoted_aspects(self, domain: str) -> list[str]:
        data = self.load(domain)
        return [a for a, meta in data.items() if meta.get("promoted")]

    def merge_similar(self, domain: str, similarity_func: Any, threshold: float = 0.85):
        """Phase 3: Semantic Merging. Aliases similar discovered terms."""
        data = self.load(domain)
        labels = list(data.keys())
        if len(labels) < 2: return
        
        merged_count = 0
        to_delete = set()
        for i, label_a in enumerate(labels):
            if label_a in to_delete: continue
            for label_b in labels[i+1:]:
                if label_b in to_delete: continue
                
                sim = similarity_func(label_a, label_b)
                if sim >= threshold:
                    # Merge B into A
                    entry_a = data[label_a]
                    entry_b = data[label_b]
                    entry_a["occurrences"] += entry_b["occurrences"]
                    entry_a["max_confidence"] = max(entry_a["max_confidence"], entry_b["max_confidence"])
                    entry_a["aliases"] = sorted(list(set(entry_a.get("aliases", [])) | {label_b} | set(entry_b.get("aliases", []))))
                    if entry_b.get("promoted"):
                        entry_a["promoted"] = True
                    to_delete.add(label_b)
                    merged_count += 1
        
        for label in to_delete:
            del data[label]
        
        if merged_count > 0:
            self.save(domain)
            print(f"[+] Phase 3 Semantic Merging: Consolidated {merged_count} similar aspects for domain '{domain}'.")


def is_restaurant_domain(domain: str) -> bool:
    return _norm(domain) in RESTAURANT_DOMAINS


def canonicalize_domain_aspect(*, domain: str, aspect_label: str, surface_rationale_tag: str = "") -> str | None:
    mapping = resolve_domain_canonical_mapping(
        domain=domain,
        latent_aspect=aspect_label,
        surface_rationale_tag=surface_rationale_tag,
    )
    if not mapping:
        return None
    return str(mapping.get("canonical_label") or "").strip() or None


def resolve_domain_canonical_mapping(
    *,
    domain: str,
    latent_aspect: str,
    surface_rationale_tag: str = "",
) -> dict[str, Any] | None:
    latent = _norm(latent_aspect)
    surface = _norm(surface_rationale_tag)
    canonical_domain = _norm(domain)

    if not latent and not surface:
        return None
    if latent in _BROAD_GENERIC_LABELS:
        return None

    if is_restaurant_domain(domain):
        mapped = _RESTAURANT_LATENT_MAP.get(latent)
        if mapped:
            return {"canonical_label": mapped, "mapping_source": "domain_map", "mapping_confidence": 0.95}
        if "portion" in surface or "size" in surface:
            return {"canonical_label": "portion_size", "mapping_source": "surface_rule", "mapping_confidence": 0.9}
        if "service" in surface or "wait" in surface or "slow" in surface or "quick" in surface:
            return {"canonical_label": "service_speed", "mapping_source": "surface_rule", "mapping_confidence": 0.9}
        if any(tok in surface for tok in ("food", "taste", "flavor", "dish", "meal", "fresh")):
            return {"canonical_label": "food_quality", "mapping_source": "surface_rule", "mapping_confidence": 0.9}
        if any(tok in surface for tok in ("ambience", "atmosphere", "noise", "music", "clean")):
            return {"canonical_label": "ambience", "mapping_source": "surface_rule", "mapping_confidence": 0.88}
        return None

    if canonical_domain in {"electronics", "laptop", "laptops"}:
        mapped = _ELECTRONICS_LATENT_MAP.get(latent)
        if mapped:
            return {"canonical_label": mapped, "mapping_source": "domain_map", "mapping_confidence": 0.95}
        for symptom, mapped_surface in _ELECTRONICS_SURFACE_MAP.items():
            if symptom in surface:
                return {"canonical_label": mapped_surface, "mapping_source": "surface_rule", "mapping_confidence": 0.9}

    canonical = _to_canonical_token(latent or surface)
    if not canonical or canonical in _BROAD_GENERIC_LABELS:
        return None
    if not _is_registry_worthy_canonical_aspect(domain=domain, canonical_aspect=canonical):
        return None
    return {"canonical_label": canonical, "mapping_source": "token_normalization", "mapping_confidence": 0.8}


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
            if not canonical or not _is_registry_worthy_canonical_aspect(domain=domain, canonical_aspect=canonical):
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
    mapping = resolve_domain_canonical_mapping(
        domain=domain,
        latent_aspect=latent_aspect,
        surface_rationale_tag=surface_rationale_tag,
    )
    candidate = str((mapping or {}).get("canonical_label") or "").strip()
    if not candidate:
        return None
    if not _is_registry_worthy_canonical_aspect(domain=domain, canonical_aspect=candidate):
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
