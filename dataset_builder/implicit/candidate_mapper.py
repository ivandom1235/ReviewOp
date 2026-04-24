from __future__ import annotations

import re
from dataclasses import dataclass

from .latent_families import score_family_match


@dataclass(frozen=True)
class CandidateMapping:
    aspect_raw: str
    latent_family: str
    aspect_canonical: str
    confidence: float
    mapping_source: str


def _canonicalize(value: str) -> str:
    canonical = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return canonical or "unknown"


def map_to_canonical_candidate(candidate: str, domain_map: dict[str, str] | None = None) -> CandidateMapping:
    raw = str(candidate or "").strip()
    domain_map = domain_map or {}
    family = score_family_match(raw)
    canonical = domain_map.get(raw.lower(), _canonicalize(raw))
    source = "domain_map" if raw.lower() in domain_map else "open_world"
    confidence = max(family.confidence, 0.55 if source == "open_world" and raw else 0.0)
    return CandidateMapping(raw, family.latent_family, canonical, confidence, source)


def assign_latent_family(candidate: str) -> str:
    return score_family_match(candidate).latent_family


def compute_mapping_confidence(candidate: str) -> float:
    return map_to_canonical_candidate(candidate).confidence
