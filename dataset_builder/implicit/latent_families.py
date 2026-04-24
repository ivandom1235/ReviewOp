from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from ..canonical.domain_registry import DomainRegistry

@dataclass(frozen=True)
class FamilyScore:
    latent_family: str
    confidence: float
    matched_terms: tuple[str, ...] = ()

def load_latent_families(domain: str | None = None) -> dict[str, list[str]]:
    """Load latent families from registry."""
    return DomainRegistry.get_latent_families(domain)

def score_family_match(
    text: str, 
    families: dict[str, list[str]] | None = None,
    symptom_prior: str | None = None,
    domain: str | None = None
) -> FamilyScore:
    """
    Score a text against latent families. 
    Uses registry-loaded terms + an optional soft prior from symptoms.
    """
    families = families or load_latent_families(domain)
    text_norm = str(text or "").lower()
    
    best_family = symptom_prior or "unknown"
    best_matches: list[str] = []
    
    for family, terms in families.items():
        matches = [term for term in terms if term.lower() in text_norm]
        if len(matches) > len(best_matches):
            best_family = family
            best_matches = matches
            
    if not best_matches and not symptom_prior:
        return FamilyScore("unknown", 0.0, ())
        
    # Boost confidence if we have both symptom prior and keyword match
    base_conf = 0.45 if best_matches else 0.3
    if symptom_prior and best_matches and best_family == symptom_prior:
        base_conf += 0.2
        
    confidence = min(1.0, base_conf + 0.1 * len(best_matches))
    return FamilyScore(best_family, confidence, tuple(best_matches))
