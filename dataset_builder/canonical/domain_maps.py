from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from .domain_registry import DomainRegistry
from ..schemas.interpretation import Interpretation

@dataclass
class CanonicalMappingResult:
    aspect_canonical: str | None
    latent_family: str | None = None
    mapping_source: str = "unknown"
    mapping_confidence: float = 0.0
    matched_key: str | None = None
    ambiguity_flag: bool = False

def lookup_domain_map(domain: str | None, target: Any, config_dir: Path | None = None) -> CanonicalMappingResult:
    """
    Look up a canonical aspect from a domain map with multi-step precedence.
    """
    if target is None:
        return CanonicalMappingResult(None)

    # Pre-step: Extract info from Interpretation if provided
    raw_phrase = None
    anchor = None
    modifiers = ()
    source_type = "unknown"
    existing_canonical = None

    if isinstance(target, Interpretation):
        raw_phrase = target.aspect_raw
        anchor = target.aspect_anchor
        modifiers = target.modifier_terms
        source_type = target.source_type
        existing_canonical = target.aspect_canonical
    else:
        raw_phrase = str(target)

    # 1. Preserve trusted canonical (learned store)
    if source_type == "implicit_learned" and existing_canonical and existing_canonical != "unknown":
        return CanonicalMappingResult(
            aspect_canonical=existing_canonical,
            mapping_source="trusted_learned",
            mapping_confidence=1.0
        )

    # Load config
    config = DomainRegistry.get_config(domain, config_dir=config_dir) if config_dir else DomainRegistry.get_config(domain)
    domain_map = config.get("domain_maps", {})
    modifier_map = config.get("modifier_maps", {})

    # 2. Exact phrase match
    if raw_phrase:
        lookup_phrase = raw_phrase.lower().strip()
        if lookup_phrase in domain_map:
            return CanonicalMappingResult(
                aspect_canonical=domain_map[lookup_phrase],
                mapping_source="exact_phrase",
                mapping_confidence=1.0,
                matched_key=lookup_phrase
            )

    # 3. Anchor + Modifier contextual match
    if anchor:
        lookup_anchor = anchor.lower().strip()
        if lookup_anchor in modifier_map:
            sub_map = modifier_map[lookup_anchor]
            found_canonicals = []
            for mod in modifiers:
                mod_clean = mod.lower().strip()
                if mod_clean in sub_map:
                    found_canonicals.append((sub_map[mod_clean], mod_clean))
            
            if found_canonicals:
                # Handle conflict
                unique_canons = sorted(list(set(c[0] for c in found_canonicals)))
                if len(unique_canons) > 1:
                    return CanonicalMappingResult(
                        aspect_canonical=unique_canons[0], # Take first for now
                        mapping_source="anchor_modifier",
                        mapping_confidence=0.7,
                        matched_key=f"{lookup_anchor}.{found_canonicals[0][1]}",
                        ambiguity_flag=True
                    )
                return CanonicalMappingResult(
                    aspect_canonical=unique_canons[0],
                    mapping_source="anchor_modifier",
                    mapping_confidence=0.85,
                    matched_key=f"{lookup_anchor}.{found_canonicals[0][1]}"
                )

    # 4. Simple anchor match
    if anchor:
        lookup_anchor = anchor.lower().strip()
        if lookup_anchor in domain_map:
            return CanonicalMappingResult(
                aspect_canonical=domain_map[lookup_anchor],
                mapping_source="anchor_only",
                mapping_confidence=0.8,
                matched_key=lookup_anchor
            )

    # 5. Controlled whole-token fallback
    if raw_phrase:
        # Split into tokens and look for matches
        tokens = [t.strip().lower() for t in raw_phrase.split() if len(t.strip()) > 2]
        for token in tokens:
            if token in domain_map:
                return CanonicalMappingResult(
                    aspect_canonical=domain_map[token],
                    mapping_source="token_fallback",
                    mapping_confidence=0.55,
                    matched_key=token
                )

    return CanonicalMappingResult(None, mapping_source="unmapped")
