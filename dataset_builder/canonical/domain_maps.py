from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from .domain_registry import DomainRegistry
from .fuzzy_match import FuzzyMatcher
from ..schemas.interpretation import Interpretation

@dataclass(frozen=True)
class CanonicalizationPolicy:
    domain_mode: str = "full"
    provisional_policy: str = "strict"
    allow_generic: bool = True
    allow_domain_specific: bool = True
    allow_learned_store: bool = True
    allow_open_world: bool = True

    @classmethod
    def from_builder_config(cls, cfg: Any) -> "CanonicalizationPolicy":
        mode = getattr(cfg, "domain_mode", "full")
        provisional = getattr(cfg, "provisional_policy", "strict")
        return cls(
            domain_mode=mode,
            provisional_policy=provisional,
            allow_generic=mode in {"generic_only", "generic_plus_learned", "generic_plus_domain", "full"},
            allow_domain_specific=mode in {"generic_plus_domain", "domain_only", "full"},
            allow_learned_store=mode in {"generic_plus_learned", "full"},
            allow_open_world=mode == "full"
        )

@dataclass
class CanonicalMappingResult:
    aspect_canonical: str | None
    latent_family: str | None = None
    mapping_source: str = "unknown"
    mapping_confidence: float = 0.0
    matched_key: str | None = None
    ambiguity_flag: bool = False
    mapping_layers: tuple[str, ...] = field(default_factory=tuple)
    mapping_scope: str = "unknown"

def lookup_domain_map(domain: str | None, target: Any, config_dir: Path | None = None, policy: CanonicalizationPolicy | None = None) -> CanonicalMappingResult:
    """
    Look up a canonical aspect from a domain map with multi-step precedence.
    """
    if target is None:
        return CanonicalMappingResult(None)

    if policy is None:
        policy = CanonicalizationPolicy()

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
        latent_family = target.latent_family
    else:
        raw_phrase = str(target)

    # 1. Preserve trusted canonical (learned store)
    if source_type == "implicit_learned" and existing_canonical and existing_canonical != "unknown":
        if policy.allow_learned_store:
            return CanonicalMappingResult(
                aspect_canonical=existing_canonical,
                mapping_source="trusted_learned",
                mapping_confidence=1.0,
                mapping_scope="symptom_store",
                mapping_layers=("symptom_store",),
            )
        else:
            return CanonicalMappingResult(None, mapping_source="no_match", mapping_scope="unmapped_internal")

    # Load config hierarchy
    effective_domain = str(domain or "generic").lower()
    source_cfg = DomainRegistry.get_source_config(effective_domain, config_dir=config_dir) if config_dir else DomainRegistry.get_source_config(effective_domain)

    generic_map = source_cfg.generic.get("domain_maps", {})
    domain_map = source_cfg.domain_raw.get("domain_maps", {})
    full_map = source_cfg.merged.get("domain_maps", {})
    generic_modifiers = source_cfg.generic.get("modifier_maps", {})
    domain_modifiers = source_cfg.domain_raw.get("modifier_maps", {})
    full_modifiers = source_cfg.merged.get("modifier_maps", {})

    # Helper to build result with layers
    def make_result(aspect, source, confidence=1.0, key=None, ambiguity=False):
        key_l = str(key or "").lower()
        layers: list[str] = []
        if key_l:
            in_generic = key_l in generic_map or key_l in generic_modifiers
            in_domain = key_l in domain_map or key_l in domain_modifiers
            if in_generic and policy.allow_generic:
                layers.append("generic")
            if in_domain and policy.allow_domain_specific:
                layers.append("domain_specific")
        
        # If no allowed layers matched, reject the result
        if not layers:
            return CanonicalMappingResult(None, mapping_source="no_match", mapping_scope="unmapped_internal")

        if layers == ["generic"]:
            scope = "generic"
        elif layers == ["domain_specific"]:
            scope = "domain_specific"
        elif "generic" in layers and "domain_specific" in layers:
            scope = "generic+domain_specific"
        else:
            scope = "generic"

        return CanonicalMappingResult(
            aspect_canonical=aspect,
            mapping_source=source,
            mapping_confidence=confidence,
            matched_key=key,
            ambiguity_flag=ambiguity,
            mapping_layers=tuple(layers),
            mapping_scope=scope
        )

    # 2. Anchor + Modifier contextual match
    if anchor:
        lookup_anchor = anchor.lower().strip()
        if lookup_anchor in full_modifiers:
            sub_map = full_modifiers[lookup_anchor]
            found_canonicals = []
            for mod in modifiers:
                mod_clean = mod.lower().strip()
                if mod_clean in sub_map:
                    found_canonicals.append((sub_map[mod_clean], mod_clean))
            
            if found_canonicals:
                unique_canons = sorted(list(set(c[0] for c in found_canonicals)))
                mod_key = found_canonicals[0][1]
                in_generic = lookup_anchor in generic_modifiers and mod_key in generic_modifiers.get(lookup_anchor, {})
                in_domain = lookup_anchor in domain_modifiers and mod_key in domain_modifiers.get(lookup_anchor, {})
                layers = []
                if in_generic and policy.allow_generic:
                    layers.append("generic")
                if in_domain and policy.allow_domain_specific:
                    layers.append("domain_specific")
                
                if layers:
                    scope = "generic+domain_specific" if set(layers) == {"generic", "domain_specific"} else layers[0]
                    return CanonicalMappingResult(
                        aspect_canonical=unique_canons[0],
                        mapping_source="anchor_modifier",
                        mapping_confidence=0.7 if len(unique_canons) > 1 else 0.85,
                        matched_key=f"{lookup_anchor}.{mod_key}",
                        ambiguity_flag=len(unique_canons) > 1,
                        mapping_layers=tuple(layers),
                        mapping_scope=scope,
                    )

    # 3. Exact phrase match
    if raw_phrase:
        lookup_phrase = raw_phrase.lower().strip()
        if lookup_phrase in full_map:
            result = make_result(full_map[lookup_phrase], "exact_phrase", 1.0, lookup_phrase)
            if result.aspect_canonical: return result

    # 4. Simple anchor match
    if anchor:
        lookup_anchor = anchor.lower().strip()
        if lookup_anchor in full_map:
            result = make_result(full_map[lookup_anchor], "anchor_only", 0.8, lookup_anchor)
            if result.aspect_canonical: return result

    # 5. Controlled whole-token fallback
    if raw_phrase:
        tokens = [t.strip().lower() for t in raw_phrase.split() if len(t.strip()) > 2]
        for token in tokens:
            if token in full_map:
                result = make_result(full_map[token], "token_fallback", 0.55, token)
                if result.aspect_canonical: return result
    
    # 6. Fuzzy Semantic match
    if raw_phrase:
        broad_labels = source_cfg.merged.get("broad_labels", [])
        fuzzy_res = FuzzyMatcher.find_best_match(raw_phrase, broad_labels, threshold=85.0)
        if fuzzy_res:
            matched, score = fuzzy_res
            result = make_result(matched, "fuzzy_canonical", score / 100.0, matched)
            if result.aspect_canonical: return result
            
        all_aliases = list(full_map.keys())
        fuzzy_res = FuzzyMatcher.find_best_match(raw_phrase, all_aliases, threshold=80.0)
        if fuzzy_res:
            matched_key, score = fuzzy_res
            result = make_result(full_map[matched_key], "fuzzy_alias", (score / 100.0) * 0.9, matched_key)
            if result.aspect_canonical: return result

    # 7. Latent Family Inference
    if isinstance(target, Interpretation) and target.latent_family and target.latent_family != "unknown":
        broad_labels = source_cfg.merged.get("broad_labels", [])
        family_clean = target.latent_family.lower().strip()
        
        if family_clean in broad_labels:
            result = make_result(family_clean, "latent_family_inference", 0.55, family_clean)
            if result.aspect_canonical: return result
            
        if family_clean in full_map:
            result = make_result(full_map[family_clean], "latent_family_map", 0.5, family_clean)
            if result.aspect_canonical: return result
            
        fuzzy_res = FuzzyMatcher.find_best_match(family_clean, broad_labels, threshold=90.0)
        if fuzzy_res:
            matched, score = fuzzy_res
            result = make_result(matched, "fuzzy_family_inference", (score / 100.0) * 0.45, matched)
            if result.aspect_canonical: return result

    return CanonicalMappingResult(None, mapping_source="no_match", mapping_scope="unmapped_internal")
