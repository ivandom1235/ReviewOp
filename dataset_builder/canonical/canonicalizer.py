from __future__ import annotations
from dataclasses import replace
from .domain_maps import lookup_domain_map, CanonicalizationPolicy
from .open_world_fallback import classify_unmapped_candidate, keep_open_world_candidate, mark_provisional_canonical, strip_sentiment_modifiers
from ..schemas.interpretation import Interpretation

def canonicalize_label(
    target: str | Interpretation, 
    domain: str = "unknown", 
    *,
    domain_mode: str | None = None,
    policy: CanonicalizationPolicy | None = None
) -> str:
    if policy is None:
        if domain_mode:
            from ..config import BuilderConfig
            cfg = BuilderConfig(domain_mode=domain_mode)
            policy = CanonicalizationPolicy.from_builder_config(cfg)
        else:
            policy = CanonicalizationPolicy()
            
    res = lookup_domain_map(domain, target, policy=policy)
    if res.aspect_canonical:
        return res.aspect_canonical
    
    label = target.aspect_raw if isinstance(target, Interpretation) else str(target)
    provisional = mark_provisional_canonical(label)
    return provisional if provisional else "unknown"

def canonicalize_interpretation(
    item: Interpretation,
    domain: str = "unknown",
    *,
    domain_mode: str | None = None,
    provisional_policy: str | None = None,
    policy: CanonicalizationPolicy | None = None
) -> Interpretation:
    """Canonicalize an interpretation using multi-step lookup."""
    if policy is None:
        if domain_mode or provisional_policy:
            from ..config import BuilderConfig
            cfg = BuilderConfig(
                domain_mode=domain_mode or "full", 
                provisional_policy=provisional_policy or "strict"
            )
            policy = CanonicalizationPolicy.from_builder_config(cfg)
        else:
            policy = CanonicalizationPolicy()

    # 1. Preliminary Noise Filtering
    from ..explicit.phrase_cleaning import is_noisy_label
    if is_noisy_label(item.aspect_raw):
        new_canonical = "unknown"
        mapping_source = "dropped_noise"
        mapping_scope = "dropped_noise"
        mapping_layers = ("dropped_noise",)
        confidence = 0.0
    else:
        res = lookup_domain_map(domain, item, policy=policy)
        new_canonical = res.aspect_canonical
        mapping_source = res.mapping_source
        mapping_scope = res.mapping_scope
        mapping_layers = res.mapping_layers
        confidence = res.mapping_confidence

        if not new_canonical:
            decision = classify_unmapped_candidate(
                item.aspect_raw,
                item.evidence_text,
                support_count=1,
                provisional_policy=policy.provisional_policy,
            )
            provisional = mark_provisional_canonical(item.aspect_raw)
            if decision.bucket == "provisional" and provisional:
                new_canonical = provisional
                mapping_source = "provisional"
                mapping_scope = "provisional"
                mapping_layers = ("provisional",)
                confidence = 0.35
            elif decision.bucket == "open_world" and keep_open_world_candidate(item.aspect_raw, 0.0) and policy.allow_open_world:
                cleaned = strip_sentiment_modifiers(item.aspect_raw)
                new_canonical = str(cleaned or item.aspect_raw or "open_world").strip().lower().replace(" ", "_")
                mapping_source = "open_world"
                mapping_scope = "open_world"
                mapping_layers = ("open_world",)
                confidence = 0.25
            else:
                if decision.bucket == "dropped_noise":
                    new_canonical = "unknown"
                    mapping_source = "dropped_noise"
                    mapping_scope = "dropped_noise"
                    mapping_layers = ("dropped_noise",)
                    confidence = 0.0
                else:
                    # Phase 7: Rename memory_candidate -> open_world_candidate
                    new_canonical = mark_provisional_canonical(item.aspect_raw) or "unknown"
                    mapping_source = "open_world_candidate"
                    mapping_scope = "open_world_candidate"
                    mapping_layers = ("open_world_candidate",)
                    confidence = 0.15

    # 2. Anchor-Modifier Enhancement
    if (
        new_canonical and new_canonical != "unknown"
        and str(getattr(item, "aspect_anchor", "") or "").strip()
        and tuple(getattr(item, "modifier_terms", ()) or ())
    ):
        if mapping_source in {"exact_phrase", "anchor_only", "token_fallback", "fuzzy_alias", "fuzzy_canonical", "open_world"}:
            mapping_source = "anchor_modifier"
            mapping_layers = ("anchor_modifier",) + tuple(layer for layer in mapping_layers if layer != "anchor_modifier")
            confidence = max(confidence, 0.8)

    # Note: infer_generic_parent removed as per Phase 2 (Domain-Biased Hint Removal)
    # The AspectMemory now handles discovery of parents via evidence clusters.
        
    return replace(
        item, 
        aspect_canonical=new_canonical,
        mapping_source=mapping_source,
        canonical_confidence=confidence,
        mapping_scope=mapping_scope,
        mapping_layers=tuple(mapping_layers),
    )
