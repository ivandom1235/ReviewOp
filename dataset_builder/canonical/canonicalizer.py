from __future__ import annotations
from dataclasses import replace
from .domain_maps import lookup_domain_map
from .open_world_fallback import mark_provisional_canonical
from ..schemas.interpretation import Interpretation

def canonicalize_label(target: str | Interpretation, domain: str = "unknown") -> str:
    res = lookup_domain_map(domain, target)
    if res.aspect_canonical:
        return res.aspect_canonical
    
    label = target.aspect_raw if isinstance(target, Interpretation) else str(target)
    provisional = mark_provisional_canonical(label)
    return provisional if provisional else "unknown"

def canonicalize_interpretation(item: Interpretation, domain: str = "unknown") -> Interpretation:
    """Canonicalize an interpretation using multi-step lookup."""
    res = lookup_domain_map(domain, item)
    new_canonical = res.aspect_canonical
    mapping_source = res.mapping_source
    
    if not new_canonical:
        new_canonical = mark_provisional_canonical(item.aspect_raw) or "unknown"
        mapping_source = "provisional" if new_canonical != "unknown" else "unmapped"
        
    return replace(
        item, 
        aspect_canonical=new_canonical,
        mapping_source=mapping_source,
        canonical_confidence=res.mapping_confidence
    )
