from __future__ import annotations

from dataclasses import replace

from .domain_maps import lookup_domain_map
from .open_world_fallback import mark_provisional_canonical
from ..schemas.interpretation import Interpretation


def canonicalize_label(label: str, domain: str = "unknown") -> str:
    mapped = lookup_domain_map(domain, label)
    if mapped:
        return mapped
    
    provisional = mark_provisional_canonical(label)
    return provisional if provisional else "unknown"


def canonicalize_interpretation(item: Interpretation, domain: str = "unknown") -> Interpretation:
    # Use aspect_raw for mapping
    new_canonical = canonicalize_label(item.aspect_raw, domain)
    return replace(item, aspect_canonical=new_canonical)
