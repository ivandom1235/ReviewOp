from __future__ import annotations
from .domain_registry import DomainRegistry

def load_domain_maps() -> dict[str, dict[str, str]]:
    """Legacy function, now returns mappings from generic registry."""
    return {}

def lookup_domain_map(domain: str, label: str) -> str | None:
    """Lookup a canonical label for a raw aspect in a given domain."""
    return DomainRegistry.get_domain_map(domain).get(str(label or "").lower())
