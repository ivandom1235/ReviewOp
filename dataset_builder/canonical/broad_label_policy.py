from __future__ import annotations

from ..schemas.interpretation import Interpretation


from .domain_registry import DomainRegistry


def is_broad_label(label: str, domain: str | None = None) -> bool:
    broad_labels = DomainRegistry.get_broad_labels(domain)
    return str(label or "").strip().lower() in broad_labels


def prune_broad_labels(items: list[Interpretation], domain: str | None = None) -> tuple[list[Interpretation], dict[str, int]]:
    if len(items) <= 1:
        return items, {"dropped_broad_labels": 0}
    specific = [item for item in items if not is_broad_label(item.aspect_canonical, domain)]
    if not specific:
        return items, {"dropped_broad_labels": 0}
    kept = [item for item in items if item in specific or not is_broad_label(item.aspect_canonical, domain)]
    return kept, {"dropped_broad_labels": len(items) - len(kept)}
