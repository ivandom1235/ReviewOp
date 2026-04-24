from __future__ import annotations

from collections import Counter

from ..schemas.raw_review import RawReview
from ..canonical.domain_registry import DomainRegistry
from rich.progress import track, Progress


def estimate_explicit_density(rows: list[RawReview]) -> float:
    if not rows:
        return 0.0
    # Use generic broad labels as a proxy for explicit density
    tokens = DomainRegistry.get_broad_labels("generic")
    hits = sum(1 for row in rows if any(token in row.text.lower() for token in tokens))
    return hits / len(rows)


def estimate_implicit_density(rows: list[RawReview]) -> float:
    if not rows:
        return 0.0
    # Use generic latent family keywords as proxy for implicit density
    families = DomainRegistry.get_latent_families("generic")
    tokens = [token for terms in families.values() for token in terms]
    hits = sum(1 for row in rows if any(token in row.text.lower() for token in tokens))
    return hits / len(rows)


def estimate_domain_mix(rows: list[RawReview]) -> dict[str, int]:
    return dict(Counter(row.domain for row in rows))


def estimate_synthetic_fraction(rows: list[RawReview]) -> float:
    if not rows:
        return 0.0
    synthetic = sum(1 for row in rows if bool(row.metadata.get("synthetic")))
    return synthetic / len(rows)


def profile_dataset(rows: list[RawReview]) -> dict[str, object]:
    with Progress() as progress:
        task = progress.add_task("[blue]Profiling dataset...", total=5)
        
        explicit = estimate_explicit_density(rows)
        progress.update(task, advance=1)
        
        implicit = estimate_implicit_density(rows)
        progress.update(task, advance=1)
        
        mode = "mixed"
        if explicit > implicit * 1.5:
            mode = "explicit_heavy"
        elif implicit > explicit * 1.5:
            mode = "implicit_heavy"
        
        mix = estimate_domain_mix(rows)
        progress.update(task, advance=1)
        
        synthetic = estimate_synthetic_fraction(rows)
        progress.update(task, advance=2) # 4th and 5th steps combined or just finish
        
    return {
        "row_count": len(rows),
        "domain_mix": mix,
        "explicit_density": explicit,
        "implicit_density": implicit,
        "synthetic_fraction": synthetic,
        "profile_mode": mode,
    }
