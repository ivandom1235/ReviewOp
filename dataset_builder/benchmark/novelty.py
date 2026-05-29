from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..schemas.interpretation import Interpretation


@dataclass(frozen=True)
class NoveltyAssessment:
    status: str
    score: float
    reason: str


def detect_novelty(
    aspect_canonical: str, 
    known_canonicals: set[str], 
    mapping_confidence: float = 1.0,
    mapping_source: str = "none"
) -> str:
    return assess_novelty(
        aspect_canonical, 
        known_canonicals, 
        mapping_confidence=mapping_confidence,
        mapping_source=mapping_source
    ).status


def assess_novelty(
    aspect_canonical: str,
    known_canonicals: set[str],
    *,
    mapping_confidence: float = 1.0,
    mapping_source: str = "none",
    evidence_supported: bool = True,
) -> NoveltyAssessment:
    """
    Assesses if an aspect is novel, known, or boundary.
    Phase 6: Novelty Quality Gate - requires decent confidence for novelty.
    """
    aspect = str(aspect_canonical or "").strip().lower()
    source = str(mapping_source or "none").strip().lower()
    
    # Phase 7: Canonical renaming check (memory_candidate -> open_world_candidate)
    if source == "memory_candidate":
        source = "open_world_candidate"

    # 0. Unknown/Noise: Boundary cases
    if not aspect or aspect in {"unknown", "none", "null"}:
        return NoveltyAssessment("boundary", 0.4, "unknown_aspect_canonical")

    # 1. Open-World Candidates (High Discovery Potential)
    if source == "open_world_candidate":
        if mapping_confidence >= 0.70 and evidence_supported:
            return NoveltyAssessment("novel", 0.85, "open_world_discovery")
        return NoveltyAssessment("boundary", 0.55, "weak_open_world_candidate")

    # 2. Known: Canonical is in the official registry
    if aspect in known_canonicals:
        if mapping_confidence >= 0.65:
            return NoveltyAssessment("known", 0.0, "canonical_in_registry")
        return NoveltyAssessment("boundary", 0.3, "weak_registry_match")
        
    # 3. Novel: Unmapped but high confidence
    if mapping_confidence >= 0.75 and evidence_supported:
        return NoveltyAssessment("novel", 0.9, "high_confidence_unmapped_discovery")
        
    return NoveltyAssessment("boundary", 0.5, "low_confidence_unmapped")


def aggregate_row_novelty(interpretations: list[Interpretation]) -> str:
    """
    Weighted novelty aggregation for a row.
    Prevents a single weak/provisional interpretation from marking a whole row novel.
    """
    if not interpretations:
        return "known"

    if any(getattr(i, "novelty_status", "unknown") == "novel" for i in interpretations):
        return "novel"

    if any(getattr(i, "novelty_status", "unknown") == "boundary" for i in interpretations):
        return "boundary"

    high_conf = [i for i in interpretations if (i.canonical_confidence or 0.0) >= 0.6]
    if not high_conf:
        return "boundary"

    novel_count = sum(1 for i in high_conf if getattr(i, "novelty_status", "unknown") == "novel")
    
    # Rule: If > 40% of high-confidence interpretations are novel, row is novel
    if novel_count / len(high_conf) >= 0.4:
        return "novel"
        
    if novel_count > 0:
        return "boundary"
        
    return "known"
