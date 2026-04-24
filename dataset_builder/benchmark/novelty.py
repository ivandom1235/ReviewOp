from __future__ import annotations

from dataclasses import dataclass


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
    aspect = str(aspect_canonical or "").strip()
    
    # 1. Known: Canonical is in the official registry AND high confidence
    if aspect and aspect != "unknown" and aspect in known_canonicals and mapping_confidence >= 0.75:
        return NoveltyAssessment("known", 0.0, "canonical_in_registry")
        
    # 2. Boundary: Weak confidence OR untrusted/provisional mapping
    if mapping_source in ["token_fallback", "provisional", "unmapped"] or 0.0 < mapping_confidence < 0.75:
        return NoveltyAssessment("boundary", 0.6, "weak_mapping_or_untrusted_source")
        
    if not evidence_supported:
        return NoveltyAssessment("boundary", 0.5, "insufficient_evidence")

    # 3. Novel: Unmapped OR unknown to registry with decent confidence
    if aspect and aspect != "unknown" and aspect not in known_canonicals:
        return NoveltyAssessment("novel", 0.9, "high_confidence_unknown_canonical")
        
    return NoveltyAssessment("novel", 1.0, "unmapped_evidence_supported")


def aggregate_row_novelty(interpretations: list[Interpretation]) -> str:
    """
    Weighted novelty aggregation for a row.
    Prevents a single weak/provisional interpretation from marking a whole row novel.
    """
    if not interpretations:
        return "known"

    high_conf = [i for i in interpretations if (i.canonical_confidence or 0.0) >= 0.6]
    if not high_conf:
        # If everything is low confidence, the row is at best 'boundary'
        return "boundary"

    # 1. Primary interpretation check
    primary = max(high_conf, key=lambda i: (i.canonical_confidence or 0.0))
    
    # 2. Counts
    novel_count = sum(1 for i in high_conf if getattr(i, "novelty_status", "unknown") == "novel")
    boundary_count = sum(1 for i in high_conf if getattr(i, "novelty_status", "unknown") == "boundary")
    
    # Rule A: Primary is strongly novel
    if getattr(primary, "novelty_status", "unknown") == "novel" and (primary.canonical_confidence or 0.0) >= 0.75:
        return "novel"
        
    # Rule B: Majority of high-confidence interpretations are novel
    if novel_count / len(high_conf) > 0.5:
        return "novel"
        
    # Rule C: Some novelty or boundary presence
    if novel_count > 0 or boundary_count > 0:
        return "boundary"
        
    return "known"


def balance_novelty_across_splits(splits: dict[str, list[object]]) -> dict[str, int]:
    return {split: len(rows) for split, rows in splits.items()}
