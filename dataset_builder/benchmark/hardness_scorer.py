from __future__ import annotations
from ..schemas.benchmark_row import BenchmarkRow

def score_row_hardness(row: BenchmarkRow) -> str:
    """
    Score the hardness level of a benchmark row (H0-H3).
    H0: Explicitly stated aspects with clear sentiment.
    H1: Contains implicit cues/symptoms.
    H2: Contains mixed sentiment or high interpretation count (> 4).
    H3: High ambiguity or cross-domain complexity.
    """
    if not row.gold_interpretations:
        return "H0"
        
    has_implicit = any(i.label_type == "implicit" for i in row.gold_interpretations)
    sentiments = {i.sentiment.lower() for i in row.gold_interpretations if i.sentiment}
    has_mixed = len(sentiments) > 1
    if str(row.novelty_status or "") == "novel" and float(row.ambiguity_score or 0.0) >= 0.5:
        return "H3"
    if any(i.aspect_canonical == "unknown" for i in row.gold_interpretations) and float(row.ambiguity_score or 0.0) >= 0.5:
        return "H3"
    
    # High complexity signal
    if len(row.gold_interpretations) > 4:
        return "H2"
        
    if has_mixed:
        return "H2"
    if has_implicit:
        return "H1"
        
    return "H0"
