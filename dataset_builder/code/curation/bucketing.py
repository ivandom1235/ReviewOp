from __future__ import annotations
from row_contracts import QualityScored, BucketAssigned

def assign_bucket(row: QualityScored) -> BucketAssigned:
    """Assign row to a quality bucket."""
    score = row.quality_score
    
    if score >= 0.8:
        bucket = "benchmark_gold"
    elif score >= 0.5:
        bucket = "train_keep"
    else:
        bucket = "hard_reject"
        
    return BucketAssigned(
        row_id=row.row_id,
        review_text=row.review_text,
        domain=row.domain,
        group_id=row.group_id,
        interpretations=row.interpretations,
        quality_score=row.quality_score,
        is_v7_gold=row.is_v7_gold,
        reason=row.reason,
        bucket=bucket
    )
