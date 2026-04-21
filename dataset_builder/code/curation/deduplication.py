from __future__ import annotations
import hashlib
from typing import List, Tuple, Dict, Any, Optional

try:
    from row_contracts import BucketAssigned, DedupChecked
except ImportError:
    try:
        from ..row_contracts import BucketAssigned, DedupChecked
    except ImportError:
        # Fallback for complex script environments
        BucketAssigned = Any
        DedupChecked = Any

def _get_semantic_key(row: Any) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    """Generate a stable semantic key based on text and interpretations."""
    # Handle both BucketAssigned objects and raw dicts used in build_dataset.py
    if hasattr(row, 'review_text') and row.review_text is not None:
        text = row.review_text
        interpretations = getattr(row, 'interpretations', [])
    elif isinstance(row, dict):
        text = row.get("review_text") or row.get("text") or ""
        interpretations = row.get("interpretations", [])
    else:
        text = str(row)
        interpretations = []
    
    text_key = text.strip().lower()
    interp_keys = []
    for interp in interpretations:
        if hasattr(interp, 'aspect'):
            interp_keys.append((str(interp.aspect), str(interp.sentiment)))
        elif isinstance(interp, dict):
            interp_keys.append((str(interp.get("aspect", "")), str(interp.get("sentiment", ""))))
    interp_keys.sort()
    return (text_key, tuple(interp_keys))

def semantic_cluster_id(row: Any) -> str:
    """Generate a stable cluster ID for a row."""
    key = _get_semantic_key(row)
    return hashlib.md5(str(key).encode("utf-8")).hexdigest()

def exact_or_fuzzy_duplicate_key(row: Any) -> str:
    """Legacy alias for semantic key generation."""
    return str(_get_semantic_key(row))

def apply_semantic_dedup(rows: List[BucketAssigned]) -> List[DedupChecked]:
    seen_keys = {}
    results = []
    
    for row in rows:
        key = _get_semantic_key(row)
        is_duplicate = False
        duplicate_of = None
        
        if key in seen_keys:
            is_duplicate = True
            duplicate_of = seen_keys[key]
        else:
            seen_keys[key] = row.row_id
            
        results.append(DedupChecked(
            row_id=row.row_id,
            review_text=row.review_text,
            domain=row.domain,
            group_id=row.group_id,
            interpretations=row.interpretations,
            quality_score=row.quality_score,
            is_v7_gold=row.is_v7_gold,
            reason=row.reason,
            bucket=row.bucket,
            is_duplicate=is_duplicate,
            duplicate_of=duplicate_of
        ))
        
    return results

def deduplicate_by_cluster(rows: List[Dict[str, Any]], max_per_cluster: int = 1, max_per_split: int = 1) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Legacy V6 deduplication logic implemented for V7 compatibility."""
    deduped = []
    seen_clusters = {}
    stats = {"dropped": 0, "total": len(rows), "clusters": 0}
    
    for row in rows:
        c_id = semantic_cluster_id(row)
        if c_id not in seen_clusters:
            seen_clusters[c_id] = 1
            deduped.append(row)
            stats["clusters"] += 1
        else:
            if seen_clusters[c_id] < max_per_cluster:
                seen_clusters[c_id] += 1
                deduped.append(row)
            else:
                stats["dropped"] += 1
                
    return deduped, stats
