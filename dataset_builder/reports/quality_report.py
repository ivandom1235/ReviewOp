from __future__ import annotations

from collections import Counter
from ..schemas.reports import QualityReport


def build_quality_report(splits: dict[str, list[object]], original_sample_size: int = 0) -> QualityReport:
    counts = {split: len(rows) for split, rows in splits.items()}
    rejected_interps = 0
    reason_counts = Counter()
    for rows in splits.values():
        for row in rows:
            if not hasattr(row, "gold_interpretations"):
                continue
            for interp in getattr(row, "gold_interpretations", []):
                if interp and hasattr(interp, "quality_flags"):
                    for flag in interp.quality_flags:
                        if flag in ("llm_drop", "repair_failed", "low_quality"):
                            rejected_interps += 1
                            reason_counts[flag] += 1
                            
    total_exported = sum(counts.values())
    return QualityReport(
        total_exported=total_exported, 
        export_counts=counts,
        original_sample_size=original_sample_size,
        total_discarded=max(0, original_sample_size - total_exported),
        rejected_interpretations=rejected_interps,
        reason_counts=dict(reason_counts)
    )
