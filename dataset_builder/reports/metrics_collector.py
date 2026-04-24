from __future__ import annotations
import json
from pathlib import Path
from typing import Any, dict

def collect_metrics(
    output_dir: Path,
    quality_report: Any,
    leakage: dict[str, int],
    gate_results: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate all run metrics into a standardized summary."""
    
    summary = {
        "timestamp": quality_report.timestamp if hasattr(quality_report, "timestamp") else None,
        "counts": {
            "input": quality_report.total_input if hasattr(quality_report, "total_input") else 0,
            "exported": quality_report.total_exported if hasattr(quality_report, "total_exported") else 0,
            "rejected": quality_report.total_rejected if hasattr(quality_report, "total_rejected") else 0,
        },
        "distributions": {
            "source_type": quality_report.source_type_distribution if hasattr(quality_report, "source_type_distribution") else {},
            "canonical": quality_report.canonical_distribution if hasattr(quality_report, "canonical_distribution") else {},
        },
        "quality": {
            "evidence_match_rate": quality_report.evidence.get("exact_match_rate", 0) if hasattr(quality_report, "evidence") else 0,
            "avg_gold_per_row": quality_report.avg_gold_per_row if hasattr(quality_report, "avg_gold_per_row") else 0,
        },
        "leakage": leakage,
        "gate_results": gate_results
    }
    
    # Write to file
    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return summary
