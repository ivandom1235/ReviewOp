from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..reports.quality_report import build_quality_report
from ..split.leakage_checks import check_cross_split_leakage

def run_release_gate(output_dir: Path, cfg: Any) -> tuple[bool, dict[str, Any]]:
    """Generate reports and verify the artifact is ready for release."""
    splits = {}
    for split in ["train", "val", "test"]:
        path = output_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                splits[split] = [json.loads(line) for line in f]
        except Exception:
            continue
            
    if not splits:
        return False, {"error": "no splits found"}
        
    # 1. Leakage Checks
    leakage = check_cross_split_leakage(splits)
    
    # 2. Quality Report
    report = build_quality_report(splits)
    
    metrics = {
        "total_rows": sum(len(r) for r in splits.values()),
        "leakage": leakage,
        "quality": report.__dict__ if hasattr(report, "__dict__") else report
    }
    
    try:
        assert_release_ready(splits, reports={"quality": metrics["quality"]}, leakage=leakage)
        return True, metrics
    except Exception as e:
        metrics["error"] = str(e)
        return False, metrics

def assert_release_ready(
    splits: dict[str, list[Any]],
    *,
    reports: dict[str, Any],
    leakage: dict[str, int],
) -> None:
    total = sum(len(rows) for rows in splits.values())
    if total <= 0:
        raise ValueError("benchmark export is empty")
    
    empty_splits = [split for split in ("train", "val", "test") if len(splits.get(split, [])) <= 0]
    if empty_splits:
        raise ValueError(f"benchmark export has empty splits: {', '.join(empty_splits)}")
        
    if int(leakage.get("grouped_leakage", 0)) != 0:
        raise ValueError("grouped split leakage detected")
    if int(leakage.get("exact_text_leakage", 0)) != 0:
        raise ValueError("exact text leakage detected")
        
    quality = reports.get("quality", {})
    if isinstance(quality, dict):
        report_total = quality.get("total_rows", total)
    else:
        report_total = getattr(quality, "total_rows", total)
        
    if int(report_total) != total:
        raise ValueError(f"report/export count mismatch: {report_total} != {total}")
