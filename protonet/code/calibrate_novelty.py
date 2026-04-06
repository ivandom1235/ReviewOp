from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def calibrate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"best_threshold": 0.5, "best_f1": 0.0, "sweep": []}
    truth = [1 if bool(row.get("novel_aspect_acceptable", False)) else 0 for row in rows]
    scores = [float(row.get("novelty_score", 0.0)) for row in rows]
    thresholds = np.linspace(0.05, 0.95, 19)
    sweep: list[dict[str, float]] = []
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = [1 if s >= float(threshold) else 0 for s in scores]
        tp = sum(1 for p, t in zip(preds, truth) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(preds, truth) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(preds, truth) if p == 0 and t == 1)
        f1 = _f1(tp, fp, fn)
        sweep.append({"threshold": float(threshold), "f1": float(f1)})
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return {"best_threshold": best_threshold, "best_f1": max(0.0, best_f1), "sweep": sweep}


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate novelty threshold from validation predictions.")
    parser.add_argument("--input", type=Path, required=True, help="Path to prediction jsonl.")
    parser.add_argument("--output", type=Path, default=Path("protonet/output/novelty_calibration.json"))
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    result = calibrate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved calibration to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
