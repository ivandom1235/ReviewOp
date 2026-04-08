from __future__ import annotations

import argparse
import hashlib
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


def _snapshot_hash(rows: list[dict[str, Any]]) -> str:
    basis = []
    for row in rows:
        basis.append(
            (
                str(row.get("true_label") or ""),
                str(row.get("pred_label") or ""),
                round(float(row.get("confidence", 0.0)), 6),
                round(float(row.get("novelty_score", 0.0)), 6),
                int(bool(row.get("novel_acceptable", False))),
            )
        )
    payload = json.dumps(basis, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def calibrate_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "version": "v2",
            "scorer": "distance_energy",
            "thresholds": {"T_known": 0.35, "T_novel": 0.65},
            "best_open_set_f1": 0.0,
            "validation_snapshot_hash": "",
            "sweep": [],
        }

    truth = np.asarray([1 if bool(row.get("novel_acceptable", False)) else 0 for row in rows], dtype=int)
    has_known = bool(np.any(truth == 0))
    has_novel = bool(np.any(truth == 1))
    if not has_known or not has_novel:
        return {
            "version": "v2",
            "scorer": "distance_ambiguity_energy",
            "thresholds": {"T_known": 0.35, "T_novel": 0.65},
            "best_open_set_f1": 0.0,
            "validation_snapshot_hash": _snapshot_hash(rows),
            "sweep": [],
            "warning": "Calibration skipped: validation must contain both known and novel positives.",
            "not_applicable": True,
        }
    scores = np.asarray([float(row.get("novelty_score", 0.0)) for row in rows], dtype=float)

    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    sweep: list[dict[str, float]] = []
    best = {"objective": -1.0, "f1": -1.0, "T_known": 0.35, "T_novel": 0.65}
    for t_known in candidate_thresholds:
        for t_novel in candidate_thresholds:
            if float(t_known) >= float(t_novel):
                continue
            preds = []
            for score in scores:
                if score <= float(t_known):
                    preds.append(0)
                elif score >= float(t_novel):
                    preds.append(1)
                else:
                    preds.append(-1)
            tp = sum(1 for pred, gold in zip(preds, truth) if pred == 1 and gold == 1)
            fp = sum(1 for pred, gold in zip(preds, truth) if pred == 1 and gold == 0)
            fn = sum(1 for pred, gold in zip(preds, truth) if pred != 1 and gold == 1)
            f1 = _f1(tp, fp, fn)
            abstain_rate = sum(1 for pred in preds if pred == -1) / max(1, len(preds))
            boundary_rows = [rows[idx] for idx, pred in enumerate(preds) if pred == -1]
            boundary_abstain_quality = float(
                np.mean([1.0 if bool(row.get("abstain_acceptable", False)) else 0.0 for row in boundary_rows])
            ) if boundary_rows else 0.0
            known_tp = sum(1 for pred, gold in zip(preds, truth) if pred == 0 and gold == 0)
            known_fn = sum(1 for pred, gold in zip(preds, truth) if pred != 0 and gold == 0)
            known_recall = known_tp / max(1, known_tp + known_fn)
            objective = 0.5 * float(f1) + 0.3 * float(boundary_abstain_quality) + 0.2 * float(known_recall)
            scorecard = {
                "T_known": float(t_known),
                "T_novel": float(t_novel),
                "novel_f1": float(f1),
                "boundary_abstain_quality": float(boundary_abstain_quality),
                "known_recall": float(known_recall),
                "objective": float(objective),
                "abstain_rate": float(round(abstain_rate, 4)),
            }
            sweep.append(scorecard)
            if objective > best["objective"]:
                best = {"objective": float(objective), "f1": float(f1), "T_known": float(t_known), "T_novel": float(t_novel)}

    return {
        "version": "v2",
        "scorer": "distance_ambiguity_energy",
        "thresholds": {"T_known": float(best["T_known"]), "T_novel": float(best["T_novel"])},
        "best_open_set_f1": float(max(0.0, best["f1"])),
        "best_objective": float(max(0.0, best["objective"])),
        "validation_snapshot_hash": _snapshot_hash(rows),
        "sweep": sweep,
        "not_applicable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate V2 novelty thresholds from validation predictions.")
    parser.add_argument("--input", type=Path, required=True, help="Path to val prediction jsonl.")
    parser.add_argument("--output", type=Path, default=Path("protonet/metadata/novelty_calibration_v2.json"))
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    result = calibrate_thresholds(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved calibration to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
