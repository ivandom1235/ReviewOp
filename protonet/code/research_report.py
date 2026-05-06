from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _synthetic_selective_score(row: dict[str, Any], *, ablation: str) -> float:
    confidence = _safe_float(row.get("confidence"), 0.0)
    ambiguity = _safe_float(row.get("benchmark_ambiguity_score", row.get("ambiguity_score", 0.0)))
    novelty = _safe_float(row.get("novelty_score", 0.0))
    contradiction = _safe_float(row.get("contradiction_score", 0.0))
    evidence_support = 1.0 if row.get("post_aspect_selected_aspects") else 0.0

    if ablation == "no_evidence":
        evidence_support = 0.0
    if ablation == "no_ambiguity":
        ambiguity = 0.0
    if ablation == "no_novelty":
        novelty = 0.0
    if ablation == "no_graph":
        contradiction = 0.0

    score = (
        0.35 * confidence
        + 0.20 * evidence_support
        - 0.10 * ambiguity
        - 0.10 * novelty
        - 0.10 * contradiction
    )
    return max(0.0, min(1.0, score))


def _counterfactual_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("counterfactual_group_id") or "").strip()
        if group_id:
            groups[group_id].append(row)

    evaluated = 0
    consistent = 0
    for group_rows in groups.values():
        if len(group_rows) < 2:
            continue
        roles = {str(row.get("counterfactual_role") or "").strip() for row in group_rows}
        if not roles.intersection({"original", "counterfactual"}):
            continue
        evaluated += 1
        predicted_labels = {str(row.get("pred_label") or "").strip() for row in group_rows if str(row.get("pred_label") or "").strip()}
        if len(predicted_labels) >= 2:
            consistent += 1
            continue
        if any(bool(row.get("post_aspect_abstained")) for row in group_rows):
            consistent += 1

    return {
        "evaluated_groups": evaluated,
        "consistent_groups": consistent,
        "consistency_rate": float(consistent / evaluated) if evaluated else None,
    }


def _abstention_stats(rows: list[dict[str, Any]], *, ablation: str) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"coverage": 0.0, "abstention_precision": 0.0, "abstention_recall": 0.0, "accepted_accuracy": 0.0}

    accepted = 0
    abstained = 0
    correct_accepted = 0
    abstained_incorrect = 0
    incorrect_total = 0

    for row in rows:
        score = _synthetic_selective_score(row, ablation=ablation)
        abstain_threshold = 0.25
        contradiction = _safe_float(row.get("contradiction_score", 0.0))
        if ablation != "no_graph" and contradiction >= 0.45:
            abstain = True
        else:
            abstain = score < abstain_threshold or bool(row.get("abstained"))
        correct = bool(row.get("correct"))
        if not correct:
            incorrect_total += 1
        if abstain:
            abstained += 1
            if not correct:
                abstained_incorrect += 1
        else:
            accepted += 1
            if correct:
                correct_accepted += 1

    abstention_precision = float(abstained_incorrect / abstained) if abstained else 0.0
    abstention_recall = float(abstained_incorrect / incorrect_total) if incorrect_total else 0.0
    accepted_accuracy = float(correct_accepted / accepted) if accepted else 0.0
    return {
        "coverage": float(accepted / total),
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "accepted_accuracy": accepted_accuracy,
    }


def build_research_pack(
    *,
    report_dir: Path,
    predictions_dir: Path | None = None,
) -> dict[str, Any]:
    report = _load_json(report_dir / "report.json")
    predictions_dir = predictions_dir or (report_dir.parent / "output" / "predictions")
    input_summary = report.get("input_summary", {}) if isinstance(report.get("input_summary", {}), dict) else {}
    extra_artifacts = input_summary.get("extra_artifacts", {}) if isinstance(input_summary.get("extra_artifacts", {}), dict) else {}
    splits = {}
    for split in ("train", "val", "test", "domain_holdout"):
        rows = _load_jsonl(predictions_dir / f"{split}_predictions.jsonl")
        if rows:
            splits[split] = {
                "rows": len(rows),
                "metrics": _abstention_stats(rows, ablation="baseline"),
                "counterfactual": _counterfactual_consistency(rows),
            }

    all_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test", "domain_holdout"):
        all_rows.extend(_load_jsonl(predictions_dir / f"{split}_predictions.jsonl"))

    ablations = {}
    for ablation in ("baseline", "no_evidence", "no_ambiguity", "no_novelty", "no_graph"):
        rows = []
        for split_name in ("val", "test", "domain_holdout"):
            rows.extend(_load_jsonl(predictions_dir / f"{split_name}_predictions.jsonl"))
        ablations[ablation] = _abstention_stats(rows, ablation=ablation)

    pack = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_dir / "report.json"),
        "config": report.get("config", {}),
        "splits": splits,
        "ablations": ablations,
        "counterfactual": {
            "from_report": extra_artifacts.get("counterfactual_pairs", {}),
            "observed": _counterfactual_consistency(all_rows),
        },
        "paper_readiness": {
            "dataset_contract": bool(report.get("metrics", {})),
            "ablation_suite": bool(ablations),
            "counterfactual_eval": bool(_counterfactual_consistency(all_rows).get("evaluated_groups")),
            "domain_holdout": "domain_holdout" in splits,
        },
    }
    return pack


def write_research_pack(pack: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    lines = [
        "# ReviewOp Research Pack",
        "",
        f"- Generated at: {pack.get('generated_at')}",
        f"- Source report: {pack.get('source_report')}",
        "",
        "## Split Summary",
    ]
    for split, payload in pack.get("splits", {}).items():
        metrics = payload.get("metrics", {})
        lines.extend(
            [
                f"- {split}: rows={payload.get('rows', 0)} coverage={metrics.get('coverage', 0.0):.3f} accepted_accuracy={metrics.get('accepted_accuracy', 0.0):.3f}",
            ]
        )
    lines.append("")
    lines.append("## Ablations")
    for ablation, metrics in pack.get("ablations", {}).items():
        lines.append(
            f"- {ablation}: coverage={metrics.get('coverage', 0.0):.3f} abstention_precision={metrics.get('abstention_precision', 0.0):.3f} accepted_accuracy={metrics.get('accepted_accuracy', 0.0):.3f}"
        )
    lines.append("")
    lines.append("## Paper Readiness")
    for key, value in pack.get("paper_readiness", {}).items():
        lines.append(f"- {key}: {value}")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a paper-facing research pack from ReviewOp evaluation outputs.")
    parser.add_argument("--report-dir", type=Path, required=True, help="Protonet metadata directory containing report.json")
    parser.add_argument("--predictions-dir", type=Path, default=None, help="Directory containing *_predictions.jsonl files")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (defaults to report_dir/research_pack.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or (args.report_dir / "research_pack.json")
    pack = build_research_pack(report_dir=args.report_dir, predictions_dir=args.predictions_dir)
    write_research_pack(pack, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
