from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_builder.evidence.span_validator import validate_span
from dataset_builder.split.leakage_checks import check_group_leakage, check_text_duplication


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(payload)
    return rows


def _validate_row(row: dict, split: str, index: int) -> None:
    row_ref = f"{split}[{index}]"
    if not str(row.get("group_id") or "").strip():
        raise ValueError(f"{row_ref} has unresolved group_id")
    review_text = str(row.get("review_text") or "")
    if not review_text.strip():
        raise ValueError(f"{row_ref} has empty review_text")
    interpretations = row.get("gold_interpretations")
    if not isinstance(interpretations, list) or not interpretations:
        raise ValueError(f"{row_ref} has no gold_interpretations")
    for interp_index, interp in enumerate(interpretations):
        if not isinstance(interp, dict):
            raise ValueError(f"{row_ref}.gold_interpretations[{interp_index}] is not an object")
        evidence_text = str(interp.get("evidence_text") or "")
        evidence_span = interp.get("evidence_span")
        result = validate_span(review_text, evidence_text, evidence_span)
        if not result.valid:
            raise ValueError(f"{row_ref}.gold_interpretations[{interp_index}] invalid evidence span: {result.reason_codes}")


def validate_artifact(path: str | Path) -> dict[str, int]:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    quality_path = root / "quality_report.json"
    if not quality_path.exists():
        raise FileNotFoundError(quality_path)
    counts: dict[str, int] = {}
    rows_by_split: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        split_path = root / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        rows = _load_jsonl(split_path)
        rows_by_split[split] = rows
        counts[split] = len(rows)
        if not rows:
            raise ValueError(f"{split} split is empty")
        for index, row in enumerate(rows):
            _validate_row(row, split, index)
    if sum(counts.values()) <= 0:
        raise ValueError("artifact is empty")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    expected_counts = quality.get("export_counts")
    if expected_counts and {split: int(expected_counts.get(split, -1)) for split in ("train", "val", "test")} != counts:
        raise ValueError("quality report export_counts do not match split files")
    if int(quality.get("total_rows", sum(counts.values()))) != sum(counts.values()):
        raise ValueError("quality report total_rows does not match split files")
    group_leakage = check_group_leakage(rows_by_split)
    if int(group_leakage["grouped_leakage"]) != 0:
        raise ValueError("group leakage detected")
    text_leakage = check_text_duplication(rows_by_split)
    if int(text_leakage["exact_text_leakage"]) != 0:
        raise ValueError("exact text leakage detected")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir")
    args = parser.parse_args()
    print(validate_artifact(args.artifact_dir))


if __name__ == "__main__":
    main()
