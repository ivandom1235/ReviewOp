from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .ids import stable_group_id, stable_review_id
from .normalization import normalize_domain, normalize_metadata, normalize_text
from .schema_detect import infer_text_field
from ..schemas.raw_review import RawReview


def _is_absa_row(row: dict[str, Any]) -> bool:
    keys = set(row)
    return "id" in keys and bool({"aspect", "polarity"} & keys)

def _flatten_nested_aspects(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "aspects" in row and isinstance(row["aspects"], list):
        out = []
        for aspect_dict in row["aspects"]:
            flat = dict(row)
            del flat["aspects"]
            if "term" in aspect_dict:
                flat["aspect"] = aspect_dict["term"]
            flat.update(aspect_dict)
            out.append(flat)
        return out if out else [row]
    return [row]


def _domain_from_source(source_name: str) -> str:
    stem = Path(source_name).stem
    for suffix in ("_train", "_test", "_val", "-train", "-test", "-val"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return normalize_domain(stem)


def _raw_review_from_mapping(row: dict[str, Any], source_name: str) -> RawReview:
    text_field = infer_text_field(row)
    text = normalize_text(str(row[text_field]))
    normalized = {
        **row,
        "text": text,
        "source_name": source_name,
    }
    if _is_absa_row(row):
        normalized.setdefault("group_id", str(row.get("id") or "").strip())
        normalized.setdefault(
            "review_id",
            ":".join(
                str(part or "").strip()
                for part in (source_name, row.get("id"), row.get("aspect"), row.get("from"), row.get("to"))
            ),
        )
    domain = normalize_domain(row.get("domain"))
    if domain == "unknown" and _is_absa_row(row):
        domain = _domain_from_source(source_name)
    return RawReview(
        review_id=stable_review_id(normalized),
        group_id=stable_group_id(normalized),
        domain=domain,
        domain_family=normalize_domain(row.get("domain_family")) if row.get("domain_family") else domain,
        text=text,
        source_name=source_name,
        source_split=str(row.get("split") or "unknown"),
        metadata=normalize_metadata(row),
    )


def load_csv_reviews(path: str | Path) -> list[RawReview]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [_raw_review_from_mapping(row, path.name) for row in csv.DictReader(handle)]


def load_jsonl_reviews(path: str | Path) -> list[RawReview]:
    path = Path(path)
    rows: list[RawReview] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_no} in {path} is not an object")
            for flat_row in _flatten_nested_aspects(payload):
                rows.append(_raw_review_from_mapping(flat_row, path.name))
    return rows


def load_hf_dataset(*_args: Any, **_kwargs: Any) -> list[RawReview]:
    raise RuntimeError("Hugging Face loading is intentionally optional in P0")
