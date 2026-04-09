from __future__ import annotations

import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZIP_DEFLATED, ZipFile


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "")).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def token_count(text: str) -> int:
    return len(tokenize(text))


def split_sentences(text: str) -> list[str]:
    clean = normalize_whitespace(text)
    if not clean:
        return []
    parts = [part.strip() for part in SENTENCE_RE.split(clean) if part.strip()]
    return parts or [clean]


def stable_id(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted([to_jsonable(item) for item in value])
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compress_output_folder(output_dir: Path) -> Path | None:
    """Compresses the output folder into a ZIP file in the sibling 'zip' directory."""
    if not output_dir.exists():
        return None
    zip_root = output_dir.parent / "zip"
    zip_root.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_base = f"output_{timestamp}"
    zip_target = zip_root / zip_base
    
    archive_path = shutil.make_archive(str(zip_target), "zip", output_dir)
    return Path(archive_path)


def compress_dataset_artifacts(output_dir: Path) -> Path | None:
    """
    Compress train/val/test benchmark files + reports into output/zip.
    Returns the generated zip path, or None when no expected files exist.
    """
    benchmark_dir = output_dir / "benchmark" / "ambiguity_grounded"
    reports_dir = output_dir / "reports"
    zip_root = output_dir / "zip"

    files_to_pack = [
        benchmark_dir / "train.jsonl",
        benchmark_dir / "val.jsonl",
        benchmark_dir / "test.jsonl",
        benchmark_dir / "metadata.json",
        benchmark_dir / "review_queue.jsonl",
        reports_dir / "build_report.json",
        reports_dir / "data_quality_report.json",
        reports_dir / "benchmark_v2_novelty_report.json",
        reports_dir / "research_manifest.json",
    ]
    existing_files = [path for path in files_to_pack if path.exists()]
    if not existing_files:
        return None

    zip_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = zip_root / f"dataset_artifacts_{timestamp}.zip"

    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for file_path in existing_files:
            archive.write(file_path, arcname=file_path.name)
    return zip_path
