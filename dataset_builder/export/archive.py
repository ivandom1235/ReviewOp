from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_ARCHIVE_NAME = "artifact.zip"


def write_artifact_zip(output_dir: str | Path, archive_name: str = DEFAULT_ARCHIVE_NAME) -> Path:
    output_dir = Path(output_dir)
    archive_path = output_dir / archive_name
    artifact_names = ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "quality_report.json")
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name in artifact_names:
            path = output_dir / name
            if not path.exists():
                raise FileNotFoundError(path)
            archive.write(path, arcname=name)
    return archive_path
