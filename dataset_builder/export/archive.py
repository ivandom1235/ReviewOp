from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_ARCHIVE_NAME = "artifact.zip"


def write_artifact_zip(output_dir: str | Path, archive_name: str = DEFAULT_ARCHIVE_NAME) -> Path:
    output_dir = Path(output_dir)
    archive_path = output_dir / archive_name
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file() or path == archive_path:
                continue
            archive.write(path, arcname=path.relative_to(output_dir).as_posix())
    return archive_path
