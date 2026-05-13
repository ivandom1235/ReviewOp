from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def finalize_artifact_status(artifact_dir: str | Path) -> dict:
    artifact_dir = Path(artifact_dir)
    verification_path = artifact_dir / "artifact_verification.json"
    manifest_path = artifact_dir / "manifest.json"
    if not verification_path.exists():
        raise RuntimeError(f"Missing artifact_verification.json: {verification_path}")
    verification = load_json(verification_path, {})
    manifest = load_json(manifest_path, {})

    artifact_status = verification.get("artifact_status", "unknown")
    ready = bool(verification.get("ready_for_protonet", False))
    failed_checks = list(verification.get("failed_checks", []) or [])

    final_release_status = "passed" if artifact_status == "pass" and ready else "failed"
    manifest["final_release_status"] = final_release_status
    manifest["artifact_status"] = artifact_status
    manifest["ready_for_protonet"] = ready
    manifest["failed_checks"] = failed_checks
    manifest["source_of_truth"] = "artifact_verification.json"
    manifest["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()

    write_json(manifest_path, manifest)
    final_marker = {
        "final_release_status": final_release_status,
        "artifact_status": artifact_status,
        "ready_for_protonet": ready,
        "failed_checks": failed_checks,
        "source_of_truth": "artifact_verification.json",
    }
    write_json(artifact_dir / "final_release_status.json", final_marker)
    return final_marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir")
    args = parser.parse_args()
    print(json.dumps(finalize_artifact_status(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
