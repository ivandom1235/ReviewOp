from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_METADATA = {
    "metrics_schema_version",
    "artifact_dir",
    "artifact_status",
    "ready_for_protonet",
    "stale_output_guard",
    "artifact_sha256",
}
EXPECTED_SCHEMA = "ec_v2_eval_v1"


def verify_run(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    metadata_path = output_dir / "run_metadata.json"
    metrics_path = output_dir / "metrics.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Missing run_metadata.json in {output_dir}")
    if not metrics_path.exists():
        raise RuntimeError(f"Missing metrics.json in {output_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise RuntimeError(f"Missing metadata fields: {missing}")
    if metadata.get("metrics_schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError(
            f"metrics_schema_version is {metadata.get('metrics_schema_version')}, expected {EXPECTED_SCHEMA}"
        )
    if metadata["artifact_status"] != "pass":
        raise RuntimeError("artifact_status is not pass")
    if metadata["ready_for_protonet"] is not True:
        raise RuntimeError("ready_for_protonet is not true")
    if metadata["stale_output_guard"] != "pass":
        raise RuntimeError("stale_output_guard is not pass")
    if int(metadata.get("prototype_count", 0)) <= 0:
        raise RuntimeError("prototype_count must be > 0")
    if int(metadata.get("prediction_count", 0)) <= 0:
        raise RuntimeError("prediction_count must be > 0")
    if float(metrics.get("coverage", 0.0)) > 1.0:
        raise RuntimeError("coverage > 1.0")
    return {"status": "pass", "metadata": metadata, "metric_keys": sorted(metrics.keys())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(verify_run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
