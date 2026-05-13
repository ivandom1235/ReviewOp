from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_active_contract(artifact_dir: str | Path, output: str | Path = "CURRENT_ACTIVE_ARTIFACT.json") -> dict:
    artifact_dir = Path(artifact_dir).resolve()
    verification = load_json(artifact_dir / "artifact_verification.json", {})
    if verification.get("artifact_status") != "pass" or verification.get("ready_for_protonet") is not True:
        raise RuntimeError(
            "Refusing to create CURRENT_ACTIVE_ARTIFACT.json for a failed/non-ready artifact.\n"
            f"artifact_status={verification.get('artifact_status')}\n"
            f"ready_for_protonet={verification.get('ready_for_protonet')}\n"
            f"failed_checks={verification.get('failed_checks')}"
        )
    contract = {
        "active_dataset_artifact": str(artifact_dir),
        "artifact_status": "pass",
        "ready_for_protonet": True,
        "metrics_schema_version": "dataset_builder_v1",
        "source_of_truth": "artifact_verification.json",
        "do_not_use": [
            "dataset_builder/output",
            "dataset_builder/output/run_stability_candidate",
            "dataset_builder/output/run_stability_candidate_seed1",
            "dataset_builder/output/run_800_final",
            "dataset_builder/output/run_400_stability",
            "dataset_builder/output/_tmp_400_row_fix",
        ],
    }
    output = Path(output)
    output.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", default="CURRENT_ACTIVE_ARTIFACT.json")
    args = parser.parse_args()
    print(json.dumps(write_active_contract(args.artifact_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
