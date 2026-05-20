from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ECProtoNetV2Config
from .io_utils import read_json, sha256_tree


class ArtifactGuardError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactGuardReport:
    artifact_dir: str
    artifact_status: str
    ready_for_protonet: bool
    stale_output_guard: str
    artifact_sha256: str
    failed_checks: list[str]
    active_contract_status: str
    raw_verification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "artifact_status": self.artifact_status,
            "ready_for_protonet": self.ready_for_protonet,
            "stale_output_guard": self.stale_output_guard,
            "artifact_sha256": self.artifact_sha256,
            "failed_checks": self.failed_checks,
            "active_contract_status": self.active_contract_status,
            "raw_verification": self.raw_verification,
        }


def _find_verification_file(artifact_dir: Path) -> Path | None:
    direct = artifact_dir / "artifact_verification.json"
    if direct.exists():
        return direct
    # Handles flattened or copied artifacts with suffix names.
    candidates = sorted(artifact_dir.glob("*artifact_verification.json"))
    if candidates:
        return candidates[0]
    return None


def assert_verified_artifact(artifact_dir: str | Path, config: ECProtoNetV2Config) -> ArtifactGuardReport:
    artifact_dir = Path(artifact_dir).resolve()
    if not artifact_dir.exists():
        raise ArtifactGuardError(f"Artifact directory does not exist: {artifact_dir}")

    verification_path = _find_verification_file(artifact_dir)
    if verification_path:
        verification = read_json(verification_path, default={}) or {}
    else:
        verification = {"artifact_status": "missing", "ready_for_protonet": False, "failed_checks": ["verification file missing"]}
    
    artifact_status = str(verification.get("artifact_status", "unknown"))
    ready = bool(verification.get("ready_for_protonet", False))
    failed_checks = list(verification.get("failed_checks", []) or [])
    memory_consistency = verify_memory_consistency(artifact_dir)
    if not memory_consistency.get("ok", False):
        failed_checks.append(memory_consistency.get("error", "memory consistency failed"))

    if config.require_artifact_pass and (artifact_status != "pass" or ready is not True or bool(failed_checks)):
        raise ArtifactGuardError(
            "Artifact is not ready for EC-ProtoNet V2.\n"
            f"artifact_dir={artifact_dir}\n"
            f"artifact_status={artifact_status}\n"
            f"ready_for_protonet={ready}\n"
            f"failed_checks={failed_checks}"
        )

    active_status = "not_required"
    if config.require_active_contract:
        contract_path = Path(config.active_contract_path).resolve()
        if not contract_path.exists():
            raise ArtifactGuardError(
                f"Missing active artifact contract: {contract_path}. "
                "Create CURRENT_ACTIVE_ARTIFACT.json after artifact verification passes."
            )
        contract = read_json(contract_path, default={}) or {}
        expected_raw = contract.get("active_dataset_artifact")
        if not expected_raw:
            raise ArtifactGuardError("CURRENT_ACTIVE_ARTIFACT.json has no active_dataset_artifact.")
        expected_path = Path(expected_raw)
        if not expected_path.is_absolute():
            expected_path = (contract_path.parent / expected_path).resolve()
        expected = expected_path
        if expected != artifact_dir:
            raise ArtifactGuardError(
                "Stale artifact path used.\n"
                f"Expected active artifact: {expected}\n"
                f"Actual artifact: {artifact_dir}"
            )
        if contract.get("artifact_status") != "pass" or contract.get("ready_for_protonet") is not True:
            raise ArtifactGuardError("Active artifact contract is not marked pass/ready_for_protonet=true.")
        active_status = "pass"

    return ArtifactGuardReport(
        artifact_dir=str(artifact_dir),
        artifact_status=artifact_status,
        ready_for_protonet=ready,
        stale_output_guard="pass" if active_status in {"pass", "not_required"} else "fail",
        artifact_sha256=sha256_tree(artifact_dir),
        failed_checks=failed_checks,
        active_contract_status=active_status,
        raw_verification=verification,
    )


def verify_memory_consistency(artifact_dir: Path) -> dict[str, Any]:
    summary = read_json(artifact_dir / "aspect_memory_summary.json", default={}) or {}
    promoted = read_json(artifact_dir / "aspect_memory_promoted.json", default={}) or {}

    if isinstance(promoted, dict):
        if isinstance(promoted.get("items"), list):
            promoted_items = promoted.get("items", [])
        elif isinstance(promoted.get("data"), list):
            promoted_items = promoted.get("data", [])
        else:
            promoted_items = []
    elif isinstance(promoted, list):
        promoted_items = promoted
    else:
        promoted_items = []

    promoted_count_file = len(promoted_items)
    promoted_count_summary = int(summary.get("promoted_count", summary.get("promoted_entries_total", 0)) or 0)

    if promoted_count_file != promoted_count_summary:
        return {
            "ok": False,
            "promoted_count": promoted_count_file,
            "error": (
                f"Memory inconsistency: summary promoted_count={promoted_count_summary}, "
                f"promoted file contains {promoted_count_file} items"
            ),
        }
    return {"ok": True, "promoted_count": promoted_count_file}
