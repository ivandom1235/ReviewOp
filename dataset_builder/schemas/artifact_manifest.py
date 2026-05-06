from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactManifest:
    version: str
    dataset_inputs: list[str]
    profile_summary: dict[str, Any]
    policies_used: dict[str, Any]
    split_summary: dict[str, int]
    release_status: str
    gate_status: str = "UNKNOWN"
    
    # Reproducibility metadata
    run_id: str = ""
    run_command: str = ""
    code_hash: str = ""
    config_hash: str = ""
    artifact_created_at: str = ""
    artifact_version: str = "1.0.0"
    
    # Sample size verification
    sample_size_requested: int = 0
    sample_size_loaded: int = 0
    
    artifact_checksums: dict[str, str] = field(default_factory=dict)
