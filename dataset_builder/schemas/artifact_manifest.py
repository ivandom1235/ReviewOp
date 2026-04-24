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
    artifact_checksums: dict[str, str] = field(default_factory=dict)
