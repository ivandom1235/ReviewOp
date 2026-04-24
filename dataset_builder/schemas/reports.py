from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QualityReport:
    total_exported: int
    export_counts: dict[str, int]
    original_sample_size: int = 0
    total_discarded: int = 0
    rejected_interpretations: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticsReport:
    gate_failures: dict[str, int]
    broad_label_drops: int = 0
    fragment_drops: int = 0
    recovery_stats: dict[str, Any] = field(default_factory=dict)
