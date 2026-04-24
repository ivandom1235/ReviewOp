from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QualityReport:
    total_exported: int
    export_counts: dict[str, int]
    requested_rows: int = 0
    loaded_rows: int = 0
    processed_rows: int = 0
    rejected_rows: int = 0
    discarded_rows: int = 0
    mapping_source_distribution: dict[str, int] = field(default_factory=dict)
    original_sample_size: int = 0
    total_discarded: int = 0
    rejected_interpretations: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    source_type_distribution: dict[str, int] = field(default_factory=dict)
    label_type_distribution: dict[str, int] = field(default_factory=dict)
    novelty_distribution: dict[str, int] = field(default_factory=dict)
    hardness_distribution: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, float] = field(default_factory=dict)
    canonicalization: dict[str, float] = field(default_factory=dict)
    gold_stats: dict[str, float] = field(default_factory=dict)
    accounting_valid: bool = True


@dataclass(frozen=True)
class DiagnosticsReport:
    gate_failures: dict[str, int]
    broad_label_drops: int = 0
    fragment_drops: int = 0
    recovery_stats: dict[str, Any] = field(default_factory=dict)
