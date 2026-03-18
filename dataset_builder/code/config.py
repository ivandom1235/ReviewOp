"""Configuration for dataset builder."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class BuilderConfig:
    input_dir: Path
    output_dir: Path
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    random_seed: int = 42
    max_aspects_per_review: int = 5
    confidence_threshold: float = 0.35
    prefer_canonical: bool = True
    dry_run: bool = False
    sample_preview_count: int = 3


DEFAULT_CONFIG_PATH = Path("dataset_builder") / "config.json"
