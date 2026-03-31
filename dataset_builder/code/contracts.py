from __future__ import annotations

from dataclasses import dataclass, field
import shutil
from pathlib import Path
from typing import Any


@dataclass
class BuilderConfig:
    input_dir: Path = Path(__file__).resolve().parents[1] / "input"
    output_dir: Path = Path(__file__).resolve().parents[1] / "output"
    reports_subdir: str = "reports"
    random_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    text_column_override: str | None = None
    sample_size: int | None = None
    chunk_size: int | None = None
    chunk_offset: int = 0
    dry_run: bool = False
    preview_only: bool = False
    confidence_threshold: float = 0.6
    max_aspects: int = 20
    min_text_tokens: int = 4
    implicit_mode: str = "heuristic"
    benchmark_key: str | None = None
    model_family: str = "heuristic_latent"
    augmentation_mode: str = "none"
    prompt_mode: str = "constrained"
    reset_output: bool = True

    @property
    def explicit_dir(self) -> Path:
        return self.output_dir / "explicit"

    @property
    def implicit_dir(self) -> Path:
        return self.output_dir / "implicit"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / self.reports_subdir

    def ensure_dirs(self, *, reset_output: bool | None = None) -> None:
        should_reset = self.reset_output if reset_output is None else reset_output
        if should_reset and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        for path in (self.output_dir, self.explicit_dir, self.implicit_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ReviewRecord:
    id: str
    split: str
    source_file: str
    source_text: str
    domain: str
    language: str = "en"
    gold_labels: list[dict[str, Any]] = field(default_factory=list)
    explicit: dict[str, Any] = field(default_factory=dict)
    implicit: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
