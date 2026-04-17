from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
import os
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np
import torch
from dotenv import load_dotenv


PROTONET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROTONET_ROOT.parent

# Load .env from workspace root
load_dotenv(REPO_ROOT / ".env")
CODE_ROOT = PROTONET_ROOT / "code"
INPUT_ROOT = PROTONET_ROOT / "input"
OUTPUT_ROOT = PROTONET_ROOT / "output"
METADATA_ROOT = PROTONET_ROOT / "metadata"
BENCHMARK_INPUT_ROOT = REPO_ROOT / "dataset_builder" / "output" / "benchmark" / "ambiguity_grounded"


def env_value(*names: str, default: str | None = None) -> str | None:
    """Read an environment variable from a list of possible names."""
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def resolve_default_input_dir(input_type: str) -> Path:
    if input_type == "benchmark":
        return BENCHMARK_INPUT_ROOT
    return INPUT_ROOT / input_type


@dataclass
class ProtonetConfig:
    input_type: str = "benchmark"
    input_dir: Path = BENCHMARK_INPUT_ROOT
    output_dir: Path = OUTPUT_ROOT
    metadata_dir: Path = METADATA_ROOT
    checkpoint_dir: Path = OUTPUT_ROOT / "checkpoints"
    episode_cache_dir: Path = OUTPUT_ROOT / "episodes"
    predictions_dir: Path = OUTPUT_ROOT / "predictions"

    encoder_backend: str = "auto"
    encoder_model_name: str = env_value("REVIEWOP_PROTONET_ENCODER_MODEL", "PROTONET_ENCODER_MODEL", default="microsoft/deberta-v3-base") or "microsoft/deberta-v3-base"
    bow_dim: int = 512
    max_length: int = 160
    projection_dim: int = 256
    dropout: float = 0.15
    train_encoder: bool = True

    n_way: int = 3
    k_shot: int = 2
    q_query: int = 2
    max_train_episodes: int = 120
    max_eval_episodes: int = 48
    protocol_eval_enabled: bool = True
    protocol_eval_splits: tuple[str, ...] = ("random", "grouped", "domain_holdout")

    warmup_epochs: int = 1
    epochs: int = 12
    patience: int = 4
    learning_rate: float = 6e-4
    encoder_learning_rate: float = 1.2e-5
    weight_decay: float = 2.5e-4
    gradient_accumulation_steps: int = 2
    batch_size_hint: int = 1
    use_amp: bool = True
    contrastive_weight: float = 0.22
    contrastive_temperature: float = 0.2
    ortho_weight: float = 0.05
    focal_gamma: float = 1.8
    prototype_smoothing: float = 0.08
    low_confidence_threshold: float = 0.25
    top_k_debug: int = 3
    selective_alpha: float = 1.0
    selective_beta: float = 0.0
    selective_gamma: float = 0.0
    selective_delta: float = 0.0
    abstain_threshold: float = 0.01
    multi_label_margin: float = 0.10
    sentiment_pipeline: str = "both"
    novelty_threshold: float = 0.70
    novelty_known_threshold: float = 0.50
    novelty_novel_threshold: float = 0.80
    novelty_calibration_path: Path = METADATA_ROOT / "novelty_calibration_v2.json"
    runtime_cache_max_items: int = 20000

    seed: int = 42
    no_progress: bool = False
    force_rebuild_episodes: bool = False
    save_predictions: bool = True
    strict_encoder: bool = False
    production_require_transformer: bool = False
    allow_model_download: bool = False
    compile_model: bool = False

    joint_label_separator: str = "__"
    min_examples_per_label: int = 4
    temperature_init: float = 1.0

    def ensure_dirs(self) -> None:
        for path in [
            PROTONET_ROOT / "input" / "benchmark",
            self.output_dir,
            self.checkpoint_dir,
            self.episode_cache_dir,
            self.predictions_dir,
            self.metadata_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @cached_property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def progress_enabled(self) -> bool:
        return not self.no_progress

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["device"] = str(self.device)
        return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_file(input_dir: Path, split: str) -> Path:
    return input_dir / f"{split}.jsonl"
