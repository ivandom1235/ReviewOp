from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np
import torch


PROTONET_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROTONET_ROOT / "code"
INPUT_ROOT = PROTONET_ROOT / "input"
OUTPUT_ROOT = PROTONET_ROOT / "output"
METADATA_ROOT = PROTONET_ROOT / "metadata"


@dataclass
class ProtonetConfig:
    input_type: str = "episodic"
    input_dir: Path = INPUT_ROOT / "episodic"
    output_dir: Path = OUTPUT_ROOT
    metadata_dir: Path = METADATA_ROOT
    checkpoint_dir: Path = OUTPUT_ROOT / "checkpoints"
    episode_cache_dir: Path = OUTPUT_ROOT / "episodes"
    predictions_dir: Path = OUTPUT_ROOT / "predictions"

    encoder_backend: str = "auto"
    encoder_model_name: str = "microsoft/deberta-v3-base"
    bow_dim: int = 512
    max_length: int = 160
    projection_dim: int = 256
    dropout: float = 0.1
    train_encoder: bool = True

    n_way: int = 3
    k_shot: int = 2
    q_query: int = 2
    max_train_episodes: int = 120
    max_eval_episodes: int = 48

    warmup_epochs: int = 1
    epochs: int = 8
    patience: int = 3
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 2
    batch_size_hint: int = 1
    use_amp: bool = True
    contrastive_weight: float = 0.15
    prototype_smoothing: float = 0.05
    low_confidence_threshold: float = 0.55
    top_k_debug: int = 3

    seed: int = 42
    no_progress: bool = False
    force_rebuild_episodes: bool = False
    save_predictions: bool = True
    strict_encoder: bool = False
    production_require_transformer: bool = False
    allow_model_download: bool = False

    joint_label_separator: str = "__"
    min_examples_per_label: int = 4
    temperature_init: float = 1.0

    def ensure_dirs(self) -> None:
        for path in [
            PROTONET_ROOT / "input" / "reviewlevel",
            PROTONET_ROOT / "input" / "episodic",
            self.output_dir,
            self.checkpoint_dir,
            self.episode_cache_dir,
            self.predictions_dir,
            self.metadata_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def progress_enabled(self) -> bool:
        return not self.no_progress

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["device"] = self.device
        return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_file(input_dir: Path, split: str) -> Path:
    return input_dir / f"{split}.jsonl"
