from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_INPUT_DIR = Path("dataset_builder/input")
DEFAULT_OUTPUT_DIR = Path("dataset_builder/output")
SUPPORTED_LLM_PROVIDERS = {"none", "openai", "groq"}

def get_default_llm_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-5-nano")

def get_default_llm_provider() -> str:
    return os.environ.get("REVIEWOP_DEFAULT_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "none"))

def get_env_model(provider: str, current_model: str | None = None) -> str:
    """Gets the model for a provider, checking env vars for overrides."""
    # Use current_model as fallback if provided, otherwise use global default
    fallback = current_model if current_model and current_model != "gpt-5-nano" else get_default_llm_model()
    
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", fallback)
    elif provider == "groq":
        return os.environ.get("GROQ_MODEL", fallback)
    elif provider == "claude":
        return os.environ.get("CLAUDE_MODEL", fallback)
    elif provider == "ollama":
        return os.environ.get("OLLAMA_MODEL", fallback)
    return os.environ.get("LLM_MODEL", fallback)


@dataclass(frozen=True)
class BuilderConfig:
    input_dir: Path = DEFAULT_INPUT_DIR
    input_paths: tuple[Path, ...] = ()
    output_dir: Path = DEFAULT_OUTPUT_DIR
    random_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    sample_size: int | None = None
    chunk_size: int | None = None
    chunk_offset: int = 0
    min_confidence_train: float = 0.5
    llm_provider: str = "none" # Default will be handled in load_config or __post_init__
    llm_model: str = "gpt-5-nano"
    dry_run: bool = False
    overwrite: bool = False
    use_cache: bool = True
    symptom_store_path: Optional[str] = None


def load_config(path: str | Path | None = None) -> BuilderConfig:
    import json
    payload = {}
    if path and Path(path).exists():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    
    # Prioritize: 1. JSON config file, 2. Env vars, 3. Defaults
    llm_provider = str(payload.get("llm_provider", get_default_llm_provider()))
    llm_model = str(payload.get("llm_model", get_env_model(llm_provider)))

    return BuilderConfig(
        input_dir=Path(payload.get("input_dir", DEFAULT_INPUT_DIR)),
        input_paths=tuple(Path(value) for value in payload.get("input_paths", [])),
        output_dir=Path(payload.get("output_dir", DEFAULT_OUTPUT_DIR)),
        random_seed=int(payload.get("random_seed", 42)),
        train_ratio=float(payload.get("train_ratio", 0.8)),
        val_ratio=float(payload.get("val_ratio", 0.1)),
        test_ratio=float(payload.get("test_ratio", 0.1)),
        sample_size=None if payload.get("sample_size") is None else int(payload["sample_size"]),
        chunk_size=None if payload.get("chunk_size") is None else int(payload["chunk_size"]),
        chunk_offset=int(payload.get("chunk_offset", 0)),
        min_confidence_train=float(payload.get("min_confidence_train", 0.5)),
        llm_provider=llm_provider,
        llm_model=llm_model,
        dry_run=bool(payload.get("dry_run", False)),
        overwrite=bool(payload.get("overwrite", False)),
        use_cache=bool(payload.get("use_cache", True)),
        symptom_store_path=payload.get("symptom_store_path"),
    )


def validate_config(cfg: BuilderConfig) -> None:
    ratios = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    if cfg.min_confidence_train < 0 or cfg.min_confidence_train > 1:
        raise ValueError("min_confidence_train must be in [0, 1]")
    if cfg.llm_provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(f"unsupported llm_provider: {cfg.llm_provider}")
    if cfg.sample_size is not None and cfg.sample_size < 0:
        raise ValueError("sample_size must be >= 0")
    if cfg.chunk_size is not None and cfg.chunk_size < 0:
        raise ValueError("chunk_size must be >= 0")
    if cfg.chunk_offset < 0:
        raise ValueError("chunk_offset must be >= 0")


def to_jsonable(cfg: BuilderConfig) -> dict[str, Any]:
    return {
        "input_dir": str(cfg.input_dir),
        "input_paths": [str(path) for path in cfg.input_paths],
        "output_dir": str(cfg.output_dir),
        "random_seed": cfg.random_seed,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
        "sample_size": cfg.sample_size,
        "chunk_size": cfg.chunk_size,
        "chunk_offset": cfg.chunk_offset,
        "min_confidence_train": cfg.min_confidence_train,
        "llm_provider": cfg.llm_provider,
        "llm_model": cfg.llm_model,
        "dry_run": cfg.dry_run,
        "overwrite": cfg.overwrite,
        "use_cache": cfg.use_cache,
        "symptom_store_path": cfg.symptom_store_path,
    }
