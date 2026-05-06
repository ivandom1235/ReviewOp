from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_INPUT_DIR = Path("dataset_builder/input")
DEFAULT_OUTPUT_DIR = Path("dataset_builder/output")
SUPPORTED_LLM_PROVIDERS = {"none", "openai", "groq", "claude", "anthropic", "openrouter", "huggingface", "ollama", "lightning", "gemini"}

def get_default_llm_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-5-nano")

def get_default_llm_provider() -> str:
    return os.environ.get("REVIEWOP_DEFAULT_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "none"))

def get_env_model(provider: str, current_model: str | None = None) -> str:
    """Gets the model for a provider, checking env vars for overrides."""
    # Use current_model as fallback if provided, otherwise use global default
    fallback = current_model if current_model and current_model != "gpt-5-nano" else get_default_llm_model()
    
    if provider == "openai":
        val = os.environ.get("OPENAI_MODEL")
        return val.strip() if val else fallback
    elif provider == "groq":
        val = os.environ.get("GROQ_MODEL")
        return val.strip() if val else fallback
    elif provider in ("claude", "anthropic"):
        val = os.environ.get("CLAUDE_MODEL") or os.environ.get("ANTHROPIC_MODEL")
        return val.strip() if val else "claude-sonnet-4-6"
    elif provider == "gemini":
        val = os.environ.get("GEMINI_MODEL")
        return val.strip() if val else "gemini-3.1-flash-lite-preview"
    elif provider == "openrouter":
        val = os.environ.get("OPENROUTER_MODEL")
        return val.strip() if val else "openrouter/free"
    elif provider == "huggingface":
        val = os.environ.get("HUGGINGFACE_MODEL")
        return val.strip() if val else "mistralai/Mistral-7B-Instruct-v0.3"
    elif provider == "ollama":
        val = os.environ.get("OLLAMA_MODEL")
        return val.strip() if val else fallback
    elif provider == "lightning":
        val = os.environ.get("LIGHTNING_MODEL")
        return val.strip() if val else "lightning-ai/gpt-oss-20b"
    val = os.environ.get("LLM_MODEL")
    return val.strip() if val else fallback


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
    strict: bool = False
    symptom_store_path: Optional[str] = None
    domain_mode: str = "full"
    provisional_policy: str = "strict"
    evidence_window_tokens: int = 8
    aspect_memory_auto_promote: bool = False
    aspect_memory_review_queue_min_support: int = 3
    aspect_memory_review_queue_min_reviews: int = 3
    aspect_memory_review_queue_min_surface_forms: int = 2
    aspect_memory_path: Optional[str] = None
    domain_holdout_domain: Optional[str] = None
    max_workers: int = 20


def load_config(path: str | Path | None = None) -> BuilderConfig:
    import json
    payload = {}
    if path and Path(path).exists():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    
    # Prioritize: 1. JSON config file, 2. Env vars, 3. Defaults
    llm_provider = str(payload.get("llm_provider", get_default_llm_provider())).strip()
    llm_model = str(payload.get("llm_model", get_env_model(llm_provider))).strip()

    # Tune max_workers based on provider to avoid rate limits
    max_workers = int(payload.get("max_workers", 20))
    if llm_provider == "groq" and "max_workers" not in payload:
        max_workers = 4 # Conservative for Groq 6000 TPM limit
    elif llm_provider == "gemini" and "max_workers" not in payload:
        max_workers = 10 # Conservative for Gemini free tier

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
        strict=bool(payload.get("strict", False)),
        symptom_store_path=payload.get("symptom_store_path"),
        domain_mode=str(payload.get("domain_mode", "full")),
        provisional_policy=str(payload.get("provisional_policy", "strict")),
        evidence_window_tokens=int(payload.get("evidence_window_tokens", 8)),
        aspect_memory_auto_promote=bool(payload.get("aspect_memory_auto_promote", False)),
        aspect_memory_review_queue_min_support=int(payload.get("aspect_memory_review_queue_min_support", 3)),
        aspect_memory_review_queue_min_reviews=int(payload.get("aspect_memory_review_queue_min_reviews", 3)),
        aspect_memory_review_queue_min_surface_forms=int(payload.get("aspect_memory_review_queue_min_surface_forms", 2)),
        aspect_memory_path=payload.get("aspect_memory_path"),
        domain_holdout_domain=payload.get("domain_holdout_domain"),
        max_workers=max_workers,
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
    if cfg.domain_mode not in {"generic_only", "generic_plus_domain", "generic_plus_learned", "full"}:
        raise ValueError(f"unsupported domain_mode: {cfg.domain_mode}")
    if cfg.provisional_policy not in {"loose", "strict", "memory_only"}:
        raise ValueError(f"unsupported provisional_policy: {cfg.provisional_policy}")


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
        "strict": cfg.strict,
        "symptom_store_path": cfg.symptom_store_path,
        "domain_mode": cfg.domain_mode,
        "provisional_policy": cfg.provisional_policy,
        "evidence_window_tokens": cfg.evidence_window_tokens,
        "aspect_memory_auto_promote": cfg.aspect_memory_auto_promote,
        "aspect_memory_review_queue_min_support": cfg.aspect_memory_review_queue_min_support,
        "aspect_memory_review_queue_min_reviews": cfg.aspect_memory_review_queue_min_reviews,
        "aspect_memory_review_queue_min_surface_forms": cfg.aspect_memory_review_queue_min_surface_forms,
        "aspect_memory_path": cfg.aspect_memory_path,
        "domain_holdout_domain": cfg.domain_holdout_domain,
        "max_workers": cfg.max_workers,
    }
