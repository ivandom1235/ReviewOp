from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LLMSettings:
    provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5.4-mini"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    timeout_seconds: int = 45
    max_retries: int = 3


@dataclass
class BuilderConfig:
    input_dir: Path = Path("input/raw")
    output_dir: Path = Path("output")
    split_ratios: dict = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    confidence_threshold: float = 0.35
    prefer_canonical: bool = True
    dry_run: bool = False
    sample_preview_count: int = 3
    mode: str = "all"
    augment: bool = True
    use_api: bool = True
    clean_first: bool = True
    preserve_official_splits: bool = True
    cross_domain: bool = False
    random_seed: int = 42
    min_review_length: int = 8
    max_aspects_per_review: int = 5
    near_dup_threshold: float = 0.9
    preserve_row_count: bool = True
    
    # Stage 2: Data Source Strategy
    target_domain_weight: float = 2.0
    open_corpora_max_share: float = 0.40
    gold_benchmark_eval_only: bool = True
    domain_agnostic_mode: str = "auto"
    
    n_way: int = 3
    k_shot: int = 2

    q_query: int = 2
    strict_quality_filter: bool = True
    target_multi_aspect_min: int = 2
    target_implicit_ratio: float = 0.2
    min_implicit_vote_sources: int = 2
    strong_senticnet_threshold: float = 0.8
    max_canonical_share: float = 0.45
    max_other_domain_share: float = 0.2
    episodic_max_aspect_share: float = 0.35
    hard_negative_k: int = 2
    implicit_query_only: bool = True
    min_evidence_span_chars: int = 5
    require_phrase_evidence: bool = True
    drop_sentence_fallback: bool = True
    conservative_second_aspect_extraction: bool = True
    senticnet_enabled: bool = True
    senticnet_resource_path: Path = PROJECT_ROOT / "resources" / "senticnet_seed.json"
    episode_class_balance_tolerance: float = 0.0
    enforce_labels_field: bool = True
    cross_domain_eval: bool = False
    aspect_definitions_enabled: bool = True
    domain_family_implicit_targets: str = "electronics:0.2,telecom:0.2,ecommerce:0.2,mobility:0.2,healthcare:0.2,services:0.2"
    cross_domain_min_domains: int = 2
    fallback_episode_policy: str = "relax_implicit_query,reduced_shots,reduced_way"
    max_evidence_fallback_rate: float = 0.15
    episode_task_mix: str = "aspect_classification:0.4,implicit_aspect_inference:0.3,aspect_sentiment_classification:0.3"
    hard_negative_strategy: str = "hybrid"
    workflow: str = "single"
    decision_policy: str = "deterministic"
    decision_temperature: float = 0.0
    min_confidence_for_hard_map: float = 0.75
    confidence_uncertainty_threshold: float = 0.75
    confidence_calibration_blend: float = 0.55
    memory_mode: str = "off"
    freeze_memory_during_eval: bool = True
    memory_dir: Path = Path("output/reports/aspect_memory")
    domains_include: List[str] = field(default_factory=list)
    llm: LLMSettings = field(default_factory=LLMSettings)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_config() -> LLMSettings:
    _load_dotenv(PROJECT_ROOT / ".env")
    _load_dotenv(PROJECT_ROOT.parent / ".env")
    return LLMSettings(
        provider=os.getenv("DEFAULT_LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    )


def llm_available(cfg: BuilderConfig) -> bool:
    if not cfg.use_api:
        return False
    if cfg.llm.provider == "anthropic":
        return bool(cfg.llm.anthropic_api_key)
    if cfg.llm.provider == "groq":
        return bool(cfg.llm.groq_api_key)
    return bool(cfg.llm.openai_api_key)
