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
    run_profile: str = "research"
    confidence_threshold: float = 0.6
    max_aspects: int = 20
    min_text_tokens: int = 4
    implicit_min_tokens: int = 5
    implicit_mode: str = "zeroshot"
    multilingual_mode: str = "shared_vocab"
    use_coref: bool = False
    language_detection_mode: str = "heuristic"
    supported_languages: tuple[str, ...] = ("en", "es", "fr", "de", "pt", "it", "nl", "hi", "zh", "ar", "ru")
    no_drop: bool = False
    enable_llm_fallback: bool = True
    llm_fallback_threshold: float = 0.65
    enable_reasoned_recovery: bool = True
    llm_provider: str | None = None  # options: runpod, openai, anthropic, ollama
    llm_model_name: str = "llama3-8b-instruct"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_max_retries: int = 3
    benchmark_key: str | None = None
    model_family: str = "heuristic_latent"
    augmentation_mode: str = "none"
    prompt_mode: str = "constrained"
    output_version: str = "v4"
    reset_output: bool = True
    high_difficulty: bool = False
    adversarial_refine: bool = False
    multi_aspect_ratio: float = 0.0
    gold_annotations_path: Path | None = None
    emit_review_set: bool = False
    review_set_size: int = 300
    evaluation_protocol: str = "random"
    domain_holdout: str | None = None
    enforce_grounding: bool = True
    use_domain_conditioning: bool = True
    strict_domain_conditioning: bool = False
    domain_conditioning_mode: str = "adaptive_soft"
    train_domain_conditioning_mode: str = "strict_hard"
    eval_domain_conditioning_mode: str = "adaptive_soft"
    domain_prior_boost: float = 0.05
    domain_prior_penalty: float = 0.08
    weak_domain_support_row_threshold: int = 80
    progress: bool = True
    unseen_non_general_coverage_min: float = 0.55
    unseen_implicit_not_ready_rate_max: float = 0.35
    unseen_domain_leakage_row_rate_max: float = 0.02
    train_fallback_general_policy: str = "cap"
    train_fallback_general_cap_ratio: float = 0.15
    train_review_filter_mode: str = "reasoned_strict"
    train_salvage_mode: str = "recover_non_general"
    train_salvage_confidence_threshold: float = 0.56
    train_salvage_accepted_support_types: tuple[str, ...] = ("exact", "near_exact", "gold")
    train_sentiment_balance_mode: str = "cap_neutral_with_dual_floor"
    train_neutral_cap_ratio: float = 0.5
    train_min_negative_ratio: float = 0.12
    train_min_positive_ratio: float = 0.12
    train_max_positive_ratio: float = 0.5
    train_neutral_max_ratio: float = 0.58
    train_topup_recovery_mode: str = "strict_topup"
    train_topup_confidence_threshold: float = 0.58
    train_topup_staged_recovery: bool = True
    train_topup_stage_b_confidence_threshold: float = 0.54
    train_topup_allow_weak_support_in_stage_c: bool = True
    train_topup_stage_c_confidence_threshold: float = 0.52
    train_topup_allowed_support_types: tuple[str, ...] = ("exact", "near_exact", "gold")
    train_target_min_rows: int = 1600
    train_target_max_rows: int = 2000
    strict_implicit_enabled: bool = True
    strict_review_sample_size: int = 200
    strict_explicit_in_implicit_rate_max: float = 0.0
    strict_boundary_fp_max: int = 0
    strict_h2_h3_ratio_min: float = 0.35
    strict_multi_aspect_ratio_min: float = 0.12
    strict_challenge_macro_f1_min: float = 0.5
    max_workers: int = 10

    @property
    def explicit_dir(self) -> Path:
        return self.output_dir / "explicit"

    @property
    def implicit_dir(self) -> Path:
        return self.output_dir / "implicit"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / self.reports_subdir

    @property
    def implicit_strict_dir(self) -> Path:
        return self.output_dir / "implicit_strict"

    def ensure_dirs(self, *, reset_output: bool | None = None) -> None:
        should_reset = self.reset_output if reset_output is None else reset_output
        if self.dry_run or self.preview_only:
            should_reset = False
        if should_reset and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        for path in (self.output_dir, self.explicit_dir, self.implicit_dir, self.implicit_strict_dir, self.reports_dir):
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
