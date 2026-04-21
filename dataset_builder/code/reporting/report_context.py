from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ReportContext:
    cfg: Any
    generated_at: str
    run_profile: str
    artifact_mode: str
    config: dict[str, Any]
    text_column: str
    frame: Any
    prepared: Any
    sample_frame: Any
    train_built: list[dict[str, Any]]
    val_built: list[dict[str, Any]]
    test_built: list[dict[str, Any]]
    finalized_rows: list[dict[str, Any]]
    candidate_aspects: list[str]
    candidate_aspects_by_language: dict[str, list[str]]
    candidate_aspects_by_domain_train: dict[str, list[str]]
    chunk_preview: list[Any]
    prepared_language_distribution: dict[str, Any]
    schema: Any
    train_domain_conditioning_mode: str
    eval_domain_conditioning_mode: str
    research: dict[str, Any]
    diagnostics: dict[str, Any]
    pipeline_state: dict[str, dict[str, Any]]
    train_review_filter_stats: dict[str, Any]
    train_quarantine_recoverable_rows: list[dict[str, Any]]
    train_quarantine_stats: dict[str, Any]
    train_review_dropped_soft_rows: list[dict[str, Any]]
    train_review_dropped_hard_rows: list[dict[str, Any]]
    train_salvage_stats: dict[str, Any]
    train_leakage_filter_stats_before_salvage: dict[str, Any]
    train_leakage_filter_stats_after_salvage: dict[str, Any]
    train_leakage_filter_stats_after_topup: dict[str, Any]
    train_leakage_filter_stats_after_targeting: dict[str, Any]
    train_sentiment_before_balance: dict[str, Any]
    train_sentiment_after_balance: dict[str, Any]
    train_general_dominance_rate: float
    train_domain_leakage_metrics: dict[str, Any]
    eval_domain_leakage_metrics: dict[str, Any]
    train_negative_ratio: float
    train_positive_ratio: float
    train_neutral_ratio: float
    train_target_blocking_failure: bool
    sampled_run_blocked_or_debug: bool
    quality_analysis_summary: dict[str, Any]
    explicit_metrics: dict[str, Any]
    counts_match: bool
    run_registry: dict[str, Any]
    promoted_registry: dict[str, Any]
    run_registry_version: str
    promoted_registry_version: str
    benchmark_spec: Any
    model_spec: Any
    benchmark_rows_by_split: dict[str, list[dict[str, Any]]]
    benchmark_metadata: dict[str, Any]
    core_benchmark_domains: list[str]
    synthetic_audit: dict[str, Any]
    strict_train_export_rows: list[dict[str, Any]]
    strict_val_export_rows: list[dict[str, Any]]
    strict_test_export_rows: list[dict[str, Any]]
    strict_review_queue_rows: list[dict[str, Any]]
    strict_challenge_rows: list[dict[str, Any]]
    strict_floor_stats: dict[str, Any]
    train_export_floor_rows: list[dict[str, Any]]
    grounding: dict[str, Any]
    domain_prior_boost_count: int
    domain_prior_penalty_count: int


def build_report_context(
    *,
    cfg: Any,
    generated_at: str,
    run_profile: str,
    artifact_mode: str,
    config: dict[str, Any],
    text_column: str,
    frame: Any,
    prepared: Any,
    sample_frame: Any,
    train_built: list[dict[str, Any]],
    val_built: list[dict[str, Any]],
    test_built: list[dict[str, Any]],
    finalized_rows: list[dict[str, Any]],
    candidate_aspects: list[str],
    candidate_aspects_by_language: dict[str, list[str]],
    candidate_aspects_by_domain_train: dict[str, list[str]],
    chunk_preview: list[Any],
    prepared_language_distribution: dict[str, Any],
    schema: Any,
    train_domain_conditioning_mode: str,
    eval_domain_conditioning_mode: str,
    research: dict[str, Any],
    diagnostics: dict[str, Any],
    pipeline_state: dict[str, dict[str, Any]],
    train_review_filter_stats: dict[str, Any],
    train_quarantine_recoverable_rows: list[dict[str, Any]],
    train_quarantine_stats: dict[str, Any],
    train_review_dropped_soft_rows: list[dict[str, Any]],
    train_review_dropped_hard_rows: list[dict[str, Any]],
    train_salvage_stats: dict[str, Any],
    train_leakage_filter_stats_before_salvage: dict[str, Any],
    train_leakage_filter_stats_after_salvage: dict[str, Any],
    train_leakage_filter_stats_after_topup: dict[str, Any],
    train_leakage_filter_stats_after_targeting: dict[str, Any],
    train_sentiment_before_balance: dict[str, Any],
    train_sentiment_after_balance: dict[str, Any],
    train_general_dominance_rate: float,
    train_domain_leakage_metrics: dict[str, Any],
    eval_domain_leakage_metrics: dict[str, Any],
    train_negative_ratio: float,
    train_positive_ratio: float,
    train_neutral_ratio: float,
    train_target_blocking_failure: bool,
    sampled_run_blocked_or_debug: bool,
    quality_analysis_summary: dict[str, Any],
    explicit_metrics: dict[str, Any],
    counts_match: bool,
    run_registry: dict[str, Any],
    promoted_registry: dict[str, Any],
    run_registry_version: str,
    promoted_registry_version: str,
    benchmark_spec: Any,
    model_spec: Any,
    benchmark_rows_by_split: dict[str, list[dict[str, Any]]],
    benchmark_metadata: dict[str, Any],
    core_benchmark_domains: list[str],
    synthetic_audit: dict[str, Any],
    strict_train_export_rows: list[dict[str, Any]],
    strict_val_export_rows: list[dict[str, Any]],
    strict_test_export_rows: list[dict[str, Any]],
    strict_review_queue_rows: list[dict[str, Any]],
    strict_challenge_rows: list[dict[str, Any]],
    strict_floor_stats: dict[str, Any],
    train_export_floor_rows: list[dict[str, Any]],
    grounding: dict[str, Any],
    domain_prior_boost_count: int,
    domain_prior_penalty_count: int,
) -> ReportContext:
    return ReportContext(
        cfg=cfg,
        generated_at=generated_at,
        run_profile=run_profile,
        artifact_mode=artifact_mode,
        config=config,
        text_column=text_column,
        frame=frame,
        prepared=prepared,
        sample_frame=sample_frame,
        train_built=train_built,
        val_built=val_built,
        test_built=test_built,
        finalized_rows=finalized_rows,
        candidate_aspects=candidate_aspects,
        candidate_aspects_by_language=candidate_aspects_by_language,
        candidate_aspects_by_domain_train=candidate_aspects_by_domain_train,
        chunk_preview=chunk_preview,
        prepared_language_distribution=prepared_language_distribution,
        schema=schema,
        train_domain_conditioning_mode=train_domain_conditioning_mode,
        eval_domain_conditioning_mode=eval_domain_conditioning_mode,
        research=research,
        diagnostics=diagnostics,
        pipeline_state=pipeline_state,
        train_review_filter_stats=train_review_filter_stats,
        train_quarantine_recoverable_rows=train_quarantine_recoverable_rows,
        train_quarantine_stats=train_quarantine_stats,
        train_review_dropped_soft_rows=train_review_dropped_soft_rows,
        train_review_dropped_hard_rows=train_review_dropped_hard_rows,
        train_salvage_stats=train_salvage_stats,
        train_leakage_filter_stats_before_salvage=train_leakage_filter_stats_before_salvage,
        train_leakage_filter_stats_after_salvage=train_leakage_filter_stats_after_salvage,
        train_leakage_filter_stats_after_topup=train_leakage_filter_stats_after_topup,
        train_leakage_filter_stats_after_targeting=train_leakage_filter_stats_after_targeting,
        train_sentiment_before_balance=train_sentiment_before_balance,
        train_sentiment_after_balance=train_sentiment_after_balance,
        train_general_dominance_rate=train_general_dominance_rate,
        train_domain_leakage_metrics=train_domain_leakage_metrics,
        eval_domain_leakage_metrics=eval_domain_leakage_metrics,
        train_negative_ratio=train_negative_ratio,
        train_positive_ratio=train_positive_ratio,
        train_neutral_ratio=train_neutral_ratio,
        train_target_blocking_failure=train_target_blocking_failure,
        sampled_run_blocked_or_debug=sampled_run_blocked_or_debug,
        quality_analysis_summary=quality_analysis_summary,
        explicit_metrics=explicit_metrics,
        counts_match=counts_match,
        run_registry=run_registry,
        promoted_registry=promoted_registry,
        run_registry_version=run_registry_version,
        promoted_registry_version=promoted_registry_version,
        benchmark_spec=benchmark_spec,
        model_spec=model_spec,
        benchmark_rows_by_split=benchmark_rows_by_split,
        benchmark_metadata=benchmark_metadata,
        core_benchmark_domains=core_benchmark_domains,
        synthetic_audit=synthetic_audit,
        strict_train_export_rows=strict_train_export_rows,
        strict_val_export_rows=strict_val_export_rows,
        strict_test_export_rows=strict_test_export_rows,
        strict_review_queue_rows=strict_review_queue_rows,
        strict_challenge_rows=strict_challenge_rows,
        strict_floor_stats=strict_floor_stats,
        train_export_floor_rows=train_export_floor_rows,
        grounding=grounding,
        domain_prior_boost_count=domain_prior_boost_count,
        domain_prior_penalty_count=domain_prior_penalty_count,
    )
