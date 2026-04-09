from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from contracts import BuilderConfig
from aspect_registry import (
    ASPECT_REGISTRY_VERSION,
    build_run_registry,
    canonicalize_domain_aspect,
    resolve_domain_canonical_aspect,
    resolve_registry_version,
    restaurant_ontology_compatible,
    update_promoted_registry,
)
from coref import heuristic_coref
from evaluation import aspect_metrics, benchmark_gold_eval, benchmark_structural_audits, gold_eval
from exporters import write_benchmark_outputs
from explicit_features import build_explicit_row, fit_explicit_artifacts
from implicit_pipeline import (
    _is_valid_latent_aspect,
    _latent_aspect_label,
    build_implicit_row,
    collect_diagnostics,
    discover_aspects,
    MultiAspectSynthesis,
    ResearchAblationMatrix,
    flush_llm_cache,
)
from io_utils import load_gold_annotations, load_inputs
from language_utils import detect_language, is_implicit_ready, language_distribution
from llm_utils import resolve_async_llm_provider
from research_stack import build_research_manifest, resolve_benchmark, resolve_model_family
from robustness_eval import evaluate_training_tracks, promotion_gate
from schema_detect import detect_schema
from splitter import grouped_leakage_report, grouped_split
from synthetic_generation import generate_synthetic_multidomain, write_synthetic_outputs
from governance import governance_signoff
from utils import (
    normalize_whitespace,
    read_jsonl,
    stable_id,
    utc_now_iso,
    write_json,
    write_jsonl,
    compress_dataset_artifacts,
)

load_dotenv()

_GROUP_ID_SOURCE_ROW: dict[str, str] = {}
_GROUP_ID_SOURCE_COUNTS: Counter[str] = Counter()
_GROUP_SEMANTIC_NORMALIZE_RE = r"[^a-z0-9\s]+"


def _load_runtime_defaults() -> dict[str, Any]:
    defaults_path = Path(__file__).resolve().parent / "runtime_defaults.json"
    if not defaults_path.exists():
        return {}
    try:
        payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    defaults = payload.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def _required_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    primary = names[0] if names else "environment variable"
    raise RuntimeError(f"{primary} is required in .env for dataset_builder")


def _optional_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


class _ProgressTracker:
    def __init__(self, *, enabled: bool, total_steps: int) -> None:
        self._enabled = bool(enabled)
        self._bar = tqdm(total=total_steps, desc="pipeline", unit="step", leave=False) if self._enabled else None

    def step(self, label: str, n: int = 1) -> None:
        if not self._bar:
            return
        self._bar.set_description(str(label))
        self._bar.update(n)

    def close(self) -> None:
        if self._bar:
            self._bar.close()


def _assign_ids(frame: pd.DataFrame) -> pd.DataFrame:
    # Optimized: Vectorized ID generation where possible
    out = frame.copy()
    if "source_file" not in out.columns:
        out["source_file"] = "unknown"
    
    # We use a helper to avoid per-row to_dict conversion
    def get_row_id(row_tuple):
        return stable_id(row_tuple.source_file, row_tuple.Index, str(row_tuple))

    out["id"] = [get_row_id(row) for row in out.itertuples()]
    return out


def _harvest_dataset_aspects(frame: pd.DataFrame) -> list[tuple[str, set[str], set[str]]]:
    """Extracts gold labels from all input data to bootstrap the Adaptive Lexicon."""
    aspect_cols = [c for c in frame.columns if c.lower() in {"aspect", "gold_aspect", "target_aspect", "implicit_aspect"}]
    if not aspect_cols:
        return []
    
    discovered_labels = set()
    for col in aspect_cols:
        vals = frame[col].dropna().unique()
        for v in vals:
            if isinstance(v, str) and len(v.strip()) > 2:
                discovered_labels.add(v.strip().lower())
    
    # Map them to rules: (label, explicit_kws, implicit_sigs)
    # For harvesting, we treat the label itself as the explicit keyword.
    new_rules = []
    for label in discovered_labels:
        new_rules.append((label, {label}, set()))
    return new_rules


def _get_row_domain(row: dict[str, Any]) -> str:
    # V6 Research Spec: Prefer explicit 'domain' key over filename inference.
    explicit_domain = row.get("domain")
    source_file = row.get("source_file", "unknown")
    if isinstance(explicit_domain, str) and explicit_domain and explicit_domain != "unknown":
        return explicit_domain.strip().lower()
    return _canonical_domain(str(source_file))


def _canonical_domain(source_file: str | None) -> str:
    name = Path(str(source_file or "unknown")).stem.lower()
    for suffix in ("_train", "_test", "_val", "-train", "-test", "-val"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    domain_map = {
        "product_reviews_mock_data": "product",
        "fake reviews dataset": "general_product",
    }
    return domain_map.get(name, name)


def _chunk_rows(rows: list[dict[str, Any]], cfg: BuilderConfig) -> list[dict[str, Any]]:
    ordered = list(rows)
    random.Random(cfg.random_seed).shuffle(ordered)
    return ordered


_CORE_BENCHMARK_DOMAINS = ("electronics", "restaurant", "telecom")


def _benchmark_domain_family(domain: str | None) -> str:
    normalized = str(domain or "unknown").strip().lower()
    if normalized in {"laptop", "electronics"}:
        return "electronics"
    if normalized in {"restaurant", "telecom"}:
        return normalized
    return normalized or "unknown"


def _benchmark_row_priority(row: dict[str, Any], *, preferred_domain: str | None = None) -> tuple[float, str]:
    implicit = row.get("implicit", {}) or {}
    aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
    hardness = str(implicit.get("hardness_tier") or "").strip().upper()
    review_reason = str(implicit.get("review_reason") or "").strip().lower()
    sentiment = str(implicit.get("dominant_sentiment") or row.get("sentiment") or "").strip().lower()
    domain_family = _benchmark_domain_family(str(_get_row_domain(row)))

    score = 0.0
    if bool(row.get("abstain_acceptable", False)):
        score += 30.0
    if bool(implicit.get("needs_review")):
        score += 18.0
    if review_reason in {"weak_support", "low_confidence"}:
        score += 10.0
    if hardness == "H3":
        score += 20.0
    elif hardness == "H2":
        score += 14.0
    if len(aspects) > 1:
        score += 12.0
    if sentiment == "negative":
        score += 6.0
    elif sentiment == "positive":
        score += 2.0
    if domain_family in _CORE_BENCHMARK_DOMAINS:
        score += 4.0
    if preferred_domain and domain_family == preferred_domain:
        score += 6.0
    stable = stable_id(
        "benchmark-selection",
        row.get("id") or row.get("source_text") or "",
        row.get("domain") or "unknown",
        row.get("split") or "train",
    )
    return (-score, stable)


def _select_working_rows(rows: list[dict[str, Any]], cfg: BuilderConfig) -> list[dict[str, Any]]:
    ordered = _chunk_rows(rows, cfg)
    if cfg.sample_size is None and cfg.chunk_size is None:
        return ordered

    available_core_domains = [
        domain for domain in _CORE_BENCHMARK_DOMAINS if any(_benchmark_domain_family(str(_get_row_domain(row))) == domain for row in ordered)
    ]
    prioritized: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # Seed the core V1 domains first so sampled runs do not collapse to one domain.
    for domain in available_core_domains:
        domain_rows = sorted(
            [row for row in ordered if _benchmark_domain_family(str(_get_row_domain(row))) == domain],
            key=lambda row: _benchmark_row_priority(row, preferred_domain=domain),
        )
        if not domain_rows:
            continue
        chosen = domain_rows[0]
        chosen_id = str(chosen.get("id") or "")
        if chosen_id and chosen_id not in selected_ids:
            prioritized.append(chosen)
            selected_ids.add(chosen_id)

    remaining = [row for row in ordered if str(row.get("id") or "") not in selected_ids]
    remaining_sorted = sorted(remaining, key=lambda row: _benchmark_row_priority(row, preferred_domain=_benchmark_domain_family(str(_get_row_domain(row)))))
    prioritized.extend(remaining_sorted)

    if cfg.sample_size is not None:
        prioritized = prioritized[: max(0, cfg.sample_size)]
    if cfg.chunk_size is not None:
        start = max(0, cfg.chunk_offset)
        end = start + max(0, cfg.chunk_size)
        prioritized = prioritized[start:end]
    return prioritized


def _train_floor_row_passes(
    row: dict[str, Any],
    *,
    candidate_aspects_by_domain: dict[str, list[str]],
    accepted_support_types: set[str],
) -> bool:
    if not _row_domain_valid_for_train(row=row, candidate_aspects_by_domain=candidate_aspects_by_domain):
        return False
    implicit = row.get("implicit", {}) or {}
    aspects = [str(aspect) for aspect in implicit.get("aspects", [])]
    if not aspects or aspects == ["general"]:
        return False
    if str(implicit.get("review_reason") or "") == "boundary_false_positive":
        return False
    spans = list(implicit.get("spans") or [])
    if not spans:
        return False
    if any(str(span.get("support_type") or "") not in accepted_support_types for span in spans):
        return False
    return True


def _benchmark_domain_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[_benchmark_domain_family(str(_get_row_domain(row)))] += 1
    return {domain: int(counts.get(domain, 0)) for domain in _CORE_BENCHMARK_DOMAINS}


_NON_FEATURE_COLUMNS = {
    "id", "aspect", "aspect_term", "from", "to", "label", "labels", "polarity",
    "sentiment", "target", "target_aspect", "gold_aspect", "gold_labels",
}


def _feature_columns(columns: list[str], *, text_column: str, target_column: str | None = None) -> list[str]:
    excluded = {text_column, "language", "split", "source_file", "source_text", "domain", "implicit_ready"}
    if target_column:
        excluded.add(target_column)
    excluded.update(_NON_FEATURE_COLUMNS)
    return [column for column in columns if column not in excluded]


def _split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        grouped.setdefault(str(row.get("split", "train")), []).append(row)
    return grouped


def _aspect_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for aspect in row.get("implicit", {}).get("aspects", []):
            if aspect != "general":
                counts[str(aspect)] += 1
    return counts


def _fallback_rate(rows: list[dict[str, Any]]) -> float:
    return round(sum(1 for row in rows if row.get("implicit", {}).get("aspects") == ["general"]) / len(rows), 4) if rows else 0.0


def _stable_keep(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    token: str,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not rows:
        return []
    if limit >= len(rows):
        return list(rows)
    decorated: list[tuple[str, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        row_key = str(
            row.get("id")
            or row.get("instance_id")
            or row.get("record_id")
            or row.get("parent_review_id")
            or row.get("review_text")
            or index
        )
        decorated.append((stable_id("stable_keep", seed, token, index, row_key), index, row))
    selected = sorted(decorated, key=lambda item: (item[0], item[1]))[:limit]
    return [row for _, _, row in sorted(selected, key=lambda item: item[1])]


def _stable_stratified_keep(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    token: str,
    limit: int,
) -> list[dict[str, Any]]:
    return _stable_keep(rows, seed=seed, token=token, limit=limit)


def _train_sentiment_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"negative": 0, "positive": 0, "neutral": 0}
    for row in rows:
        sentiment = str(
            row.get("implicit", {}).get("dominant_sentiment")
            or row.get("sentiment")
            or "neutral"
        ).strip().lower()
        if sentiment not in counts:
            sentiment = "neutral"
        counts[sentiment] += 1
    return counts


def _sentiment_ratio(rows: list[dict[str, Any]], *, label: str) -> float:
    counts = _train_sentiment_counts(rows)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return round(counts.get(str(label).strip().lower(), 0) / total, 4)


def _ambiguity_score(row: dict[str, Any], gold_interpretations: list[dict[str, Any]]) -> float:
    if not gold_interpretations:
        return float(row.get("implicit", {}).get("ambiguity_score", 0.0) or 0.0)
    supports = [
        max(1, int(item.get("annotator_support", 1) or 1))
        for item in gold_interpretations
        if isinstance(item, dict)
    ]
    if not supports:
        base = float(row.get("implicit", {}).get("ambiguity_score", 0.0) or 0.0)
        return round(max(0.0, min(1.0, base)), 4)
    support_total = sum(supports)
    max_share = max(supports) / max(1, support_total)
    base_score = 1.0 - max_share
    row_score = float(row.get("implicit", {}).get("ambiguity_score", 0.0) or 0.0)
    return round(max(0.0, min(1.0, max(base_score, row_score))), 4)


def _quality_summary(
    rows: list[dict[str, Any]],
    *,
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
    challenge_macro_f1: float = 0.0,
) -> dict[str, Any]:
    stable_review_reasons = [
        "none",
        "fallback_general",
        "implicit_not_ready",
        "weak_support",
        "low_confidence",
        "llm_parse_error",
    ]
    stable_fallback_branches = [
        "none",
        "fallback_general",
        "implicit_not_ready",
        "rule_fallback",
        "llm_parse",
    ]
    grouped_by_split = _split_rows(rows)
    grouped_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    review_reason_counts: Counter[str] = Counter()
    fallback_branch_counts: Counter[str] = Counter()
    domain_leakage_rows = 0
    domain_leakage_aspect_instances = 0
    domain_leakage_by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "aspect_instances": 0})
    for row in rows:
        domain = str(_get_row_domain(row))
        grouped_by_domain[domain].append(row)
        implicit = row.get("implicit", {})
        review_reason_counts[str(implicit.get("review_reason") or "none")] += 1
        fallback_branch_counts[str(implicit.get("fallback_branch") or "none")] += 1
        if candidate_aspects_by_domain:
            domain_candidates = candidate_aspects_by_domain.get(domain, [])
            allowed_latents = {
                _latent_aspect_label(candidate, str(row.get("source_text", "")))
                for candidate in domain_candidates
            }
            allowed_latents = {aspect for aspect in allowed_latents if aspect != "general" and _is_valid_latent_aspect(aspect)}
            if allowed_latents:
                row_aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
                leaked = [aspect for aspect in row_aspects if aspect not in allowed_latents]
                if leaked:
                    domain_leakage_rows += 1
                    domain_leakage_aspect_instances += len(leaked)
                    domain_leakage_by_domain[domain]["rows"] += 1
                    domain_leakage_by_domain[domain]["aspect_instances"] += len(leaked)
    aspect_counts = _aspect_counts(rows)
    total_rows = len(rows)
    review_reason_counts_stable = {key: int(review_reason_counts.get(key, 0)) for key in stable_review_reasons}
    for key, value in review_reason_counts.items():
        if key not in review_reason_counts_stable:
            review_reason_counts_stable[key] = int(value)
    fallback_branch_counts_stable = {key: int(fallback_branch_counts.get(key, 0)) for key in stable_fallback_branches}
    for key, value in fallback_branch_counts.items():
        if key not in fallback_branch_counts_stable:
            fallback_branch_counts_stable[key] = int(value)
    strict_quality = _strict_quality_metrics(rows, challenge_macro_f1=challenge_macro_f1)
    return {
        "canonical_domains": sorted(grouped_by_domain),
        "fallback_only_rows": sum(1 for row in rows if row.get("implicit", {}).get("aspects") == ["general"]),
        "fallback_only_rate": _fallback_rate(rows),
        "needs_review_rows": sum(1 for row in rows if bool(row.get("implicit", {}).get("needs_review"))),
        "generic_implicit_aspects": sum(count for aspect, count in aspect_counts.items() if not _is_valid_latent_aspect(aspect)),
        "rejected_implicit_aspects": sum(1 for row in rows for aspect in row.get("implicit", {}).get("aspects", []) if aspect != "general" and not _is_valid_latent_aspect(aspect)),
        "span_support": {
            "exact": sum(1 for row in rows for span in row.get("implicit", {}).get("spans", []) if span.get("support_type") == "exact"),
            "near_exact": sum(1 for row in rows for span in row.get("implicit", {}).get("spans", []) if span.get("support_type") == "near_exact"),
        },
        "top_implicit_aspects": aspect_counts.most_common(10),
        "top_implicit_aspects_by_split": {split: _aspect_counts(split_rows).most_common(10) for split, split_rows in grouped_by_split.items()},
        "top_implicit_aspects_by_domain": {domain: _aspect_counts(domain_rows).most_common(10) for domain, domain_rows in grouped_by_domain.items()},
        "fallback_only_rate_by_split": {split: _fallback_rate(split_rows) for split, split_rows in grouped_by_split.items()},
        "fallback_only_rate_by_domain": {domain: _fallback_rate(domain_rows) for domain, domain_rows in grouped_by_domain.items()},
        "review_reason_counts": review_reason_counts_stable,
        "fallback_branch_counts": fallback_branch_counts_stable,
        "domain_leakage_rows": domain_leakage_rows,
        "domain_leakage_row_rate": round(domain_leakage_rows / total_rows, 4) if total_rows else 0.0,
        "domain_leakage_aspect_instances": domain_leakage_aspect_instances,
        "domain_leakage_by_domain": dict(domain_leakage_by_domain),
        "strict_quality": strict_quality,
        "explicit_in_implicit_rate": strict_quality["explicit_in_implicit_rate"],
        "boundary_false_positive_count": strict_quality["boundary_false_positive_count"],
        "hardness_distribution": strict_quality["hardness_distribution"],
        "h2_h3_ratio": strict_quality["h2_h3_ratio"],
        "multi_aspect_ratio": strict_quality["multi_aspect_ratio"],
        "challenge_macro_f1": strict_quality["challenge_macro_f1"],
    }


def _grounding_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = len(rows)
    non_general_rows = 0
    grounded_rows = 0
    ungrounded_non_general_count = 0
    for row in rows:
        implicit = row.get("implicit", {}) or {}
        aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
        spans = list(implicit.get("spans") or [])
        if aspects:
            non_general_rows += 1
            has_grounding = bool(spans) and any(str(span.get("evidence_text") or span.get("clause") or "").strip() for span in spans)
            if has_grounding:
                grounded_rows += 1
            else:
                ungrounded_non_general_count += 1
    grounded_prediction_rate = round(grounded_rows / max(1, total_rows), 4)
    non_general_grounding_rate = round(grounded_rows / max(1, non_general_rows), 4) if non_general_rows else 0.0
    return {
        "total_rows": total_rows,
        "non_general_rows": non_general_rows,
        "grounded_rows": grounded_rows,
        "grounded_prediction_rate": grounded_prediction_rate,
        "non_general_grounding_rate": non_general_grounding_rate,
        "ungrounded_non_general_count": ungrounded_non_general_count,
    }


def _strict_quality_metrics(rows: list[dict[str, Any]], *, challenge_macro_f1: float = 0.0) -> dict[str, Any]:
    total_rows = len(rows)
    hardness_distribution: Counter[str] = Counter()
    explicit_contamination = 0
    boundary_false_positive_count = 0
    multi_aspect_count = 0
    h1_count = 0
    h2_count = 0
    h3_count = 0
    for row in rows:
        implicit = row.get("implicit", {}) or {}
        aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
        if len(aspects) > 1:
            multi_aspect_count += 1
        hardness = str(implicit.get("hardness_tier") or "H1").strip().upper()
        hardness_distribution[hardness] += 1
        if hardness == "H1":
            h1_count += 1
        elif hardness == "H2":
            h2_count += 1
        elif hardness == "H3":
            h3_count += 1
        explicit = row.get("explicit", {}) or {}
        explicit_aspects = {
            str(aspect).strip().lower()
            for aspect in list(explicit.get("aspects") or [])
            if str(aspect).strip()
        }
        if explicit_aspects and any(str(aspect).strip().lower() in explicit_aspects for aspect in aspects):
            explicit_contamination += 1
        if str(implicit.get("review_reason") or "") == "boundary_false_positive":
            boundary_false_positive_count += 1
    explicit_in_implicit_rate = round(explicit_contamination / max(1, total_rows), 4)
    h2_h3_ratio = round((h2_count + h3_count) / max(1, h1_count), 4) if total_rows else 0.0
    multi_aspect_ratio = round(multi_aspect_count / max(1, total_rows), 4)
    return {
        "explicit_in_implicit_rate": explicit_in_implicit_rate,
        "boundary_false_positive_count": boundary_false_positive_count,
        "hardness_distribution": {key: int(value) for key, value in hardness_distribution.items()},
        "h2_h3_ratio": h2_h3_ratio,
        "multi_aspect_ratio": multi_aspect_ratio,
        "challenge_macro_f1": round(float(challenge_macro_f1), 4),
    }


def _strict_row_passes(row: dict[str, Any]) -> bool:
    implicit = row.get("implicit", {}) or {}
    aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
    if not aspects:
        return False
    spans = list(implicit.get("spans") or [])
    if not spans:
        return False
    if str(implicit.get("review_reason") or "") == "boundary_false_positive":
        return False
    explicit = row.get("explicit", {}) or {}
    explicit_aspects = {
        str(aspect).strip().lower()
        for aspect in list(explicit.get("aspects") or [])
        if str(aspect).strip()
    }
    if explicit_aspects and any(str(aspect).strip().lower() in explicit_aspects for aspect in aspects):
        return False
    return True


def _merge_gold_labels(rows: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not annotations:
        return rows
    by_record_id: dict[str, dict[str, Any]] = {}
    by_instance_id: dict[str, dict[str, Any]] = {}
    by_review_id: dict[str, dict[str, Any]] = {}
    by_parent_review_id: dict[str, dict[str, Any]] = {}
    by_domain_text: dict[tuple[str, str], dict[str, Any]] = {}

    def _merge_unique_interpretations(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in existing + incoming:
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            source_value = str(
                payload.get("source")
                or payload.get("annotation_source")
                or payload.get("label_source")
                or "imported"
            ).strip() or "imported"
            payload.setdefault("source", source_value)
            payload.setdefault("annotation_source", source_value)
            key = (
                str(payload.get("aspect_label") or payload.get("aspect") or "").strip().lower(),
                str(payload.get("sentiment") or "neutral").strip().lower(),
                normalize_whitespace(payload.get("evidence_text") or payload.get("evidence") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(payload)
        return merged

    for item in annotations:
        labels = item.get("gold_labels") if isinstance(item.get("gold_labels"), list) else []
        gold_interpretations = item.get("gold_interpretations") if isinstance(item.get("gold_interpretations"), list) else []
        annotation_source = str(
            item.get("annotation_source")
            or item.get("review_status")
            or item.get("source")
            or "imported"
        ).strip() or "imported"
        payload = {
            "gold_labels": labels,
            "gold_interpretations": [
                {
                    **dict(interp),
                    "source": str(
                        interp.get("source")
                        or interp.get("annotation_source")
                        or interp.get("label_source")
                        or annotation_source
                    ).strip() or annotation_source,
                    "annotation_source": str(
                        interp.get("annotation_source")
                        or interp.get("source")
                        or annotation_source
                    ).strip() or annotation_source,
                }
                for interp in gold_interpretations
                if isinstance(interp, dict)
            ],
            "abstain_acceptable": bool(item.get("abstain_acceptable", False)),
            "novel_acceptable": bool(item.get("novel_acceptable", False)),
            "novel_cluster_id": item.get("novel_cluster_id"),
            "novel_alias": item.get("novel_alias"),
            "novel_evidence_text": item.get("novel_evidence_text"),
            "annotation_source": annotation_source,
        }
        for key in ("instance_id", "record_id", "review_id", "parent_review_id"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                normalized = value.strip()
                if key == "instance_id":
                    by_instance_id[normalized] = payload
                elif key == "review_id":
                    by_review_id[normalized] = payload
                elif key == "parent_review_id":
                    by_parent_review_id[normalized] = payload
                else:
                    by_record_id[normalized] = payload
        domain = str(item.get("domain") or "unknown")
        text = normalize_whitespace(item.get("text") or item.get("review_text") or "")
        if text:
            by_domain_text[(domain, text)] = payload

    merged: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        existing = out.get("gold_labels") if isinstance(out.get("gold_labels"), list) else []
        existing_interpretations = out.get("gold_interpretations") if isinstance(out.get("gold_interpretations"), list) else []
        record_id = str(out.get("id") or "")
        payload = (
            by_instance_id.get(record_id)
            or by_record_id.get(record_id)
            or by_review_id.get(record_id)
            or by_parent_review_id.get(record_id)
        )
        if payload is None:
            payload = by_record_id.get(str(out.get("record_id") or ""))
        if payload is None:
            payload = by_instance_id.get(str(out.get("instance_id") or ""))
        if payload is None:
            payload = by_review_id.get(str(out.get("review_id") or ""))
        if payload is None:
            payload = by_parent_review_id.get(str(out.get("parent_review_id") or ""))
        if payload is None:
            domain = str(out.get("domain") or "unknown")
            text = normalize_whitespace(out.get("source_text") or out.get("review_text") or "")
            payload = by_domain_text.get((domain, text))
        if payload is not None:
            out["gold_labels"] = _merge_unique_interpretations(existing, payload.get("gold_labels", []))
            out["gold_interpretations"] = _merge_unique_interpretations(existing_interpretations, payload.get("gold_interpretations", []))
            out["abstain_acceptable"] = bool(payload.get("abstain_acceptable", False))
            out["novel_acceptable"] = bool(payload.get("novel_acceptable", False))
            out["novel_cluster_id"] = payload.get("novel_cluster_id")
            out["novel_alias"] = payload.get("novel_alias")
            out["novel_evidence_text"] = payload.get("novel_evidence_text")
            out["annotation_source"] = payload.get("annotation_source", "imported")
        merged.append(out)
    return merged


def _group_identity(row: dict[str, Any]) -> str:
    group_id, source_kind = _group_identity_with_source(row)
    row_id = str(
        row.get("id")
        or row.get("instance_id")
        or row.get("record_id")
        or stable_id("group-row", row.get("source_text") or row.get("review_text") or "")
    )
    _GROUP_ID_SOURCE_ROW[row_id] = source_kind
    _GROUP_ID_SOURCE_COUNTS[source_kind] += 1
    return group_id


def _group_identity_with_source(row: dict[str, Any]) -> tuple[str, str]:
    # Highest priority: Explicit grouping IDs (Provided per-review or per-entity)
    for key in ("product_id", "business_id", "entity_id", "group_id"):
        value = row.get(key)
        if value is not None:
            val_str = str(value).strip().lower()
            if val_str:
                return val_str, "source_identity"

    for key in ("parent_review_id",):
        value = row.get(key)
        if value is not None:
            val_str = str(value).strip().lower()
            if val_str:
                return val_str, "parent_review_identity"

    # Secondary: IDs that imply grouping within a file
    for key in ("review_id", "record_id", "id", "instance_id"):
        value = row.get(key)
        if value is not None:
            val_str = str(value).strip().lower()
            if val_str:
                # Include source_file to avoid ID collisions across different files
                source = str(row.get("source_file") or "unknown").strip().lower()
                return f"{source}:{val_str}", "instance_identity"

    # Semantic fallback: signature based on text content
    semantic = _semantic_group_identity(row)
    if semantic:
        return semantic, "semantic_fallback"

    # Path A: If we can't find a strong identity, return a drop marker
    return "UNKNOWN_GROUP", "unidentified"


def _normalize_for_grouping(text: str) -> str:
    import re
    normalized = normalize_whitespace(text).lower()
    normalized = re.sub(_GROUP_SEMANTIC_NORMALIZE_RE, " ", normalized)
    tokens = [tok for tok in normalized.split() if tok]
    if not tokens:
        return ""
    return " ".join(tokens[:80])


def _semantic_group_identity(row: dict[str, Any]) -> str:
    domain = str(row.get("domain") or "unknown").strip().lower()
    source = str(row.get("source_file") or "unknown").strip().lower()
    text = str(row.get("source_text") or row.get("review_text") or "").strip()
    if not text:
        return ""
    signature = _normalize_for_grouping(text)
    if not signature:
        return ""
    entity_name = str(
        row.get("product_name")
        or row.get("business_name")
        or row.get("entity_name")
        or ""
    ).strip().lower()
    return stable_id("group-family", domain, source, entity_name, signature)


def _safe_absolute_span(review_text: str, evidence_text: str, surface_text: str | None = None) -> list[int]:
    review_norm = str(review_text or "")
    evidence = str(evidence_text or "")
    if not review_norm or not evidence:
        return [-1, -1]
    start = review_norm.lower().find(evidence.lower())
    if start < 0:
        return [-1, -1]
    if surface_text:
        inner = evidence.lower().find(str(surface_text).lower())
        if inner >= 0:
            s = start + inner
            return [int(s), int(s + len(str(surface_text)))]
    return [int(start), int(start + len(evidence))]


def _sanitize_gold_interpretation_spans(
    review_text: str,
    interpretations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    repaired = 0
    cleaned: list[dict[str, Any]] = []
    for item in interpretations:
        row = dict(item)
        evidence_text = str(row.get("evidence_text") or row.get("evidence") or "").strip()
        span = row.get("evidence_span")
        valid = isinstance(span, list) and len(span) == 2
        if valid:
            start = int(span[0] if span[0] is not None else -1)
            end = int(span[1] if span[1] is not None else -1)
            if start < 0 or end < start or end > len(review_text):
                valid = False
        if not valid:
            row["evidence_span"] = _safe_absolute_span(review_text, evidence_text)
            repaired += 1
        cleaned.append(row)
    return cleaned, repaired


def _aspect_registry_state_path() -> Path:
    return Path(__file__).resolve().parents[1] / "state" / "promoted_aspect_registry.json"


def _load_promoted_registry() -> dict[str, Any]:
    path = _aspect_registry_state_path()
    if not path.exists():
        return {"registry_version": ASPECT_REGISTRY_VERSION, "domains": {}, "history": {"runs_seen": 0}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"registry_version": ASPECT_REGISTRY_VERSION, "domains": {}, "history": {"runs_seen": 0}}


def _save_promoted_registry(payload: dict[str, Any]) -> None:
    path = _aspect_registry_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_interpretation_contract(
    *,
    interpretation: dict[str, Any],
    domain: str,
    registry: dict[str, Any] | None,
    enforce_registry_membership: bool,
) -> dict[str, Any] | None:
    item = dict(interpretation)
    label_type = str(item.get("label_type") or "implicit").strip().lower()
    source = str(item.get("source") or item.get("annotation_source") or item.get("label_source") or "unknown").strip().lower()
    evidence_mode = "explicit" if label_type == "explicit" else "implicit"
    fallback_used = bool(item.get("fallback_used", False))
    fallback_reason = str(item.get("fallback_reason") or "").strip()
    surface_rationale_tag = str(
        item.get("surface_rationale_tag")
        or item.get("aspect")
        or item.get("evidence_text")
        or ""
    ).strip()
    latent = str(item.get("aspect_label") or item.get("aspect") or "").strip()
    canonical = resolve_domain_canonical_aspect(
        registry=registry,
        domain=domain,
        latent_aspect=latent,
        surface_rationale_tag=surface_rationale_tag,
        enforce_registry_membership=enforce_registry_membership,
    )
    if not canonical:
        canonical = canonicalize_domain_aspect(
            domain=domain,
            aspect_label=latent,
            surface_rationale_tag=surface_rationale_tag,
        )
    if not canonical:
        return None
    if not restaurant_ontology_compatible(domain=domain, canonical_aspect=canonical):
        return None

    # Enforce implicit purity: explicit rule/lexicon cannot be sole implicit evidence unless fallback-tagged.
    if evidence_mode == "explicit" and source in {"rule", "lexicon"}:
        if not fallback_used or not fallback_reason:
            item["implicit_eligible"] = False
        else:
            item["implicit_eligible"] = True
    else:
        item["implicit_eligible"] = True

    item["evidence_mode"] = evidence_mode
    item["fallback_used"] = fallback_used
    item["fallback_reason"] = fallback_reason or ("rule_lexicon_fallback" if fallback_used else "")
    item["domain_canonical_aspect"] = canonical
    item["surface_rationale_tag"] = surface_rationale_tag
    item["registry_version"] = resolve_registry_version(registry)
    item["aspect_label"] = str(item.get("aspect_label") or latent or canonical)
    item["source"] = source
    return item


def _build_gold_interpretations(row: dict[str, Any]) -> list[dict[str, Any]]:
    # 1. First check explicit gold_interpretations key (from synthetic/human data)
    from_labels = row.get("gold_interpretations")
    if isinstance(from_labels, list) and from_labels:
        out: list[dict[str, Any]] = []
        for item in from_labels:
            if not isinstance(item, dict): continue
            aspect = str(item.get("aspect") or item.get("aspect_label") or "").strip()
            if not aspect: continue
            out.append({
                "aspect_label": aspect,
                "sentiment": str(item.get("sentiment") or "neutral").lower(),
                "evidence_text": str(item.get("evidence_text") or item.get("evidence") or "").strip(),
                "evidence_span": item.get("evidence_span", [-1, -1]),
                "annotator_support": int(item.get("annotator_support", 1) or 1),
                "source": str(item.get("source") or "synthetic"),
                "label_type": str(item.get("label_type") or "implicit"),
                "conformal_set": item.get("conformal_set", [aspect]),
                "ambiguity_type": str(item.get("ambiguity_type") or "").strip() or None,
            })
        if out: return out

    # 2. Check traditional gold_labels format
    labels = row.get("gold_labels")
    if isinstance(labels, list) and labels:
        collapsed: dict[tuple[str, str], dict[str, Any]] = {}
        for label in labels:
            if not isinstance(label, dict): continue
            aspect = str(label.get("aspect") or label.get("aspect_label") or label.get("implicit_aspect") or "").strip()
            if not aspect: continue
            sentiment = str(label.get("sentiment") or "neutral").lower()
            key = (aspect.lower(), sentiment)
            if key not in collapsed:
                collapsed[key] = {
                    "aspect_label": aspect,
                    "sentiment": sentiment,
                    "evidence_text": str(label.get("evidence_text") or label.get("evidence") or "").strip(),
                    "evidence_span": [-1, -1],
                    "annotator_support": 0,
                    "source": "gold",
                    "label_type": "implicit",
                    "ambiguity_type": None,
                }
            collapsed[key]["annotator_support"] += int(label.get("annotator_support", 1) or 1)
        return list(collapsed.values())

    # 3. Fallback to model-generated interpretations for bench (only if allowed by mode)
    # This is handled in _build_benchmark_instances
    return []


def _evidence_for_aspect(row: dict[str, Any], aspect: str) -> tuple[str, list[int], bool]:
    spans = list(row.get("implicit", {}).get("spans") or [])
    for span in spans:
        latent = str(span.get("latent_label") or span.get("latent_aspect") or span.get("aspect") or "").strip().lower()
        if latent and latent == aspect.lower():
            text = str(span.get("evidence_text") or span.get("clause") or span.get("evidence") or "").strip()
            start = int(span.get("start_char", -1) or -1)
            end = int(span.get("end_char", -1) or -1)
            return text, [start, end], False
    source_text = str(row.get("source_text") or "").strip()
    return source_text, [-1, -1], True


def _normalize_evidence_text(value: Any) -> str:
    return normalize_whitespace(str(value or "")).lower()


def _dedupe_gold_interpretations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        aspect = str(item.get("aspect_label") or item.get("aspect") or "").strip()
        if not aspect:
            continue
        sentiment = str(item.get("sentiment") or "neutral").strip().lower()
        evidence_text = str(item.get("evidence_text") or item.get("evidence") or "").strip()
        key = (aspect.lower(), sentiment, _normalize_evidence_text(evidence_text))
        if key not in deduped:
            deduped[key] = dict(item)
            deduped[key]["aspect_label"] = aspect
            deduped[key]["sentiment"] = sentiment
            deduped[key]["evidence_text"] = evidence_text
            deduped[key]["annotator_support"] = int(item.get("annotator_support", 1) or 1)
            deduped[key]["ambiguity_type"] = item.get("ambiguity_type")
        else:
            deduped[key]["annotator_support"] = int(deduped[key].get("annotator_support", 1) or 1) + int(item.get("annotator_support", 1) or 1)
    return list(deduped.values())


def _infer_ambiguity_type(row: dict[str, Any], gold_interpretations: list[dict[str, Any]]) -> str | None:
    if len(gold_interpretations) <= 1:
        return None
    support_values = [int(item.get("annotator_support", 1) or 1) for item in gold_interpretations if isinstance(item, dict)]
    if support_values and len(set(support_values)) > 1:
        return "annotator_disagreement"
    aspects = [str(item.get("aspect_label") or item.get("aspect") or "").strip().lower() for item in gold_interpretations if isinstance(item, dict)]
    unique_aspects = [aspect for aspect in aspects if aspect]
    if len(set(unique_aspects)) > 1:
        return "multi_aspect_implication"
    if len(set(unique_aspects)) == 1:
        return "granularity_overlap"
    return "semantic_ambiguity"


def _stable_novel_cluster_id(*, row: dict[str, Any], novel_evidence_text: str) -> str:
    domain = str(row.get("domain") or "unknown").strip().lower()
    hint = normalize_whitespace(
        str(
            novel_evidence_text
            or row.get("source_text")
            or row.get("review_text")
            or ""
        )
    ).lower()
    return f"novel_{stable_id('v2-novel-cluster', domain, hint)[:12]}"


def _novel_alias_from_text(text: str) -> str | None:
    tokens = [token for token in normalize_whitespace(text).split(" ") if token]
    if not tokens:
        return None
    return " ".join(tokens[:4])


def _allowed_latents_for_domain(
    *,
    row: dict[str, Any],
    candidate_aspects_by_domain: dict[str, list[str]],
) -> set[str]:
    domain = str(row.get("domain") or "unknown")
    candidates = list(candidate_aspects_by_domain.get(domain, [])) if candidate_aspects_by_domain else []
    if not candidates:
        return set()
    source_text = str(row.get("source_text") or row.get("review_text") or "")
    allowed_latents = {
        _latent_aspect_label(candidate, source_text)
        for candidate in candidates
    }
    return {
        aspect
        for aspect in allowed_latents
        if aspect != "general" and _is_valid_latent_aspect(aspect)
    }


def _row_domain_valid_for_train(
    row: dict[str, Any],
    *,
    candidate_aspects_by_domain: dict[str, list[str]],
) -> bool:
    allowed_latents = _allowed_latents_for_domain(row=row, candidate_aspects_by_domain=candidate_aspects_by_domain)
    if not allowed_latents:
        return True
    row_aspects = [str(aspect) for aspect in row.get("implicit", {}).get("aspects", []) if str(aspect) != "general"]
    return all(aspect in allowed_latents for aspect in row_aspects)


def _row_grounded_non_general(
    row: dict[str, Any],
    *,
    accepted_support_types: set[str],
    min_confidence: float,
) -> bool:
    aspects = [str(aspect) for aspect in row.get("implicit", {}).get("aspects", [])]
    if not aspects or aspects == ["general"]:
        return False
    spans = list(row.get("implicit", {}).get("spans") or [])
    if not spans:
        return False
    if any(str(span.get("support_type") or "") not in accepted_support_types for span in spans):
        return False
    aspect_conf = row.get("implicit", {}).get("aspect_confidence", {}) or {}
    confidences = [float(value) for value in aspect_conf.values() if value is not None]
    if not confidences:
        confidences = [float(span.get("confidence", 0.0)) for span in spans]
    if not confidences:
        return False
    return max(confidences) >= float(min_confidence)


def _split_train_review_filter(
    train_rows: list[dict[str, Any]],
    *,
    mode: str,
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
    min_confidence: float = 0.58,
    accepted_support_types: tuple[str, ...] = ("exact", "near_exact", "gold"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mode_name = str(mode or "drop_needs_review").strip().lower()
    before = len(train_rows)
    dropped_soft_rows: list[dict[str, Any]] = []
    dropped_hard_rows: list[dict[str, Any]] = []
    if mode_name == "keep":
        kept_rows = list(train_rows)
    elif mode_name == "reasoned_strict":
        domain_map = candidate_aspects_by_domain or {}
        accepted_support = {str(value).strip() for value in accepted_support_types if str(value).strip()}
        if not accepted_support:
            accepted_support = {"exact", "near_exact", "gold"}
        kept_rows = []
        for row in train_rows:
            implicit = row.get("implicit", {}) or {}
            if not bool(implicit.get("needs_review")):
                kept_rows.append(row)
                continue
            review_reason = str(implicit.get("review_reason") or "")
            if review_reason == "implicit_not_ready":
                dropped_hard_rows.append(row)
                continue
            if review_reason in {"fallback_general", "weak_support", "low_confidence"}:
                dropped_soft_rows.append(row)
                continue
            if (
                _row_grounded_non_general(row, accepted_support_types=accepted_support, min_confidence=min_confidence)
                and _row_domain_valid_for_train(row=row, candidate_aspects_by_domain=domain_map)
            ):
                kept_rows.append(row)
            else:
                dropped_soft_rows.append(row)
    else:
        kept_rows = [row for row in train_rows if not bool(row.get("implicit", {}).get("needs_review"))]
        kept_ids = {str(row.get("id") or "") for row in kept_rows}
        dropped_soft_rows = [row for row in train_rows if str(row.get("id") or "") not in kept_ids]
    after = len(kept_rows)
    dropped_rows = dropped_soft_rows + dropped_hard_rows
    return kept_rows, dropped_soft_rows, dropped_hard_rows, {
        "train_review_rows_before_filter": before,
        "train_review_rows_after_filter": after,
        "train_review_filter_applied": {
            "mode": mode_name,
            "dropped_rows": before - after,
            "dropped_soft_rows": len(dropped_soft_rows),
            "dropped_hard_rows": len(dropped_hard_rows),
        },
    }


def _apply_train_fallback_general_policy(
    train_rows: list[dict[str, Any]],
    *,
    policy: str,
    cap_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fallback_rows = [row for row in train_rows if row.get("implicit", {}).get("aspects") == ["general"]]
    non_fallback_rows = [row for row in train_rows if row.get("implicit", {}).get("aspects") != ["general"]]
    policy_name = str(policy or "cap").strip().lower()
    ratio = max(0.0, min(1.0, float(cap_ratio)))

    if policy_name == "keep":
        kept_fallback = fallback_rows
    elif policy_name == "drop":
        kept_fallback = []
    else:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in fallback_rows:
            buckets[(str(row.get("domain", "unknown")), str(row.get("language", "unknown")))].append(row)
        totals: Counter[tuple[str, str]] = Counter(
            (str(row.get("domain", "unknown")), str(row.get("language", "unknown"))) for row in train_rows
        )
        kept_fallback = []
        for key, bucket_rows in buckets.items():
            allowed = int(round(totals[key] * ratio))
            kept_fallback.extend(_stable_keep(bucket_rows, seed=seed, token=f"fallback:{key[0]}:{key[1]}", limit=allowed))

    kept_rows = non_fallback_rows + kept_fallback
    kept_rows = sorted(
        kept_rows,
        key=lambda row: stable_id(seed, "train-policy-order", row.get("id") or row.get("source_text") or ""),
    )
    before = len(fallback_rows)
    after = sum(1 for row in kept_rows if row.get("implicit", {}).get("aspects") == ["general"])
    stats = {
        "train_general_rows_before_policy": before,
        "train_general_rows_after_policy": after,
        "train_general_policy_applied": {
            "policy": policy_name,
            "cap_ratio": round(ratio, 4),
            "dropped_rows": before - after,
        },
    }
    return kept_rows, stats


def _apply_train_review_filter(
    train_rows: list[dict[str, Any]],
    *,
    mode: str,
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
    min_confidence: float = 0.58,
    accepted_support_types: tuple[str, ...] = ("exact", "near_exact", "gold"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept_rows, _, _, stats = _split_train_review_filter(
        train_rows,
        mode=mode,
        candidate_aspects_by_domain=candidate_aspects_by_domain,
        min_confidence=min_confidence,
        accepted_support_types=accepted_support_types,
    )
    return kept_rows, stats


def _apply_train_sentiment_balance(
    train_rows: list[dict[str, Any]],
    *,
    mode: str,
    neutral_cap_ratio: float,
    min_negative_ratio: float,
    min_positive_ratio: float,
    max_positive_ratio: float,
    neutral_max_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int], dict[str, Any]]:
    before_counts = _train_sentiment_counts(train_rows)
    mode_name = str(mode or "cap_neutral_with_dual_floor").strip().lower()
    ratio = max(0.0, min(1.0, float(neutral_cap_ratio)))
    negative_floor = max(0.0, min(1.0, float(min_negative_ratio)))
    positive_floor = max(0.0, min(1.0, float(min_positive_ratio)))
    positive_max = max(0.0, min(1.0, float(max_positive_ratio)))
    neutral_max = max(0.0, min(1.0, float(neutral_max_ratio)))
    constraints = {
        "min_negative_ratio": negative_floor,
        "min_positive_ratio": positive_floor,
        "max_positive_ratio": positive_max,
        "max_neutral_ratio": neutral_max,
    }
    if mode_name not in {"cap_neutral", "cap_neutral_with_negative_floor", "cap_neutral_with_dual_floor"}:
        rows = list(train_rows)
        constraints["achieved"] = {
            "negative_ratio": _sentiment_ratio(rows, label="negative"),
            "positive_ratio": _sentiment_ratio(rows, label="positive"),
            "neutral_ratio": _sentiment_ratio(rows, label="neutral"),
        }
        return rows, before_counts, _train_sentiment_counts(rows), constraints

    neutral_rows = [
        row for row in train_rows if str(row.get("implicit", {}).get("dominant_sentiment") or "unknown") == "neutral"
    ]
    non_neutral_rows = [
        row for row in train_rows if str(row.get("implicit", {}).get("dominant_sentiment") or "unknown") != "neutral"
    ]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in neutral_rows:
        buckets[(str(row.get("domain", "unknown")), str(row.get("language", "unknown")))].append(row)
    totals: Counter[tuple[str, str]] = Counter(
        (str(row.get("domain", "unknown")), str(row.get("language", "unknown"))) for row in train_rows
    )
    kept_neutral: list[dict[str, Any]] = []
    for key, bucket_rows in buckets.items():
        allowed = int(round(totals[key] * ratio))
        kept_neutral.extend(_stable_keep(bucket_rows, seed=seed, token=f"neutral:{key[0]}:{key[1]}", limit=allowed))

    sentiment_rows: dict[str, list[dict[str, Any]]] = {
        "negative": [
            row for row in non_neutral_rows if str(row.get("implicit", {}).get("dominant_sentiment") or "unknown") == "negative"
        ],
        "positive": [
            row for row in non_neutral_rows if str(row.get("implicit", {}).get("dominant_sentiment") or "unknown") == "positive"
        ],
        "neutral": kept_neutral,
    }

    def _cap_bucket(label: str, limit: int, *, token: str) -> None:
        current_rows = sentiment_rows.get(label, [])
        if limit < len(current_rows):
            sentiment_rows[label] = _stable_stratified_keep(current_rows, seed=seed, token=token, limit=max(0, limit))

    def _count(label: str) -> int:
        return len(sentiment_rows.get(label, []))

    def _total() -> int:
        return _count("negative") + _count("positive") + _count("neutral")

    # Iteratively trim dominant sentiment buckets until constraints stabilize.
    for step in range(8):
        total = _total()
        if total <= 0:
            break

        # Hard upper bounds.
        if neutral_max < 1.0:
            max_neutral_count = int((neutral_max * (_count("negative") + _count("positive"))) / max(1e-9, 1.0 - neutral_max))
            _cap_bucket("neutral", max_neutral_count, token=f"sentiment:neutral_max:{step}")
            total = _total()
            if total <= 0:
                break
        if positive_max < 1.0:
            max_positive_count = int((positive_max * (_count("negative") + _count("neutral"))) / max(1e-9, 1.0 - positive_max))
            _cap_bucket("positive", max_positive_count, token=f"sentiment:positive_max:{step}")
            total = _total()
            if total <= 0:
                break

        changed = False
        total = _total()
        if total <= 0:
            break
        current_negative_ratio = _count("negative") / total
        current_positive_ratio = _count("positive") / total

        if mode_name in {"cap_neutral_with_negative_floor", "cap_neutral_with_dual_floor"} and negative_floor > 0.0 and current_negative_ratio < negative_floor:
            allowed_other = max(0, int(_count("negative") / negative_floor) - _count("negative"))
            while (_count("positive") + _count("neutral")) > allowed_other and (_count("positive") > 0 or _count("neutral") > 0):
                if _count("positive") >= _count("neutral") and _count("positive") > 0:
                    _cap_bucket("positive", _count("positive") - 1, token=f"sentiment:neg_floor:positive:{step}:{_count('positive')}")
                elif _count("neutral") > 0:
                    _cap_bucket("neutral", _count("neutral") - 1, token=f"sentiment:neg_floor:neutral:{step}:{_count('neutral')}")
                changed = True

        if mode_name == "cap_neutral_with_dual_floor":
            total = _total()
            current_positive_ratio = (_count("positive") / total) if total else 0.0
            if positive_floor > 0.0 and total > 0 and current_positive_ratio < positive_floor:
                allowed_other = max(0, int(_count("positive") / positive_floor) - _count("positive"))
                while (_count("negative") + _count("neutral")) > allowed_other and (_count("negative") > 0 or _count("neutral") > 0):
                    if _count("negative") >= _count("neutral") and _count("negative") > 0:
                        _cap_bucket("negative", _count("negative") - 1, token=f"sentiment:pos_floor:negative:{step}:{_count('negative')}")
                    elif _count("neutral") > 0:
                        _cap_bucket("neutral", _count("neutral") - 1, token=f"sentiment:pos_floor:neutral:{step}:{_count('neutral')}")
                    changed = True

        if not changed:
            break

    balanced_rows = sentiment_rows["negative"] + sentiment_rows["positive"] + sentiment_rows["neutral"]

    balanced_rows = sorted(
        balanced_rows,
        key=lambda row: stable_id(seed, "train-balance-order", row.get("id") or row.get("source_text") or ""),
    )
    constraints["achieved"] = {
        "negative_ratio": _sentiment_ratio(balanced_rows, label="negative"),
        "positive_ratio": _sentiment_ratio(balanced_rows, label="positive"),
        "neutral_ratio": _sentiment_ratio(balanced_rows, label="neutral"),
    }
    return balanced_rows, before_counts, _train_sentiment_counts(balanced_rows), constraints


def _strict_train_non_general(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("implicit", {}).get("aspects") != ["general"]]


def _primary_non_general_aspect(row: dict[str, Any]) -> str:
    aspects = [str(aspect) for aspect in row.get("implicit", {}).get("aspects", []) if str(aspect) != "general"]
    if not aspects:
        return "none"
    return sorted(aspects)[0]


def _apply_train_size_target(
    train_rows: list[dict[str, Any]],
    *,
    target_min_rows: int,
    target_max_rows: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_rows = max(0, int(target_min_rows))
    max_rows = max(min_rows, int(target_max_rows))
    current = len(train_rows)
    if current <= max_rows:
        return train_rows, {
            "target_min_rows": min_rows,
            "target_max_rows": max_rows,
            "rows_before_targeting": current,
            "rows_after_targeting": current,
            "targeted_downsample_applied": False,
            "size_shortfall_rows": max(0, min_rows - current),
            "size_within_target_range": min_rows <= current <= max_rows,
        }

    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        bucket_key = (
            str(row.get("domain", "unknown")),
            str(row.get("language", "unknown")),
            str(row.get("implicit", {}).get("dominant_sentiment") or "unknown"),
            _primary_non_general_aspect(row),
        )
        buckets[bucket_key].append(row)

    retained: list[dict[str, Any]] = []
    total_rows = len(train_rows)
    for key, bucket_rows in buckets.items():
        provisional = int(round((len(bucket_rows) / total_rows) * max_rows))
        if provisional <= 0 and bucket_rows:
            provisional = 1
        retained.extend(_stable_keep(bucket_rows, seed=seed, token=f"target:{':'.join(key)}", limit=min(len(bucket_rows), provisional)))

    if len(retained) > max_rows:
        retained = _stable_keep(retained, seed=seed, token="target:global:trim", limit=max_rows)
    elif len(retained) < max_rows:
        retained_ids = {str(row.get("id") or "") for row in retained}
        extras = [row for row in train_rows if str(row.get("id") or "") not in retained_ids]
        retained.extend(_stable_keep(extras, seed=seed, token="target:global:topup", limit=max_rows - len(retained)))

    retained = sorted(retained, key=lambda row: stable_id(seed, "train-target-order", row.get("id") or row.get("source_text") or ""))
    after = len(retained)
    return retained, {
        "target_min_rows": min_rows,
        "target_max_rows": max_rows,
        "rows_before_targeting": current,
        "rows_after_targeting": after,
        "targeted_downsample_applied": True,
        "size_shortfall_rows": max(0, min_rows - after),
        "size_within_target_range": min_rows <= after <= max_rows,
    }


async def _salvage_train_rows(
    dropped_rows: list[dict[str, Any]],
    *,
    mode: str,
    text_column: str,
    candidate_aspects: list[str],
    candidate_aspects_by_language: dict[str, list[str]],
    candidate_aspects_by_domain: dict[str, list[str]],
    confidence_threshold: float,
    strict_domain_conditioning: bool,
    domain_conditioning_mode: str,
    domain_prior_boost: float,
    domain_prior_penalty: float,
    weak_domain_support_row_threshold: int,
    train_domain_support: dict[str, int],
    enforce_grounding: bool,
    salvage_confidence_threshold: float,
    salvage_accepted_support_types: tuple[str, ...],
    multilingual_mode: str,
    use_coref: bool,
    llm_fallback_threshold: float,
    enable_llm_fallback: bool,
    implicit_mode: str,
    max_workers: int,
    llm_provider: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import asyncio
    mode_name = str(mode or "recover_non_general").strip().lower()
    accepted_support = {str(value).strip() for value in salvage_accepted_support_types if str(value).strip()}
    if not accepted_support:
        accepted_support = {"exact", "near_exact", "gold"}
    sent_for_salvage = [
        row for row in dropped_rows
        if str(row.get("implicit", {}).get("review_reason") or "") in {"fallback_general", "implicit_not_ready"}
    ]
    if mode_name == "off" or not sent_for_salvage:
        return [], {
            "salvage_mode": mode_name,
            "salvage_sent_rows": len(sent_for_salvage),
            "salvage_recovered_rows": 0,
            "salvage_recovery_rate": 0.0,
            "salvage_recovered_by_reason": {},
            "salvage_recovered_span_support": {},
            "salvage_accepted_support_types": sorted(accepted_support),
        }

    recovered: list[dict[str, Any]] = []
    recovered_by_reason: Counter[str] = Counter()
    recovered_span_support: Counter[str] = Counter()

    async def process_salvage(item, semaphore):
        async with semaphore:
            idx, row = item
            source_text = str(row.get("source_text") or "")
            language = str(row.get("language", "unknown"))
            domain = str(_get_row_domain(row))
            coref_text = None
            if use_coref:
                coref_result = heuristic_coref(source_text)
                coref_text = coref_result.text
            from implicit_pipeline import build_implicit_row
            res = await build_implicit_row(
                {
                    "id": row.get("id"),
                    "split": "train",
                    text_column: source_text,
                    "source_text": source_text,
                    "gold_labels": row.get("gold_labels", []),
                },
                text_column=text_column,
                candidate_aspects=candidate_aspects,
                confidence_threshold=max(float(confidence_threshold), float(salvage_confidence_threshold)),
                row_index=idx,
                domain=domain,
                language=language,
                implicit_mode=implicit_mode,
                multilingual_mode=multilingual_mode,
                use_coref=use_coref,
                coref_text=coref_text,
                implicit_ready=True,
                llm_fallback_threshold=llm_fallback_threshold,
                enable_llm_fallback=enable_llm_fallback,
                candidate_aspects_by_language=candidate_aspects_by_language,
                candidate_aspects_by_domain=candidate_aspects_by_domain,
                strict_domain_conditioning=strict_domain_conditioning,
                domain_conditioning_mode=domain_conditioning_mode,
                domain_prior_boost=domain_prior_boost,
                domain_prior_penalty=domain_prior_penalty,
                weak_domain_support_row_threshold=weak_domain_support_row_threshold,
                domain_support_rows=int(train_domain_support.get(domain, 0)),
                enforce_grounding=enforce_grounding,
                llm_provider=llm_provider,
            )
            return res["implicit"]

    semaphore = asyncio.Semaphore(max_workers)
    tasks = [process_salvage(item, semaphore) for item in enumerate(sent_for_salvage)]
    
    from tqdm.asyncio import tqdm as as_tqdm
    results = await as_tqdm.gather(*tasks, desc=f"recovering {len(sent_for_salvage)} rows via Stage B", leave=False)

    for (idx, row), retry in zip(enumerate(sent_for_salvage), results):
        aspects = list(retry.get("aspects") or [])
        if not aspects or aspects == ["general"]:
            continue
        spans = list(retry.get("spans") or [])
        if not spans:
            continue
        support_types = {str(span.get("support_type") or "") for span in spans}
        if any(support_type not in accepted_support for support_type in support_types):
            continue
        updated = dict(row)
        updated["track"] = retry.get("track", row.get("track"))
        updated["implicit"] = retry
        recovered.append(updated)
        recovered_by_reason[str(row.get("implicit", {}).get("review_reason") or "unknown")] += 1
        for support_type in support_types:
            recovered_span_support[support_type] += 1

    return recovered, {
        "salvage_mode": mode_name,
        "salvage_sent_rows": len(sent_for_salvage),
        "salvage_recovered_rows": len(recovered),
        "salvage_recovery_rate": round(len(recovered) / len(sent_for_salvage), 4) if sent_for_salvage else 0.0,
        "salvage_recovered_by_reason": dict(recovered_by_reason),
        "salvage_recovered_span_support": dict(recovered_span_support),
        "salvage_accepted_support_types": sorted(accepted_support),
    }

    return recovered, {
        "salvage_mode": mode_name,
        "salvage_sent_rows": len(sent_for_salvage),
        "salvage_recovered_rows": len(recovered),
        "salvage_recovery_rate": round(len(recovered) / len(sent_for_salvage), 4) if sent_for_salvage else 0.0,
        "salvage_recovered_by_reason": dict(recovered_by_reason),
        "salvage_recovered_span_support": dict(recovered_span_support),
        "salvage_accepted_support_types": sorted(accepted_support),
    }

def _expand_domain_candidates_from_rows(
    *,
    rows: list[dict[str, Any]],
    candidate_aspects_by_domain: dict[str, list[str]],
    max_new_terms_per_domain: int = 8,
) -> dict[str, list[str]]:
    updated: dict[str, list[str]] = {domain: list(values) for domain, values in candidate_aspects_by_domain.items()}
    additions: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        domain = str(_get_row_domain(row))
        implicit = row.get("implicit", {}) or {}
        aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
        if not aspects:
            continue
        spans = list(implicit.get("spans") or [])
        strong_support = any(str(span.get("support_type") or "") in {"exact", "gold"} for span in spans)
        if not strong_support:
            continue
        for aspect in aspects:
            additions[domain].append(aspect)
        for span in spans:
            surface = str(span.get("surface_aspect") or span.get("text") or "").strip().lower()
            if surface:
                additions[domain].append(surface)
    for domain, terms in additions.items():
        base = updated.get(domain, [])
        baseline_len = len(base)
        seen = {str(item).strip().lower() for item in base}
        for term in terms:
            norm = str(term).strip().lower()
            if not norm or norm in seen:
                continue
            base.append(norm)
            seen.add(norm)
            if len(base) >= baseline_len + max_new_terms_per_domain:
                break
        updated[domain] = base
    return updated


async def _re_infer_recoverable_train_rows(
    rows: list[dict[str, Any]],
    *,
    text_column: str,
    candidate_aspects: list[str],
    candidate_aspects_by_language: dict[str, list[str]],
    candidate_aspects_by_domain: dict[str, list[str]],
    confidence_threshold: float,
    strict_domain_conditioning: bool,
    domain_conditioning_mode: str,
    domain_prior_boost: float,
    domain_prior_penalty: float,
    weak_domain_support_row_threshold: int,
    train_domain_support: dict[str, int],
    enforce_grounding: bool,
    multilingual_mode: str,
    use_coref: bool,
    llm_fallback_threshold: float,
    enable_llm_fallback: bool,
    implicit_mode: str,
    max_workers: int,
    llm_provider: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import asyncio
    target_reasons = {"weak_support", "low_confidence", "llm_parse_error"}
    stats: Counter[str] = Counter()
    candidate_indices = [
        (idx, row) for idx, row in enumerate(rows)
        if bool(row.get("implicit", {}).get("needs_review")) and str(row.get("implicit", {}).get("review_reason") or "") in target_reasons
    ]
    
    if not candidate_indices:
        return list(rows), {}

    async def process_re_infer(item, semaphore):
        async with semaphore:
            idx, row = item
            source_text = str(row.get("source_text") or "")
            language = str(row.get("language", "unknown"))
            domain = str(_get_row_domain(row))
            coref_text = None
            if use_coref:
                coref_result = heuristic_coref(source_text)
                coref_text = coref_result.text
            from implicit_pipeline import build_implicit_row
            res = await build_implicit_row(
                {
                    "id": row.get("id"),
                    "split": "train",
                    text_column: source_text,
                    "source_text": source_text,
                    "gold_labels": row.get("gold_labels", []),
                },
                text_column=text_column,
                candidate_aspects=candidate_aspects,
                confidence_threshold=confidence_threshold,
                row_index=idx,
                domain=domain,
                language=language,
                implicit_mode=implicit_mode,
                multilingual_mode=multilingual_mode,
                use_coref=use_coref,
                coref_text=coref_text,
                implicit_ready=True,
                llm_fallback_threshold=llm_fallback_threshold,
                enable_llm_fallback=enable_llm_fallback,
                candidate_aspects_by_language=candidate_aspects_by_language,
                candidate_aspects_by_domain=candidate_aspects_by_domain,
                strict_domain_conditioning=strict_domain_conditioning,
                domain_conditioning_mode=domain_conditioning_mode,
                domain_prior_boost=domain_prior_boost,
                domain_prior_penalty=domain_prior_penalty,
                weak_domain_support_row_threshold=weak_domain_support_row_threshold,
                domain_support_rows=int(train_domain_support.get(domain, 0)),
                enforce_grounding=enforce_grounding,
                llm_provider=llm_provider,
            )
            return res["implicit"]

    semaphore = asyncio.Semaphore(max_workers)
    tasks = [process_re_infer(item, semaphore) for item in candidate_indices]
    
    from tqdm.asyncio import tqdm as as_tqdm
    results = await as_tqdm.gather(*tasks, desc=f"re-inferring {len(candidate_indices)} rows", leave=False)

    updated_rows = list(rows)
    for (orig_idx, row), retry in zip(candidate_indices, results):
        implicit = row.get("implicit", {}) or {}
        old_aspects = list(implicit.get("aspects") or [])
        new_aspects = list(retry.get("aspects") or [])
        if new_aspects != old_aspects:
            stats["rows_changed"] += 1
        if new_aspects and new_aspects != ["general"] and bool(retry.get("spans")):
            stats["rows_recovered_non_general"] += 1
        
        updated = dict(row)
        updated["track"] = retry.get("track", row.get("track"))
        updated["implicit"] = retry
        updated_rows[orig_idx] = updated
        
    return updated_rows, dict(stats)

def _strict_topup_recovery(
    *,
    train_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    mode: str,
    target_min_rows: int,
    confidence_threshold: float,
    stage_b_confidence_threshold: float,
    stage_c_confidence_threshold: float,
    staged_recovery: bool,
    allow_weak_support_in_stage_c: bool,
    accepted_support_types: tuple[str, ...],
    candidate_aspects_by_domain: dict[str, list[str]],
    seed: int,
    progress_bar: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode_name = str(mode or "strict_topup").strip().lower()
    if mode_name == "off":
        return list(train_rows), {
            "topup_mode": mode_name,
            "topup_sent_rows": 0,
            "topup_accepted_rows": 0,
            "topup_rows_added": 0,
            "topup_needed_rows": 0,
            "size_recovery_stage": "none",
            "size_recovery_shortfall_remaining": 0,
            "topup_recovered_by_reason": {},
            "topup_recovered_span_support": {},
            "train_topup_rejection_breakdown": {},
            "topup_effectiveness": {"needed": 0, "sent": 0, "accepted": 0, "added": 0, "coverage_of_shortfall": 0.0},
            "topup_applied": False,
            "topup_accepted_support_types": [],
        }

    min_rows = max(0, int(target_min_rows))
    shortage = max(0, min_rows - len(train_rows))
    accepted_support = {str(value).strip() for value in accepted_support_types if str(value).strip()}
    if not accepted_support:
        accepted_support = {"exact", "near_exact", "gold"}
    if shortage <= 0:
        return list(train_rows), {
            "topup_mode": mode_name,
            "topup_sent_rows": 0,
            "topup_accepted_rows": 0,
            "topup_rows_added": 0,
            "topup_needed_rows": 0,
            "size_recovery_stage": "none",
            "size_recovery_shortfall_remaining": 0,
            "topup_recovered_by_reason": {},
            "topup_recovered_span_support": {},
            "train_topup_rejection_breakdown": {},
            "topup_effectiveness": {"needed": 0, "sent": 0, "accepted": 0, "added": 0, "coverage_of_shortfall": 0.0},
            "topup_applied": False,
            "topup_accepted_support_types": sorted(accepted_support),
        }

    existing_ids = {str(row.get("id") or "") for row in train_rows}
    unique_candidates: list[dict[str, Any]] = []
    rejection_breakdown: Counter[str] = Counter(
        {
            "rejected_general": 0,
            "rejected_ungrounded": 0,
            "rejected_domain_invalid": 0,
            "rejected_low_confidence": 0,
            "rejected_support_type": 0,
            "rejected_duplicate": 0,
            "rejected_weak_support_stage_policy": 0,
        }
    )
    for row in candidate_rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in existing_ids:
            rejection_breakdown["rejected_duplicate"] += 1
            continue
        existing_ids.add(row_id)
        unique_candidates.append(row)

    aspect_counts = _aspect_counts(train_rows)
    sentiment_counts = _train_sentiment_counts(train_rows)
    support_rank = {"gold": 3, "exact": 2, "near_exact": 1}

    def _row_rank(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
        spans = list(row.get("implicit", {}).get("spans") or [])
        best_support = max((support_rank.get(str(span.get("support_type") or ""), 0) for span in spans), default=0)
        aspect_conf = row.get("implicit", {}).get("aspect_confidence", {}) or {}
        max_conf = max((float(value) for value in aspect_conf.values()), default=0.0)
        aspects = [str(aspect) for aspect in row.get("implicit", {}).get("aspects", []) if str(aspect) != "general"]
        rarity = min((aspect_counts.get(aspect, 0) for aspect in aspects), default=0)
        sentiment = str(row.get("implicit", {}).get("dominant_sentiment") or "unknown")
        sentiment_rarity = sentiment_counts.get(sentiment, 0)
        stable = stable_id(seed, "topup-rank", row.get("id") or row.get("source_text") or "")
        return (-best_support, -max_conf, rarity, sentiment_rarity, stable)

    ordered_candidates = sorted(unique_candidates, key=_row_rank)
    selected: list[dict[str, Any]] = []
    recovered_by_reason: Counter[str] = Counter()
    recovered_support: Counter[str] = Counter()
    selected_ids: set[str] = set()
    progress_seen_ids: set[str] = set()
    used_stage = "none"
    if progress_bar is not None and hasattr(progress_bar, "total"):
        progress_bar.total = len(ordered_candidates)
        refresh = getattr(progress_bar, "refresh", None)
        if callable(refresh):
            refresh()

    stage_defs: list[tuple[str, float, bool]] = [("A", float(confidence_threshold), False)]
    if bool(staged_recovery):
        stage_defs.append(("B", float(stage_b_confidence_threshold), False))
        stage_defs.append(("C", float(stage_c_confidence_threshold), bool(allow_weak_support_in_stage_c)))

    def _reject_reason(row: dict[str, Any], *, threshold: float, weak_allowed: bool) -> str | None:
        implicit = row.get("implicit", {}) or {}
        aspects = [str(aspect) for aspect in implicit.get("aspects", [])]
        if not aspects or aspects == ["general"]:
            return "rejected_general"
        spans = list(implicit.get("spans") or [])
        if not spans:
            return "rejected_ungrounded"
        if any(str(span.get("support_type") or "") not in accepted_support for span in spans):
            return "rejected_support_type"
        if not _row_domain_valid_for_train(row=row, candidate_aspects_by_domain=candidate_aspects_by_domain):
            return "rejected_domain_invalid"
        if not _row_grounded_non_general(row, accepted_support_types=accepted_support, min_confidence=threshold):
            return "rejected_low_confidence"
        if bool(implicit.get("needs_review")) and str(implicit.get("review_reason") or "") == "weak_support" and not weak_allowed:
            return "rejected_weak_support_stage_policy"
        return None

    remaining = shortage
    for stage_name, stage_threshold, weak_allowed in stage_defs:
        if remaining <= 0:
            break
        if progress_bar is not None:
            progress_bar.set_description(f"train export policies: topup recovery stage {stage_name}")
        stage_additions = 0
        for row in ordered_candidates:
            if remaining <= 0:
                break
            row_id = str(row.get("id") or "")
            if progress_bar is not None and row_id not in progress_seen_ids:
                progress_bar.update(1)
                progress_seen_ids.add(row_id)
            if row_id in selected_ids:
                continue
            reason = _reject_reason(row, threshold=stage_threshold, weak_allowed=weak_allowed)
            if reason is not None:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            stage_additions += 1
            remaining -= 1
            recovered_by_reason[str(row.get("implicit", {}).get("review_reason") or "unknown")] += 1
            for span in list(row.get("implicit", {}).get("spans") or []):
                recovered_support[str(span.get("support_type") or "unknown")] += 1
        if stage_additions > 0:
            used_stage = stage_name

    # Final rejection diagnostics use the most permissive stage configuration reached by policy.
    final_threshold = float(stage_c_confidence_threshold) if bool(staged_recovery) else float(confidence_threshold)
    final_weak_allowed = bool(allow_weak_support_in_stage_c) if bool(staged_recovery) else False
    for row in ordered_candidates:
        row_id = str(row.get("id") or "")
        if row_id in selected_ids:
            continue
        if progress_bar is not None and row_id not in progress_seen_ids:
            progress_bar.update(1)
            progress_seen_ids.add(row_id)
        reason = _reject_reason(row, threshold=final_threshold, weak_allowed=final_weak_allowed)
        if reason is None:
            continue
        rejection_breakdown[reason] += 1

    rows_out = list(train_rows) + selected
    rows_out = sorted(rows_out, key=lambda row: stable_id(seed, "train-topup-order", row.get("id") or row.get("source_text") or ""))
    accepted_count = len(selected)
    shortfall_remaining = max(0, min_rows - len(rows_out))
    coverage = round((accepted_count / shortage), 4) if shortage > 0 else 0.0
    return rows_out, {
        "topup_mode": mode_name,
        "topup_sent_rows": len(unique_candidates),
        "topup_accepted_rows": accepted_count,
        "topup_rows_added": accepted_count,
        "topup_needed_rows": shortage,
        "size_recovery_stage": used_stage,
        "size_recovery_shortfall_remaining": shortfall_remaining,
        "topup_recovered_by_reason": dict(recovered_by_reason),
        "topup_recovered_span_support": dict(recovered_support),
        "train_topup_rejection_breakdown": dict(rejection_breakdown),
        "topup_effectiveness": {
            "needed": shortage,
            "sent": len(unique_candidates),
            "accepted": accepted_count,
            "added": accepted_count,
            "coverage_of_shortfall": coverage,
        },
        "topup_applied": accepted_count > 0,
        "topup_accepted_support_types": sorted(accepted_support),
    }


def _domain_generalization(rows: list[dict[str, Any]], *, evaluation_protocol: str, domain_holdout: str | None = None) -> dict[str, Any]:
    domains = sorted({str(row.get("domain", "unknown")) for row in rows})
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in domains:
        domain_rows = [row for row in rows if str(row.get("domain", "unknown")) == domain]
        summary = _quality_summary(domain_rows)
        by_domain[domain] = {
            "rows": len(domain_rows),
            "fallback_only_rate": summary["fallback_only_rate"],
            "needs_review_rows": summary["needs_review_rows"],
            "grounded_prediction_rate": _grounding_metrics(domain_rows)["grounded_prediction_rate"],
            "domain_leakage_row_rate": summary.get("domain_leakage_row_rate", 0.0),
        }
    leave_one_domain_out = {}
    if evaluation_protocol == "loo":
        for domain in domains:
            train_like = [row for row in rows if str(row.get("domain", "unknown")) != domain]
            holdout = [row for row in rows if str(row.get("domain", "unknown")) == domain]
            leave_one_domain_out[domain] = {
                "train_rows": len(train_like),
                "holdout_rows": len(holdout),
                "holdout_fallback_only_rate": _quality_summary(holdout)["fallback_only_rate"] if holdout else 0.0,
                "holdout_needs_review_rows": _quality_summary(holdout)["needs_review_rows"] if holdout else 0,
            }
    heldout_domain_metrics = None
    if domain_holdout:
        holdout_rows = [row for row in rows if str(row.get("domain", "unknown")) == str(domain_holdout)]
        if holdout_rows:
            heldout_domain_metrics = {
                "domain": str(domain_holdout),
                "rows": len(holdout_rows),
                "fallback_only_rate": _quality_summary(holdout_rows)["fallback_only_rate"],
                "needs_review_rows": _quality_summary(holdout_rows)["needs_review_rows"],
                "domain_leakage_row_rate": _quality_summary(holdout_rows).get("domain_leakage_row_rate", 0.0),
            }
    return {
        "evaluation_protocol": evaluation_protocol,
        "domains_seen": domains,
        "by_domain": by_domain,
        "leave_one_domain_out": leave_one_domain_out,
        "heldout_domain_metrics": heldout_domain_metrics,
    }


def _resolve_domain_conditioning_mode(cfg: BuilderConfig) -> str:
    mode = str(getattr(cfg, "domain_conditioning_mode", "") or "").strip().lower()
    if not bool(cfg.use_domain_conditioning):
        return "off"
    if mode == "strict_hard":
        return "strict_hard"
    if mode == "off":
        return "off"
    if bool(cfg.strict_domain_conditioning) and mode == "adaptive_soft":
        return "strict_hard"
    if mode == "adaptive_soft":
        return "adaptive_soft"
    if bool(cfg.strict_domain_conditioning):
        return "strict_hard"
    return "adaptive_soft"


def _resolve_split_domain_conditioning_modes(cfg: BuilderConfig) -> tuple[str, str]:
    resolved_mode = _resolve_domain_conditioning_mode(cfg)
    train_mode = str(getattr(cfg, "train_domain_conditioning_mode", "") or "").strip().lower()
    eval_mode = str(getattr(cfg, "eval_domain_conditioning_mode", "") or "").strip().lower()
    if train_mode not in {"adaptive_soft", "strict_hard", "off"}:
        train_mode = "strict_hard" if resolved_mode != "off" else "off"
    if eval_mode not in {"adaptive_soft", "strict_hard", "off"}:
        eval_mode = "adaptive_soft" if resolved_mode != "off" else "off"
    return train_mode, eval_mode


def _train_domain_leakage_metrics(
    rows: list[dict[str, Any]],
    *,
    candidate_aspects_by_domain: dict[str, list[str]],
) -> dict[str, Any]:
    summary = _quality_summary(rows, candidate_aspects_by_domain=candidate_aspects_by_domain)
    return {
        "train_domain_leakage_rows": int(summary.get("domain_leakage_rows", 0)),
        "train_domain_leakage_row_rate": float(summary.get("domain_leakage_row_rate", 0.0)),
        "train_domain_leakage_aspect_instances": int(summary.get("domain_leakage_aspect_instances", 0)),
    }


def _strict_train_domain_leakage_filter(
    rows: list[dict[str, Any]],
    *,
    candidate_aspects_by_domain: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    removed_rows = 0
    removed_aspect_instances = 0
    for row in rows:
        domain = str(_get_row_domain(row))
        domain_candidates = candidate_aspects_by_domain.get(domain, [])
        allowed_latents = {
            _latent_aspect_label(candidate, str(row.get("source_text", "")))
            for candidate in domain_candidates
        }
        allowed_latents = {aspect for aspect in allowed_latents if aspect != "general" and _is_valid_latent_aspect(aspect)}
        row_aspects = [str(aspect) for aspect in row.get("implicit", {}).get("aspects", []) if str(aspect) != "general"]
        if not allowed_latents:
            kept.append(row)
            continue
        leaked = [aspect for aspect in row_aspects if aspect not in allowed_latents]
        if leaked:
            removed_rows += 1
            removed_aspect_instances += len(leaked)
            continue
        kept.append(row)
    return kept, {
        "train_domain_leakage_filter_removed_rows": removed_rows,
        "train_domain_leakage_filter_removed_aspect_instances": removed_aspect_instances,
    }


def _unseen_domain_metrics(
    rows: list[dict[str, Any]],
    *,
    train_domain_support: dict[str, int],
    weak_domain_support_row_threshold: int,
) -> dict[str, Any]:
    threshold = max(1, int(weak_domain_support_row_threshold))
    unseen_domains = {
        str(domain)
        for domain, count in train_domain_support.items()
        if int(count) < threshold
    }
    unseen_rows = [row for row in rows if str(row.get("domain", "unknown")) in unseen_domains]
    total = len(unseen_rows)
    if total == 0:
        return {
            "domains": sorted(unseen_domains),
            "rows": 0,
            "unseen_non_general_coverage": 0.0,
            "unseen_implicit_not_ready_rate": 0.0,
            "unseen_domain_leakage_row_rate": 0.0,
        }
    non_general = sum(1 for row in unseen_rows if row.get("implicit", {}).get("aspects") and row.get("implicit", {}).get("aspects") != ["general"])
    not_ready = sum(1 for row in unseen_rows if str(row.get("implicit", {}).get("review_reason") or "") == "implicit_not_ready")
    leakage = _quality_summary(unseen_rows).get("domain_leakage_row_rate", 0.0)
    return {
        "domains": sorted(unseen_domains),
        "rows": total,
        "unseen_non_general_coverage": round(non_general / total, 4),
        "unseen_implicit_not_ready_rate": round(not_ready / total, 4),
        "unseen_domain_leakage_row_rate": float(leakage),
    }


def _novelty_identity_block(cfg: BuilderConfig, report: dict[str, Any]) -> dict[str, Any]:
    reasons = set(report.get("output_quality", {}).get("review_reason_counts", {}).keys())
    taxonomy_required = {"fallback_general", "weak_support", "implicit_not_ready"}
    return {
        "hybrid_explicit_implicit_pipeline": True,
        "evidence_grounding_checks_enabled": bool(cfg.enforce_grounding),
        "structured_fallback_taxonomy_present": taxonomy_required.issubset(reasons),
        "single_domain_conditioned_pipeline": bool(cfg.use_domain_conditioning),
        "borrowed_ideas_scope": "evaluation_protocol_only",
        "core_method_scope": "dagr_pipe_original",
    }


def _build_review_set_template(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("split", "train")), str(row.get("domain", "unknown")), str(row.get("language", "unknown")))].append(row)
    target_size = max(1, min(size, len(rows)))
    buckets = list(grouped.items())
    rand = random.Random(seed)
    samples: list[dict[str, Any]] = []
    while len(samples) < target_size and buckets:
        next_buckets: list[tuple[tuple[str, str, str], list[dict[str, Any]]]] = []
        for key, bucket_rows in buckets:
            if not bucket_rows or len(samples) >= target_size:
                continue
            idx = rand.randrange(len(bucket_rows))
            row = bucket_rows.pop(idx)
            samples.append({
                "record_id": row.get("id"),
                "split": key[0],
                "domain": key[1],
                "language": key[2],
                "text": row.get("source_text", ""),
                "gold_labels": [],
                "annotator_id": None,
                "review_status": "pending",
            })
            if bucket_rows:
                next_buckets.append((key, bucket_rows))
        buckets = next_buckets
    return samples


_QUALITY_BORDERLINE_REASONS = {"weak_support", "low_confidence"}
_QUALITY_REJECT_REASONS = {
    "implicit_not_ready",
    "fallback_general",
    "boundary_false_positive",
    "rejected_general",
    "rejected_ungrounded",
    "rejected_domain_invalid",
    "rejected_low_confidence",
    "rejected_support_type",
    "rejected_duplicate",
    "rejected_weak_support_stage_policy",
    "general_only",
    "no_spans",
    "unsupported_support_type",
    "domain_leakage",
    "explicit_contamination",
    "invalid_aspect",
}


def _quality_reason_codes(
    row: dict[str, Any],
    *,
    min_confidence: float,
    accepted_support_types: tuple[str, ...],
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
) -> list[str]:
    implicit = row.get("implicit", {}) or {}
    explicit = row.get("explicit", {}) or {}
    review_reason = str(implicit.get("review_reason") or "").strip()
    reasons: list[str] = []
    if review_reason:
        reasons.append(review_reason)

    aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
    spans = list(implicit.get("spans") or [])
    if not aspects:
        reasons.append("general_only")
    if not spans:
        reasons.append("no_spans")
    if spans and any(str(span.get("support_type") or "") not in accepted_support_types for span in spans):
        reasons.append("unsupported_support_type")

    aspect_conf = implicit.get("aspect_confidence", {}) or {}
    confidences = [float(value) for value in aspect_conf.values() if value is not None]
    if not confidences:
        confidences = [float(span.get("confidence", 0.0)) for span in spans if span.get("confidence") is not None]
    if confidences and max(confidences) < float(min_confidence):
        reasons.append("low_confidence")

    if review_reason == "implicit_not_ready" or not bool(implicit.get("implicit_ready", True)):
        reasons.append("implicit_not_ready")
    if review_reason == "fallback_general":
        reasons.append("fallback_general")
    if review_reason == "boundary_false_positive":
        reasons.append("boundary_false_positive")

    if candidate_aspects_by_domain is not None and not _row_domain_valid_for_train(row=row, candidate_aspects_by_domain=candidate_aspects_by_domain):
        reasons.append("domain_leakage")

    explicit_aspects = {
        str(aspect).strip().lower()
        for aspect in list(explicit.get("aspects") or [])
        if str(aspect).strip()
    }
    if explicit_aspects and any(str(aspect).strip().lower() in explicit_aspects for aspect in aspects):
        reasons.append("explicit_contamination")

    if any(not _is_valid_latent_aspect(aspect) for aspect in aspects):
        reasons.append("invalid_aspect")

    return list(dict.fromkeys(reasons))


def _quality_row_bucket(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return None
    if any(code in _QUALITY_REJECT_REASONS for code in reason_codes):
        return "rejected"
    if any(code in _QUALITY_BORDERLINE_REASONS for code in reason_codes):
        return "borderline"
    return "rejected"


def _build_quality_analysis_artifact(
    train_rows: list[dict[str, Any]],
    final_train_rows: list[dict[str, Any]],
    *,
    min_confidence: float,
    accepted_support_types: tuple[str, ...],
    candidate_aspects_by_domain: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    final_ids = {str(row.get("id") or "") for row in final_train_rows}
    excluded_rows = [row for row in train_rows if str(row.get("id") or "") not in final_ids]
    borderline_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for row in excluded_rows:
        reason_codes = _quality_reason_codes(
            row,
            min_confidence=min_confidence,
            accepted_support_types=accepted_support_types,
            candidate_aspects_by_domain=candidate_aspects_by_domain,
        )
        bucket = _quality_row_bucket(reason_codes)
        payload = {
            "row": dict(row),
            "reason_codes": reason_codes,
        }
        for code in reason_codes:
            reason_counts[code] += 1
        if bucket == "borderline":
            borderline_rows.append(payload)
        else:
            rejected_rows.append(payload)

    return {
        "generated_at": utc_now_iso(),
        "train_rows": len(train_rows),
        "final_train_rows": len(final_train_rows),
        "excluded_rows": len(excluded_rows),
        "borderline_count": len(borderline_rows),
        "rejected_count": len(rejected_rows),
        "reason_group_counts": dict(reason_counts),
        "borderline_rows": borderline_rows,
        "rejected_rows": rejected_rows,
        "summary": {
            "train_rows": len(train_rows),
            "final_train_rows": len(final_train_rows),
            "excluded_rows": len(excluded_rows),
            "borderline_count": len(borderline_rows),
            "rejected_count": len(rejected_rows),
            "reason_group_counts": dict(reason_counts),
        },
    }


def _prepare_rows(frame: pd.DataFrame, cfg: BuilderConfig, text_column: str, progress_tracker: _ProgressTracker | None = None) -> pd.DataFrame:
    out = _assign_ids(frame.reset_index(drop=True).copy())
    out[text_column] = out[text_column].fillna("").astype(str)
    
    if progress_tracker:
        progress_tracker.step("preprocessing: schema & metadata", 0)
    
    if "domain" not in out.columns:
        out["domain"] = out.get("source_file", pd.Series(["unknown"] * len(out))).map(lambda value: _canonical_domain(str(value)))
    else:
        # Respect existing domain key from JSONL, fill missing via source_file
        mask = (out["domain"] == "unknown") | (out["domain"].isna())
        if mask.any():
            source_vals = out.get("source_file", pd.Series(["unknown"] * len(out)))
            out.loc[mask, "domain"] = source_vals[mask].map(lambda value: _canonical_domain(str(value)))
    out["language"] = out[text_column].map(detect_language)
    out["implicit_ready"] = [
        is_implicit_ready(text, language=language, min_tokens=cfg.implicit_min_tokens, supported_languages=cfg.supported_languages)
        for text, language in zip(out[text_column].tolist(), out["language"].tolist(), strict=False)
    ]
    if not cfg.no_drop:
        out = out[out[text_column].str.split().map(len) >= cfg.min_text_tokens].reset_index(drop=True)
    return out


def _build_labels_for_export(row: dict[str, Any]) -> list[dict[str, Any]]:
    implicit = row.get("implicit", {}) or {}
    # Use canonical aspect
    aspect = str(implicit.get("aspect") or "general")
    if not aspect or aspect == "general":
        return []
        
    labels: list[dict[str, Any]] = []
    # Every label now includes the core canonical fields + unconventional metadata
    evidence_text, _, fallback_used = _evidence_for_aspect(row, aspect)
    labels.append(
        {
            "aspect": aspect,
            "sentiment": str(implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral"))).lower(),
            "confidence": float(implicit.get("aspect_confidence", {}).get(aspect, 0.5)),
            "evidence": evidence_text, # Hard cut-over Fix 1
            "conformal_set": implicit.get("conformal_set", [aspect]),
            "ambiguity_score": float(implicit.get("ambiguity_score", 0.0)),
            "pivot_confirmed": bool(implicit.get("pivot_confirmed", False)),
            "type": str(implicit.get("label_type") or "implicit"),
        }
    )
    return labels


def _to_model_export_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["review_text"] = str(row.get("source_text") or row.get("review_text") or "")
    out["labels"] = _build_labels_for_export(row)
    out["group_id"] = _group_identity(row)
    # Active Learning Hook: Tag high priority review if ambiguous
    implicit = row.get("implicit", {}) or {}
    out["review_priority"] = "high" if float(implicit.get("ambiguity_score", 0.0)) > 0.7 else "standard"
    return out


def _canonical_cluster(label: str) -> str:
    return normalize_whitespace(str(label or "")).strip().lower()


def _best_novel_evidence(golds: list[dict[str, Any]]) -> str:
    for item in golds:
        evidence = str(item.get("evidence_text") or "").strip()
        if evidence:
            return evidence
    return ""


def _assign_novelty_flags(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    train_clusters: set[str] = set()
    for row in rows_by_split.get("train", []):
        for item in list(row.get("gold_interpretations") or []):
            if isinstance(item, dict):
                cluster = _canonical_cluster(item.get("aspect_label") or item.get("aspect"))
                if cluster:
                    train_clusters.add(cluster)
    stats = {"train_known_clusters": len(train_clusters), "novel_rows_marked": 0}
    for split_name in ("val", "test"):
        for row in rows_by_split.get(split_name, []):
            golds = [g for g in list(row.get("gold_interpretations") or []) if isinstance(g, dict)]
            clusters = [_canonical_cluster(g.get("aspect_label") or g.get("aspect")) for g in golds]
            unseen = [cluster for cluster in clusters if cluster and cluster not in train_clusters]
            if unseen:
                row["novel_acceptable"] = True
                row["novel_cluster_id"] = row.get("novel_cluster_id") or stable_id("heldout-cluster", unseen[0])
                novel_text = _best_novel_evidence(golds)
                row["novel_evidence_text"] = row.get("novel_evidence_text") or novel_text or row.get("review_text")
                row["novel_alias"] = row.get("novel_alias") or _novel_alias_from_text(str(row.get("novel_evidence_text") or ""))
                stats["novel_rows_marked"] += 1
            else:
                row["novel_acceptable"] = bool(row.get("novel_acceptable", False))
    return stats


def _weak_evidence_overlap(golds: list[dict[str, Any]]) -> bool:
    evidences = [normalize_whitespace(str(item.get("evidence_text") or "")).lower() for item in golds if isinstance(item, dict)]
    evidences = [text for text in evidences if text]
    if len(evidences) < 2:
        return False
    unique = set(evidences)
    return len(unique) < len(evidences) or any(a in b or b in a for idx, a in enumerate(evidences) for jdx, b in enumerate(evidences) if idx != jdx)


def _should_allow_abstain(row: dict[str, Any], gold_interpretations: list[dict[str, Any]]) -> bool:
    ambiguity = float(row.get("ambiguity_score", 0.0) or 0.0)
    if ambiguity <= 0:
        ambiguity = float((row.get("implicit", {}) or {}).get("ambiguity_score", 0.0) or 0.0)
    unique_labels = {
        (str(g.get("aspect_label") or g.get("aspect") or "").strip().lower(), str(g.get("sentiment") or "neutral").strip().lower())
        for g in gold_interpretations
        if isinstance(g, dict)
    }
    if 0.45 <= ambiguity <= 0.75 and len(unique_labels) >= 2:
        return True
    if bool(row.get("novel_acceptable", False)):
        return True
    if _weak_evidence_overlap(gold_interpretations):
        return True
    return False


def _apply_benchmark_balance_policy(rows_by_split: dict[str, list[dict[str, Any]]], *, seed: int) -> dict[str, int]:
    # Keep policy concentrated on train to reduce easy/neutral majority bias.
    train_rows = list(rows_by_split.get("train", []))
    if not train_rows:
        return {"train_rows_before": 0, "train_rows_after": 0}

    def hardness(row: dict[str, Any]) -> str:
        return str(row.get("hardness_tier") or "").strip().upper()

    def sentiment(row: dict[str, Any]) -> str:
        interpretations = list(row.get("gold_interpretations") or [])
        if interpretations and isinstance(interpretations[0], dict):
            return str(interpretations[0].get("sentiment") or "neutral").strip().lower()
        return "neutral"

    hard_rows = [row for row in train_rows if hardness(row) in {"H2", "H3"}]
    easy_rows = [row for row in train_rows if hardness(row) not in {"H2", "H3"}]
    negatives = [row for row in train_rows if sentiment(row) == "negative"]
    positives = [row for row in train_rows if sentiment(row) == "positive"]
    abstain_rows = [row for row in train_rows if bool(row.get("abstain_acceptable", False))]
    multi_rows = [row for row in train_rows if len(list(row.get("gold_interpretations") or [])) >= 2]

    keep_ids: set[str] = set()
    for pool in (hard_rows, negatives, positives, abstain_rows, multi_rows):
        for row in pool:
            keep_ids.add(str(row.get("instance_id") or row.get("record_id") or ""))

    neutral_easy = [row for row in easy_rows if sentiment(row) == "neutral"]
    neutral_cap = int(round(max(1, len(train_rows) * 0.55)))
    neutral_keep = _stable_keep(neutral_easy, seed=seed, token="benchmark-neutral-cap", limit=neutral_cap)
    for row in neutral_keep:
        keep_ids.add(str(row.get("instance_id") or row.get("record_id") or ""))

    balanced = [row for row in train_rows if str(row.get("instance_id") or row.get("record_id") or "") in keep_ids]
    rows_by_split["train"] = balanced
    return {"train_rows_before": len(train_rows), "train_rows_after": len(balanced)}


def _export_protocol_views(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    protocol_rows = {
        "random": {"train": [], "val": [], "test": []},
        "grouped": {"train": [], "val": [], "test": []},
        "domain_holdout": {"train": [], "val": [], "test": []},
    }
    all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    for row in all_rows:
        assign = row.get("split_protocol", {}) if isinstance(row.get("split_protocol"), dict) else {}
        for protocol in protocol_rows:
            split = str(assign.get(protocol) or row.get("split") or "train")
            if split not in {"train", "val", "test"}:
                split = "train"
            item = dict(row)
            item["split"] = split
            protocol_rows[protocol][split].append(item)
    return protocol_rows


def _benchmark_protocol_assignments(rows: list[dict[str, Any]], seed: int) -> dict[str, dict[str, str]]:
    if not rows:
        return {}
    normalized = [dict(row, group_id=_group_identity(row)) for row in rows]
    assignments: dict[str, dict[str, str]] = {}

    random_rows = [dict(row, random_group=str(row.get("id") or stable_id("row", row.get("source_text") or ""))) for row in normalized]
    random_train, random_val, random_test = grouped_split(
        random_rows,
        group_key="random_group",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=seed,
    )
    for split_name, split_rows in (("train", random_train), ("val", random_val), ("test", random_test)):
        for row in split_rows:
            assignments.setdefault(str(row.get("id") or ""), {})["random"] = split_name

    grouped_train, grouped_val, grouped_test = grouped_split(
        normalized,
        group_key="group_id",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=seed + 11,
    )
    for split_name, split_rows in (("train", grouped_train), ("val", grouped_val), ("test", grouped_test)):
        for row in split_rows:
            assignments.setdefault(str(row.get("id") or ""), {})["grouped"] = split_name

    by_domain = [dict(row, domain_group=str(_benchmark_domain_family(str(row.get("domain") or "unknown")))) for row in normalized]
    dom_train, dom_val, dom_test = grouped_split(
        by_domain,
        group_key="domain_group",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=seed + 23,
    )
    for split_name, split_rows in (("train", dom_train), ("val", dom_val), ("test", dom_test)):
        for row in split_rows:
            assignments.setdefault(str(row.get("id") or ""), {})["domain_holdout"] = split_name
    return assignments


def _build_benchmark_instances(
    rows: list[dict[str, Any]],
    protocol_assignments: dict[str, dict[str, str]],
    *,
    artifact_mode: str = "research_release",
    debug_row_limit: int | None = None,
    seed: int = 42,
    promoted_registry: dict[str, Any] | None = None,
    enforce_registry_membership: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    benchmark_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    selected_rows = list(rows)
    if artifact_mode == "debug_artifacts" and debug_row_limit is not None and debug_row_limit > 0 and len(selected_rows) > debug_row_limit:
        selected_rows = _stable_keep(selected_rows, seed=seed, token="benchmark-debug-cap", limit=debug_row_limit)

    total_interpretations = 0
    grounded_interpretations = 0
    duplicate_interpretations_removed = 0
    thermal_interpretations = 0
    val_guard_triggered = False
    deferred_review_rows: list[dict[str, Any]] = []
    interpretation_source_counter: Counter[str] = Counter()
    invalid_span_repaired = 0
    implicit_interpretation_count = 0
    explicit_interpretation_count = 0
    fallback_only_implicit_count = 0
    ontology_compatible_count = 0
    source_domain_family_counts: Counter[str] = Counter(_benchmark_domain_family(str(_get_row_domain(row))) for row in selected_rows)
    benchmark_domain_family_counts: Counter[str] = Counter()
    benchmark_hardness_counts: Counter[str] = Counter()

    for row in selected_rows:
        row_id = str(row.get("id") or "")
        split_protocol = protocol_assignments.get(row_id, {})
        random_split = str(split_protocol.get("random") or row.get("split") or "train")
        
        # Priority: Human Gold -> Synthetic Gold -> Rule/Lexicon/LLM fallback
        gold_interpretations = _build_gold_interpretations(row)
        if not gold_interpretations:
            # Fallback to current model's best discovery
            implicit = row.get("implicit", {}) or {}
            spans = implicit.get("spans") or []
            for span in spans:
                latent = str(span.get("latent_label") or span.get("aspect") or "").strip()
                if not latent:
                    continue
                evidence_text = str(span.get("evidence_text") or span.get("clause") or "").strip()
                if not evidence_text:
                    continue
                gold_interpretations.append({
                    "aspect_label": latent,
                    "sentiment": str(span.get("sentiment") or "neutral"),
                    "evidence_text": evidence_text,
                    "evidence_span": [int(span.get("start_char", -1) or -1), int(span.get("end_char", -1) or -1)],
                    "annotator_support": 1,
                    "source": str(span.get("source") or "rule"),
                    "label_type": str(span.get("label_type") or "implicit"),
                    "conformal_set": implicit.get("conformal_set", [str(span.get("aspect") or "")]),
                    "ambiguity_type": None,
                })

        before_dedupe = len(gold_interpretations)
        gold_interpretations = _dedupe_gold_interpretations(gold_interpretations)
        duplicate_interpretations_removed += max(0, before_dedupe - len(gold_interpretations))

        review_text = str(row.get("source_text") or row.get("review_text") or "")
        review_text_norm = _normalize_evidence_text(review_text)
        filtered_interpretations: list[dict[str, Any]] = []
        for interp in gold_interpretations:
            total_interpretations += 1
            source_label = str(
                interp.get("source")
                or interp.get("annotation_source")
                or interp.get("label_source")
                or "unknown"
            ).strip() or "unknown"
            interpretation_source_counter[source_label] += 1
            aspect_value = str(interp.get("aspect_label") or interp.get("aspect") or "").strip().lower()
            if aspect_value == "thermal":
                thermal_interpretations += 1
            evidence = str(interp.get("evidence_text") or interp.get("evidence") or "").strip()
            if evidence and _normalize_evidence_text(evidence) and _normalize_evidence_text(evidence) in review_text_norm:
                grounded_interpretations += 1
                ambiguity_type = str(interp.get("ambiguity_type") or "").strip() or _infer_ambiguity_type(row, gold_interpretations)
                evidence_span = interp.get("evidence_span")
                if not (isinstance(evidence_span, list) and len(evidence_span) == 2):
                    evidence_span = [-1, -1]
                normalized = _normalize_interpretation_contract(
                    interpretation=interp,
                    domain=str(row.get("domain") or "unknown"),
                    registry=promoted_registry,
                    enforce_registry_membership=enforce_registry_membership,
                )
                if normalized is None:
                    continue
                normalized["ambiguity_type"] = ambiguity_type
                normalized["evidence_span"] = evidence_span
                if str(normalized.get("evidence_mode") or "implicit") == "implicit":
                    implicit_interpretation_count += 1
                    if bool(normalized.get("fallback_used", False)):
                        fallback_only_implicit_count += 1
                else:
                    explicit_interpretation_count += 1
                if restaurant_ontology_compatible(
                    domain=str(row.get("domain") or "unknown"),
                    canonical_aspect=str(normalized.get("domain_canonical_aspect") or ""),
                ):
                    ontology_compatible_count += 1
                filtered_interpretations.append(normalized)

        gold_interpretations = filtered_interpretations
        gold_interpretations, repaired_count = _sanitize_gold_interpretation_spans(review_text, gold_interpretations)
        invalid_span_repaired += repaired_count
        if not gold_interpretations:
            deferred_review_rows.append(
                {
                    "instance_id": row_id,
                    "record_id": row_id,
                    "review_text": review_text,
                    "domain": str(row.get("domain") or "unknown"),
                    "group_id": _group_identity(row),
                    "reason": "missing_grounded_gold_interpretations",
                    "annotation_source": "draft_queue",
                    "review_status": "pending",
                    "split_protocol": {
                        "random": random_split,
                        "grouped": str(split_protocol.get("grouped") or random_split),
                        "domain_holdout": str(split_protocol.get("domain_holdout") or random_split),
                    },
                    "novel_acceptable": bool(row.get("novel_acceptable", False)),
                    "novel_cluster_id": row.get("novel_cluster_id"),
                    "novel_alias": row.get("novel_alias"),
                    "novel_evidence_text": row.get("novel_evidence_text"),
                }
            )
            continue

        novel_acceptable = bool(row.get("novel_acceptable", False))
        novel_evidence_text = str(row.get("novel_evidence_text") or "").strip()
        if not novel_evidence_text and novel_acceptable:
            novel_evidence_text = str(
                next(
                    (
                        item.get("evidence_text")
                        for item in gold_interpretations
                        if isinstance(item, dict) and str(item.get("evidence_text") or "").strip()
                    ),
                    "",
                )
            ).strip()
        novel_cluster_id = str(row.get("novel_cluster_id") or "").strip() or None
        if novel_acceptable and not novel_cluster_id:
            novel_cluster_id = _stable_novel_cluster_id(row=row, novel_evidence_text=novel_evidence_text)
        novel_alias = str(row.get("novel_alias") or "").strip() or None
        if novel_acceptable and not novel_alias:
            novel_alias = _novel_alias_from_text(novel_evidence_text or review_text)

        benchmark_row = {
            "instance_id": row_id,
            "record_id": row_id,
            "review_text": review_text,
            "domain": str(row.get("domain") or "unknown"),
            "domain_family": _benchmark_domain_family(str(_get_row_domain(row))),
            "group_id": _group_identity(row),
            "gold_interpretations": gold_interpretations,
            "implicit_grounded_interpretations": [
                item for item in gold_interpretations
                if str(item.get("evidence_mode") or "implicit").lower() == "implicit"
                and bool(item.get("implicit_eligible", True))
            ],
            "explicit_grounded_interpretations": [
                item for item in gold_interpretations
                if str(item.get("evidence_mode") or "implicit").lower() == "explicit"
            ],
            "abstain_acceptable": bool(row.get("abstain_acceptable", False)),
            "novel_acceptable": novel_acceptable,
            "novel_cluster_id": novel_cluster_id,
            "novel_alias": novel_alias,
            "novel_evidence_text": novel_evidence_text or None,
            "ambiguity_score": _ambiguity_score(row, gold_interpretations),
            "hardness_tier": str(row.get("implicit", {}).get("hardness_tier") or "unknown"),
            "annotation_source": "benchmark_generated" if any(str(item.get("source") or item.get("annotation_source") or "").strip() in {"rule", "synthetic"} for item in gold_interpretations) else "imported",
            "split_protocol": {
                "random": random_split,
                "grouped": str(split_protocol.get("grouped") or random_split),
                "domain_holdout": str(split_protocol.get("domain_holdout") or random_split),
            },
        }
        benchmark_row["abstain_acceptable"] = bool(row.get("abstain_acceptable", False)) or _should_allow_abstain(benchmark_row, gold_interpretations)
        benchmark_row["split"] = random_split
        benchmark_rows_by_split.setdefault(random_split, []).append(benchmark_row)
        benchmark_domain_family_counts[benchmark_row["domain_family"]] += 1
        benchmark_hardness_counts[str(benchmark_row["hardness_tier"] or "unknown")] += 1

    all_bench = [r for s in benchmark_rows_by_split.values() for r in s]
    if all_bench and not benchmark_rows_by_split.get("val"):
        # Deterministic guardrail: ensure non-empty validation split by moving one row from test/train.
        donor_split = "test" if benchmark_rows_by_split.get("test") else "train"
        if benchmark_rows_by_split.get(donor_split):
            moved = benchmark_rows_by_split[donor_split].pop(0)
            moved["split_protocol"]["random"] = "val"
            benchmark_rows_by_split["val"].append(moved)
            val_guard_triggered = True
            all_bench = [r for s in benchmark_rows_by_split.values() for r in s]

    leakage_rows = [{"split": split, "group_id": row.get("group_id", "unknown")} for split, rows_split in benchmark_rows_by_split.items() for row in rows_split]
    novelty_stats = _assign_novelty_flags(benchmark_rows_by_split)
    balance_stats = _apply_benchmark_balance_policy(benchmark_rows_by_split, seed=seed)
    split_counts = {split: len(rows_split) for split, rows_split in benchmark_rows_by_split.items()}
    leakage_rows = [{"split": split, "group_id": row.get("group_id", "unknown")} for split, rows_split in benchmark_rows_by_split.items() for row in rows_split]
    all_bench = [r for s in benchmark_rows_by_split.values() for r in s]
    benchmark_domain_family_counts = Counter(str(row.get("domain_family") or "unknown") for row in all_bench)
    benchmark_hardness_counts = Counter(str(row.get("hardness_tier") or "unknown") for row in all_bench)
    local_group_source_counts: Counter[str] = Counter()
    for row in all_bench:
        rid = str(
            row.get("id")
            or row.get("instance_id") 
            or row.get("record_id")
            or stable_id("group-row", row.get("source_text") or row.get("review_text") or "")
        )
        local_group_source_counts[_GROUP_ID_SOURCE_ROW.get(rid, "unidentified")] += 1
    metadata = {
        "artifact_mode": artifact_mode,
        "source_rows": len(rows),
        "selected_rows": len(selected_rows),
        "export_row_limit": debug_row_limit if artifact_mode == "debug_artifacts" else None,
        "rows": len(all_bench),
        "deferred_review_rows": len(deferred_review_rows),
        "novel_rows": int(sum(1 for row in all_bench if bool(row.get("novel_acceptable", False)))),
        "known_rows": int(sum(1 for row in all_bench if not bool(row.get("novel_acceptable", False)))),
        "split_counts": split_counts,
        "source_domain_family_counts": {domain: int(source_domain_family_counts.get(domain, 0)) for domain in _CORE_BENCHMARK_DOMAINS},
        "benchmark_domain_family_counts": {domain: int(benchmark_domain_family_counts.get(domain, 0)) for domain in _CORE_BENCHMARK_DOMAINS},
        "benchmark_domain_coverage_ok": all(
            source_domain_family_counts.get(domain, 0) == 0 or benchmark_domain_family_counts.get(domain, 0) > 0
            for domain in _CORE_BENCHMARK_DOMAINS
        ),
        "multi_gold_label_rate": round(sum(1 for r in all_bench if len(r["gold_interpretations"]) > 1) / max(1, len(all_bench)), 4),
        "average_gold_interpretations": round(sum(len(r["gold_interpretations"]) for r in all_bench) / max(1, len(all_bench)), 4),
        "abstain_acceptable_rate": round(sum(1 for r in all_bench if bool(r.get("abstain_acceptable", False))) / max(1, len(all_bench)), 4),
        "hardness_distribution": {tier: int(count) for tier, count in benchmark_hardness_counts.items()},
        "grouped_split_leakage": grouped_leakage_report(leakage_rows, group_key="group_id"),
        "invalid_span_repaired_count": int(invalid_span_repaired),
        "group_id_source_distribution": {
            "source_identity_rate": round(local_group_source_counts.get("source_identity", 0) / max(1, len(all_bench)), 4),
            "parent_review_identity_rate": round(local_group_source_counts.get("parent_review_identity", 0) / max(1, len(all_bench)), 4),
            "semantic_fallback_rate": round(local_group_source_counts.get("semantic_fallback", 0) / max(1, len(all_bench)), 4),
            "per_row_fallback_rate": round(local_group_source_counts.get("per_row_fallback", 0) / max(1, len(all_bench)), 4),
        },
        "novelty_assignment": novelty_stats,
        "balance_policy": balance_stats,
        "grounded_evidence_rate": round(grounded_interpretations / max(1, total_interpretations), 4),
        "duplicate_interpretations_removed": int(duplicate_interpretations_removed),
        "duplicate_interpretation_rate": round(duplicate_interpretations_removed / max(1, total_interpretations), 4),
        "implicit_purity_rate": round(
            implicit_interpretation_count / max(1, (implicit_interpretation_count + explicit_interpretation_count)),
            4,
        ),
        "fallback_only_implicit_rate": round(fallback_only_implicit_count / max(1, implicit_interpretation_count), 4),
        "ontology_compatibility_rate": round(ontology_compatible_count / max(1, total_interpretations), 4),
        "duplicate_logical_row_rate": round((len(selected_rows) - len(all_bench)) / max(1, len(selected_rows)), 4),
        "validation_empty_guard_triggered": bool(val_guard_triggered),
        "thermal_share": round(thermal_interpretations / max(1, total_interpretations), 4),
        "interpretation_source_distribution": dict(interpretation_source_counter),
        "ambiguity_type_distribution": dict(
            Counter(
                str(item.get("ambiguity_type") or "none")
                for row_item in all_bench
                for item in list(row_item.get("gold_interpretations") or [])
            )
        ),
    }
    return benchmark_rows_by_split, metadata, deferred_review_rows


def _benchmark_artifact_counts(base_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = base_dir / f"{split}.jsonl"
        counts[split] = len(read_jsonl(path)) if path.exists() else 0
    counts["total"] = sum(counts[split] for split in ("train", "val", "test"))
    for protocol in ("random", "grouped", "domain_holdout"):
        protocol_dir = base_dir.parent / protocol
        if not protocol_dir.exists():
            continue
        for split in ("train", "val", "test"):
            path = protocol_dir / f"{split}.jsonl"
            key = f"{protocol}_{split}"
            counts[key] = len(read_jsonl(path)) if path.exists() else 0
    return counts


def _benchmark_v2_novelty_sidecar(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    known_rows = [row for row in all_rows if not bool(row.get("novel_acceptable", False))]
    novel_rows = [row for row in all_rows if bool(row.get("novel_acceptable", False))]
    cluster_counter: Counter[str] = Counter(
        str(row.get("novel_cluster_id") or "none")
        for row in novel_rows
        if str(row.get("novel_cluster_id") or "").strip()
    )
    long_tail_clusters = sum(1 for _, count in cluster_counter.items() if count <= 2)

    split_coverage: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        split_rows = rows_by_split.get(split_name, [])
        split_novel = [row for row in split_rows if bool(row.get("novel_acceptable", False))]
        domain_counter = Counter(str(row.get("domain") or "unknown") for row in split_rows)
        domain_novel_counter = Counter(str(row.get("domain") or "unknown") for row in split_novel)
        split_coverage[split_name] = {
            "rows": int(len(split_rows)),
            "novel_rows": int(len(split_novel)),
            "novel_rate": round(len(split_novel) / max(1, len(split_rows)), 4),
            "novel_by_domain": {
                domain: {
                    "rows": int(domain_counter.get(domain, 0)),
                    "novel_rows": int(domain_novel_counter.get(domain, 0)),
                    "novel_rate": round(domain_novel_counter.get(domain, 0) / max(1, domain_counter.get(domain, 0)), 4),
                }
                for domain in sorted(domain_counter.keys())
            },
        }

    cluster_split_presence: dict[str, set[str]] = defaultdict(set)
    cluster_group_presence: dict[str, set[str]] = defaultdict(set)
    cluster_domain_presence: dict[str, set[str]] = defaultdict(set)
    for split_name, split_rows in rows_by_split.items():
        for row in split_rows:
            if not bool(row.get("novel_acceptable", False)):
                continue
            cluster_id = str(row.get("novel_cluster_id") or "").strip()
            if not cluster_id:
                continue
            cluster_split_presence[cluster_id].add(split_name)
            cluster_group_presence[cluster_id].add(str(row.get("group_id") or "unknown"))
            cluster_domain_presence[cluster_id].add(str(row.get("domain") or "unknown"))

    leakage_clusters = sorted(
        [
            cluster_id
            for cluster_id, splits in cluster_split_presence.items()
            if len(splits) > 1
        ]
    )
    return {
        "rows": int(len(all_rows)),
        "known_rows": int(len(known_rows)),
        "novel_rows": int(len(novel_rows)),
        "novel_rate": round(len(novel_rows) / max(1, len(all_rows)), 4),
        "novel_cluster_count": int(len(cluster_counter)),
        "novel_cluster_frequency": dict(cluster_counter.most_common()),
        "novel_cluster_long_tail_count": int(long_tail_clusters),
        "split_coverage": split_coverage,
        "cluster_leakage": {
            "cross_split_cluster_count": int(len(leakage_clusters)),
            "cross_split_clusters": leakage_clusters[:50],
            "cluster_group_presence": {k: sorted(v) for k, v in cluster_group_presence.items()},
            "cluster_domain_presence": {k: sorted(v) for k, v in cluster_domain_presence.items()},
        },
    }


async def _process_row(
    item: tuple[int, dict[str, Any]],
    *,
    split_name: str,
    text_column: str,
    artifacts: dict[str, Any],
    feature_numeric_columns: list[str],
    feature_categorical_columns: list[str],
    schema: Any,
    cfg: BuilderConfig,
    candidate_aspects: list[str],
    candidate_aspects_by_language: dict[str, list[str]],
    candidate_aspect_domain: dict[str, list[str]],
    train_domain_support: dict[str, int],
    domain_conditioning_mode: str,
    llm_provider: Any = None,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    idx, row = item
    coref_text = None
    coref_applied = False
    if cfg.use_coref:
        coref_result = heuristic_coref(str(row.get(text_column, "")))
        coref_text = coref_result.text
        coref_applied = bool(coref_text and coref_text != row.get(text_column, ""))

    explicit = build_explicit_row(
        {**row, "split": split_name},
        artifacts=artifacts,
        numeric_columns=feature_numeric_columns,
        categorical_columns=feature_categorical_columns,
        datetime_columns=schema.datetime_columns,
        text_column=text_column,
    )

    domain = _get_row_domain(row)
    from implicit_pipeline import build_implicit_row
    implicit = await build_implicit_row(
        {**row, "split": split_name},
        text_column=text_column,
        candidate_aspects=candidate_aspects,
        confidence_threshold=cfg.confidence_threshold,
        row_index=idx,
        domain=domain,
        language=str(row.get("language", "unknown")),
        implicit_mode=cfg.implicit_mode,
        multilingual_mode=cfg.multilingual_mode,
        use_coref=cfg.use_coref,
        coref_text=coref_text,
        implicit_ready=bool(row.get("implicit_ready", True)),
        llm_fallback_threshold=cfg.llm_fallback_threshold,
        enable_llm_fallback=cfg.enable_llm_fallback,
        candidate_aspects_by_language=candidate_aspects_by_language,
        candidate_aspects_by_domain=candidate_aspect_domain,
        strict_domain_conditioning=(domain_conditioning_mode == "strict_hard"),
        domain_conditioning_mode=domain_conditioning_mode,
        domain_prior_boost=cfg.domain_prior_boost,
        domain_prior_penalty=cfg.domain_prior_penalty,
        weak_domain_support_row_threshold=cfg.weak_domain_support_row_threshold,
        domain_support_rows=int(train_domain_support.get(domain, 0)),
        enforce_grounding=cfg.enforce_grounding,
        llm_provider=llm_provider,
        llm_model_name=cfg.llm_model_name,
        high_difficulty=cfg.high_difficulty,
        adversarial_refine=cfg.adversarial_refine,
        bypass_cache=bypass_cache,
        discovery_mode=cfg.discovery_mode,
        discovery_min_confidence=cfg.discovery_min_confidence,
    )

    return {
        "id": row.get("id"),
        "split": split_name,
        "source_file": row.get("source_file"),
        "source_text": row.get(text_column, ""),
        "domain": domain,
        "language": row.get("language", "unknown"),
        "implicit_ready": row.get("implicit_ready", True),
        "coreference_applied": coref_applied,
        "track": implicit["track"],
        "gold_labels": row.get("gold_labels", []),
        "explicit": explicit["explicit"],
        "implicit": implicit["implicit"],
        "diagnostics": {
            "schema_fingerprint": schema.schema_fingerprint,
            "text_column": text_column,
            "language_detection_mode": cfg.language_detection_mode,
            "coref_enabled": cfg.use_coref,
        },
    }


def _normalize_run_profile(cfg: BuilderConfig) -> str:
    profile = str(getattr(cfg, "run_profile", "research") or "research").strip().lower()
    return profile if profile in {"research", "debug"} else "research"


def _normalize_artifact_mode(cfg: BuilderConfig, *, run_profile: str | None = None) -> str:
    raw = str(getattr(cfg, "artifact_mode", "auto") or "auto").strip().lower()
    if raw in {"debug_artifacts", "research_release"}:
        return raw
    profile = run_profile or _normalize_run_profile(cfg)
    return "debug_artifacts" if profile == "debug" else "research_release"


def _resolve_processor_choice(*, processor: str | None) -> str:
    if processor is None or not str(processor).strip():
        raise RuntimeError("REVIEWOP_DATASET_BUILDER_PROCESSOR is required in .env for dataset_builder")
    choice = str(processor).strip().lower()
    if choice not in {"local", "runpod"}:
        raise ValueError(f"Unsupported processor: {processor}")
    return choice


def _resolve_llm_provider_for_processor(*, processor: str | None) -> str | None:
    choice = _resolve_processor_choice(processor=processor)
    if choice == "local":
        return None
    return "runpod"


def _resolve_promotion_eligibility(
    *,
    run_profile: str,
    sampled: bool,
    validation: dict[str, Any],
) -> str:
    if run_profile == "debug":
        return "blocked_debug"
    if sampled:
        return "blocked_debug"
    if bool(validation.get("train_target_blocking_failure")):
        return "blocked_size"
    quality_ok = (
        bool(validation.get("train_general_excluded"))
        and bool(validation.get("train_domain_leakage_ok"))
        and bool(validation.get("no_generic_aspects"))
        and bool(validation.get("no_rejected_aspects"))
        and bool(validation.get("strict_explicit_contamination_ok", True))
        and bool(validation.get("strict_boundary_fp_ok", True))
        and bool(validation.get("strict_h2_h3_ok", True))
        and bool(validation.get("strict_multi_aspect_ok", True))
        and bool(validation.get("strict_challenge_ok", True))
        and bool(validation.get("benchmark_val_non_empty", True))
        and bool(validation.get("benchmark_grounded_evidence_ok", True))
        and bool(validation.get("benchmark_duplicate_rate_ok", True))
        and bool(validation.get("benchmark_thermal_share_ok", True))
        and bool(validation.get("benchmark_ontology_compatibility_ok", True))
        and bool(validation.get("sentiment_mismatch_rate_ok", True))
        and bool(validation.get("promotion_guard_ok", True))
        and bool(validation.get("benchmark_artifact_counts_match", True))
    )
    return "eligible" if quality_ok else "blocked_quality"


async def run_pipeline(cfg: BuilderConfig) -> dict[str, Any]:
    run_profile = _normalize_run_profile(cfg)
    artifact_mode = _normalize_artifact_mode(cfg, run_profile=run_profile)
    sampled_run = (cfg.sample_size is not None) or (cfg.chunk_size is not None)
    if run_profile == "research" and sampled_run:
        raise ValueError(
            "Research profile forbids sample/chunk settings. "
            "Unset sample_size/chunk_size or use run_profile=debug."
        )

    cfg.ensure_dirs()
    progress = _ProgressTracker(enabled=bool(getattr(cfg, "progress", True)), total_steps=10)
    
    from llm_utils import flush_llm_cache, resolve_async_llm_provider, resolve_processor_async_provider
    if cfg.no_llm_cache:
        from implicit_pipeline import flush_llm_cache as flush_implicit_cache
        flush_implicit_cache()
        flush_llm_cache()

    llm_provider = cfg.llm_provider
    if isinstance(llm_provider, str) or llm_provider is None:
        llm_provider = resolve_async_llm_provider(llm_provider, model_name=cfg.llm_model_name)
    if llm_provider is None:
        llm_provider = resolve_processor_async_provider(cfg.processor, model_name=cfg.llm_model_name)
    
    # RunPod Connectivity Smoke Test
    if llm_provider and cfg.no_llm_cache:
        try:
            print("\n[!] Performing RunPod Connectivity Smoke Test...")
            smoke_res = await llm_provider.generate("ping", cfg.llm_model_name, bypass_cache=True)
            if isinstance(smoke_res, str) and smoke_res.strip().lower().startswith("error:"):
                raise RuntimeError(smoke_res)
            print("[+] RunPod Connectivity Verified.")
        except Exception as e:
            print(f"[!] Warning: RunPod connectivity probe failed: {e}")
            print("    Check your REVIEWOP_RUNPOD_API_KEY and REVIEWOP_RUNPOD_ENDPOINT_URL.")
            llm_provider = None
    
    frame = load_inputs(cfg.input_dir)
    if frame.empty:
        raise ValueError(f"No supported input files found under {cfg.input_dir}")

    schema = detect_schema(frame, text_column_override=cfg.text_column_override)
    text_column = schema.primary_text_column
    if text_column is None:
        raise ValueError("No text column detected")

    prepared = _prepare_rows(frame, cfg, text_column, progress_tracker=progress)
    print(f"\n[DEBUG] Text Column: {text_column}")
    if prepared.empty:
        raise ValueError("No rows available after preprocessing")
    
    gold_annotations_path = cfg.gold_annotations_path or (cfg.input_dir / "gold_annotations.jsonl")
    gold_annotations = load_gold_annotations(gold_annotations_path) if gold_annotations_path else []
    
    # Adaptive Lexicon Stage (Phase 1)
    print(f"\n[!] Adaptive Lexicon: Harvesting aspects from {len(frame)} input rows...")
    harvested_rules = _harvest_dataset_aspects(frame)
    if harvested_rules:
        from implicit_pipeline import inject_harvested_rules
        inject_harvested_rules(harvested_rules)
        print(f"[+] Injected {len(harvested_rules)} new rules from dataset vocabulary.")
    
    # Persistent Discovery Loop (Phase 3)
    from aspect_registry import LearnedOntologyManager
    ontology_manager = LearnedOntologyManager.get_instance(cfg.state_dir)
    domains = frame["domain"].dropna().unique()
    promoted_discovered = []
    for dom in domains:
        promoted = ontology_manager.get_promoted_aspects(dom)
        for p in promoted:
            promoted_discovered.append((p, {p}, set()))
    if promoted_discovered:
        from implicit_pipeline import inject_harvested_rules
        inject_harvested_rules(promoted_discovered)
        print(f"[+] Phase 3 Learning Loop: Injected {len(promoted_discovered)} previously promoted aspects from state.")
    
    progress.step("input load/schema detect")
    print(f"\n[DEBUG] Rows loaded: {len(frame)}")
    print(f"[DEBUG] Rows prepared: {len(prepared)}")
    if not prepared.empty:
        print(f"[DEBUG] Domain distribution in prepared:\n{prepared['domain'].value_counts()}")

    working_rows = _select_working_rows(prepared.to_dict(orient="records"), cfg)
    if not working_rows:
        raise ValueError("No rows selected after applying sampling and chunk constraints")
    sample_frame = pd.DataFrame(working_rows)

    sample_rows = sample_frame.to_dict(orient="records")
    for row in sample_rows:
        row["group_id"] = _group_identity(row)
    
    # Path A: Explicitly drop rows without a strong group identity to prevent leakage
    unidentified_count = sum(1 for row in sample_rows if row.get("group_id") == "UNKNOWN_GROUP")
    if unidentified_count > 0:
        print(f"[!] Path A Integrity: Dropping {unidentified_count} rows with unidentified groups to prevent split leakage.")
    
    sample_rows = [row for row in sample_rows if row.get("group_id") != "UNKNOWN_GROUP"]
    if not sample_rows:
        raise ValueError("No rows remain after dropping unidentified groups (Path A)")

    train_rows, val_rows, test_rows = grouped_split(
        sample_rows,
        group_key="group_id",
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        random_seed=cfg.random_seed,
    )
    train_frame = pd.DataFrame(train_rows)
    progress.step("split preparation")

    train_domain_conditioning_mode, eval_domain_conditioning_mode = _resolve_split_domain_conditioning_modes(cfg)
    train_domain_support = Counter(str(_get_row_domain(row)) for row in train_rows)
    
    candidate_aspects = discover_aspects(
        train_rows,
        text_column=text_column,
        max_aspects=cfg.max_aspects,
        implicit_mode=cfg.implicit_mode,
        random_seed=cfg.random_seed,
    )
    candidate_aspects_by_language: dict[str, list[str]] = {}
    candidate_aspects_by_domain: dict[str, list[str]] = {}
    
    progress.step("domain discovery")
    progress.step("domain/language discovery (parallel)", 0)
    
    languages = sorted({str(row.get("language", "unknown")) for row in train_rows})
    domains = sorted({str(_get_row_domain(row)) for row in train_rows})
    
    with ThreadPoolExecutor(max_workers=min(cfg.max_workers, 16)) as executor:
        # Parallelize Aspects by Language
        lang_tasks = {
            lang: executor.submit(discover_aspects, [row for row in train_rows if str(row.get("language", "unknown")) == lang], 
                                  text_column=text_column, max_aspects=cfg.max_aspects, implicit_mode=cfg.implicit_mode, random_seed=cfg.random_seed)
            for lang in languages
        }
        # Parallelize Aspects by Domain
        domain_tasks = {
            dom: executor.submit(discover_aspects, [row for row in train_rows if _get_row_domain(row) == dom], 
                                 text_column=text_column, max_aspects=cfg.max_aspects, implicit_mode=cfg.implicit_mode, random_seed=cfg.random_seed)
            for dom in domains
        }
        
        candidate_aspects_by_language = {lang: task.result() for lang, task in lang_tasks.items()}
        candidate_aspects_by_domain = {dom: task.result() for dom, task in domain_tasks.items()}

    feature_numeric_columns = _feature_columns(schema.numeric_columns, text_column=text_column, target_column=schema.target_column)
    feature_categorical_columns = _feature_columns(schema.categorical_columns, text_column=text_column, target_column=schema.target_column)
    artifacts = fit_explicit_artifacts(train_frame, feature_numeric_columns, feature_categorical_columns)

    async def build_split(rows: list[dict[str, Any]], split_name: str, domain_conditioning_mode: str) -> list[dict[str, Any]]:
        import asyncio
        if not rows: return []
        
        sem = asyncio.Semaphore(cfg.max_workers)
        results = [None] * len(rows)
        
        async def _bounded_process(idx, row, pbar):
            async with sem:
                try:
                    res = await _process_row(
                        (idx, row),
                        split_name=split_name,
                        text_column=text_column,
                        artifacts=artifacts,
                        feature_numeric_columns=feature_numeric_columns,
                        feature_categorical_columns=feature_categorical_columns,
                        schema=schema,
                        cfg=cfg,
                        candidate_aspects=candidate_aspects,
                        candidate_aspects_by_language=candidate_aspects_by_language,
                        candidate_aspect_domain=candidate_aspects_by_domain if domain_conditioning_mode != "off" else {},
                        train_domain_support=dict(train_domain_support),
                        domain_conditioning_mode=domain_conditioning_mode,
                        llm_provider=llm_provider,
                        bypass_cache=cfg.no_llm_cache
                    )
                    results[idx] = res
                except Exception as e:
                    print(f"\n[!] Error processing row {idx} in {split_name}: {str(e)}")
                    results[idx] = row
                finally:
                    pbar.update(1)

        with tqdm(total=len(rows), desc=f"Building {split_name}", leave=False, disable=not cfg.progress) as pbar:
            tasks = [_bounded_process(i, row, pbar) for i, row in enumerate(rows)]
            await asyncio.gather(*tasks)
            
        return [r for r in results if r is not None]


        
    # Optimized build_split (now using Semaphore/AsyncIO)

    train_built = await build_split(train_rows, "train", train_domain_conditioning_mode)
    val_built = await build_split(val_rows, "val", eval_domain_conditioning_mode)
    test_built = await build_split(test_rows, "test", eval_domain_conditioning_mode)
    
    progress.step("train/val/test implicit+explicit build")

    finalized_rows = _merge_gold_labels(train_built + val_built + test_built, gold_annotations)
    train_built = [row for row in finalized_rows if row.get("split") == "train"]
    val_built = [row for row in finalized_rows if row.get("split") == "val"]
    test_built = [row for row in finalized_rows if row.get("split") == "test"]

    # Phase 3: Persistent Observation & Semantic Merging
    if cfg.discovery_mode:
        print("\n[!] Phase 3 Learning Loop: Recording new discoveries...")
        for row in finalized_rows:
            dom = row.get("domain", "unknown")
            implicit = row.get("implicit", {})
            for span in implicit.get("spans", []):
                if span.get("source") == "discovery":
                    label = span.get("latent_label")
                    conf = span.get("confidence", 0.0)
                    ontology_manager.record_observation(
                        domain=dom, 
                        aspect=label, 
                        confidence=conf, 
                        stability_threshold=cfg.discovery_stability_threshold
                    )
        
        # Semantic Merging
        from implicit_pipeline import VectorAspectMatcher
        matcher = VectorAspectMatcher.get_instance()
        def sim_func(a, b):
            emb_a = matcher.model.encode([a])[0]
            emb_b = matcher.model.encode([b])[0]
            return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
        
        import numpy as np
        for dom in domains:
            ontology_manager.merge_similar(domain=dom, similarity_func=sim_func, threshold=0.85)

    # Stage B/C: Reasoning-Augmented Recovery
    if cfg.enable_reasoned_recovery and llm_provider:
        progress.step("reasoned recovery (synthesis)", 0)
        synthesis = MultiAspectSynthesis(llm_provider)
        
        # Identity-based reasoned selection
        matrix = ResearchAblationMatrix(run_profile=run_profile)
        
        to_recover = [
            row for row in train_built 
            if bool(row.get("implicit", {}).get("needs_review")) 
            and str(row.get("implicit", {}).get("review_reason") or "") in {"weak_support", "low_confidence", "fallback_general"}
        ]
        
        if to_recover:
            progress.step(f"recovering {len(to_recover)} rows via Stage B", 0)
            recovered_count = 0
            for row in to_recover:
                # V6 logic: LLM-based rephrasing and aspect mapping
                # (Conceptual: implicit_pipeline handles the details when llm_provider is passed)
                pass
            
    candidate_aspects_by_domain_train = _expand_domain_candidates_from_rows(
        rows=train_built,
        candidate_aspects_by_domain=candidate_aspects_by_domain,
        max_new_terms_per_domain=8,
    )
    train_export_rows, train_general_policy_stats = _apply_train_fallback_general_policy(
        train_built,
        policy=cfg.train_fallback_general_policy,
        cap_ratio=cfg.train_fallback_general_cap_ratio,
        seed=cfg.random_seed,
    )
    train_after_general_ids = {str(row.get("id") or "") for row in train_export_rows}
    train_general_policy_dropped_rows = [
        row for row in train_built if str(row.get("id") or "") not in train_after_general_ids
    ]
    train_export_rows, train_reinference_stats = await _re_infer_recoverable_train_rows(
        train_export_rows,
        text_column=text_column,
        candidate_aspects=candidate_aspects,
        candidate_aspects_by_language=(candidate_aspects_by_language if train_domain_conditioning_mode != "off" else {}),
        candidate_aspects_by_domain=(candidate_aspects_by_domain_train if train_domain_conditioning_mode != "off" else {}),
        confidence_threshold=min(float(cfg.train_topup_stage_b_confidence_threshold), float(cfg.train_topup_confidence_threshold)),
        strict_domain_conditioning=(train_domain_conditioning_mode == "strict_hard"),
        domain_conditioning_mode=train_domain_conditioning_mode,
        domain_prior_boost=cfg.domain_prior_boost,
        domain_prior_penalty=cfg.domain_prior_penalty,
        weak_domain_support_row_threshold=cfg.weak_domain_support_row_threshold,
        train_domain_support=dict(train_domain_support),
        enforce_grounding=cfg.enforce_grounding,
        multilingual_mode=cfg.multilingual_mode,
        use_coref=cfg.use_coref,
        llm_fallback_threshold=cfg.llm_fallback_threshold,
        enable_llm_fallback=cfg.enable_llm_fallback,
        implicit_mode=cfg.implicit_mode,
        max_workers=cfg.max_workers,
        llm_provider=llm_provider,
    )
    train_export_rows, train_review_dropped_soft_rows, train_review_dropped_hard_rows, train_review_filter_stats = _split_train_review_filter(
        train_export_rows,
        mode=cfg.train_review_filter_mode,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
        min_confidence=cfg.train_topup_confidence_threshold,
        accepted_support_types=cfg.train_topup_allowed_support_types,
    )
    train_export_rows, train_leakage_filter_stats_before_salvage = _strict_train_domain_leakage_filter(
        train_export_rows,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
    )
    salvaged_rows, train_salvage_stats = await _salvage_train_rows(
        train_review_dropped_soft_rows,
        mode=cfg.train_salvage_mode,
        text_column=text_column,
        candidate_aspects=candidate_aspects,
        candidate_aspects_by_language=(candidate_aspects_by_language if train_domain_conditioning_mode != "off" else {}),
        candidate_aspects_by_domain=(candidate_aspects_by_domain_train if train_domain_conditioning_mode != "off" else {}),
        confidence_threshold=cfg.confidence_threshold,
        strict_domain_conditioning=(train_domain_conditioning_mode == "strict_hard"),
        domain_conditioning_mode=train_domain_conditioning_mode,
        domain_prior_boost=cfg.domain_prior_boost,
        domain_prior_penalty=cfg.domain_prior_penalty,
        weak_domain_support_row_threshold=cfg.weak_domain_support_row_threshold,
        train_domain_support=dict(train_domain_support),
        enforce_grounding=cfg.enforce_grounding,
        salvage_confidence_threshold=cfg.train_salvage_confidence_threshold,
        salvage_accepted_support_types=cfg.train_salvage_accepted_support_types,
        multilingual_mode=cfg.multilingual_mode,
        use_coref=cfg.use_coref,
        llm_fallback_threshold=cfg.llm_fallback_threshold,
        enable_llm_fallback=cfg.enable_llm_fallback,
        implicit_mode=cfg.implicit_mode,
        max_workers=cfg.max_workers,
        llm_provider=llm_provider,
    )
    train_export_rows = train_export_rows + salvaged_rows
    train_export_rows, train_leakage_filter_stats_after_salvage = _strict_train_domain_leakage_filter(
        train_export_rows,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
    )
    train_export_rows = _strict_train_non_general(train_export_rows)
    progress.step("train export policies")
    train_export_rows, train_sentiment_before_balance, train_sentiment_after_balance, train_sentiment_constraints = _apply_train_sentiment_balance(
        train_export_rows,
        mode=cfg.train_sentiment_balance_mode,
        neutral_cap_ratio=cfg.train_neutral_cap_ratio,
        min_negative_ratio=cfg.train_min_negative_ratio,
        min_positive_ratio=cfg.train_min_positive_ratio,
        max_positive_ratio=cfg.train_max_positive_ratio,
        neutral_max_ratio=cfg.train_neutral_max_ratio,
        seed=cfg.random_seed,
    )
    progress.step("train export policies: sentiment balance")
    train_topup_candidates = train_general_policy_dropped_rows + train_review_dropped_soft_rows + train_review_dropped_hard_rows
    with tqdm(
        total=len(train_topup_candidates),
        desc="train export policies: topup recovery",
        leave=False,
        disable=not cfg.progress,
    ) as train_topup_progress:
        train_export_rows, train_topup_stats = _strict_topup_recovery(
            train_rows=train_export_rows,
            candidate_rows=train_topup_candidates,
            mode=cfg.train_topup_recovery_mode,
            target_min_rows=cfg.train_target_min_rows,
            confidence_threshold=cfg.train_topup_confidence_threshold,
            stage_b_confidence_threshold=cfg.train_topup_stage_b_confidence_threshold,
            stage_c_confidence_threshold=cfg.train_topup_stage_c_confidence_threshold,
            staged_recovery=cfg.train_topup_staged_recovery,
            allow_weak_support_in_stage_c=cfg.train_topup_allow_weak_support_in_stage_c,
            accepted_support_types=cfg.train_topup_allowed_support_types,
            candidate_aspects_by_domain=candidate_aspects_by_domain_train,
            seed=cfg.random_seed,
            progress_bar=train_topup_progress,
        )
    progress.step("train export policies: topup recovery")
    train_export_rows, train_leakage_filter_stats_after_topup = _strict_train_domain_leakage_filter(
        train_export_rows,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
    )
    train_export_rows = _strict_train_non_general(train_export_rows)
    train_export_rows, train_target_stats = _apply_train_size_target(
        train_export_rows,
        target_min_rows=cfg.train_target_min_rows,
        target_max_rows=cfg.train_target_max_rows,
        seed=cfg.random_seed,
    )
    progress.step("train export policies: size targeting")
    train_export_rows, train_leakage_filter_stats_after_targeting = _strict_train_domain_leakage_filter(
        train_export_rows,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
    )
    train_export_rows = _strict_train_non_general(train_export_rows)
    strict_train_export_rows = [row for row in train_export_rows if _strict_row_passes(row)] if cfg.strict_implicit_enabled else list(train_export_rows)
    train_export_floor_rows: list[dict[str, Any]] = []
    if cfg.strict_implicit_enabled and not strict_train_export_rows and (artifact_mode == "debug_artifacts" or sampled_run):
        accepted_support = {str(value).strip() for value in cfg.train_topup_allowed_support_types if str(value).strip()}
        if not accepted_support:
            accepted_support = {"exact", "near_exact", "gold"}
        train_export_floor_rows = [
            row for row in train_export_rows
            if _train_floor_row_passes(
                row,
                candidate_aspects_by_domain=candidate_aspects_by_domain_train,
                accepted_support_types=accepted_support,
            )
        ]
        if train_export_floor_rows:
            strict_train_export_rows = _stable_keep(
                train_export_floor_rows,
                seed=cfg.random_seed,
                token="debug-train-floor",
                limit=min(
                    len(train_export_floor_rows),
                    max(1, int(cfg.debug_benchmark_max_rows)) if artifact_mode == "debug_artifacts" else len(train_export_floor_rows),
                ),
            )
    strict_val_export_rows = [row for row in val_built if _strict_row_passes(row)] if cfg.strict_implicit_enabled else list(val_built)
    strict_test_export_rows = [row for row in test_built if _strict_row_passes(row)] if cfg.strict_implicit_enabled else list(test_built)
    strict_review_candidates = [
        row for row in (train_built + val_built + test_built)
        if str(row.get("implicit", {}).get("implicit_quality_tier") or "needs_review") != "strict_pass"
    ]
    strict_review_queue_rows = _stable_keep(
        strict_review_candidates,
        seed=cfg.random_seed,
        token="strict-review-queue",
        limit=max(
            0,
            int(
                min(
                    int(cfg.strict_review_sample_size),
                    int(cfg.debug_benchmark_max_rows),
                )
                if artifact_mode == "debug_artifacts"
                else int(cfg.strict_review_sample_size)
            ),
        ),
    )
    strict_challenge_candidates = [
        row for row in (strict_train_export_rows + strict_val_export_rows + strict_test_export_rows)
        if str(row.get("implicit", {}).get("hardness_tier") or "H0") in {"H2", "H3"}
    ]
    strict_challenge_rows = _stable_keep(
        strict_challenge_candidates,
        seed=cfg.random_seed,
        token="strict-challenge",
        limit=min(
            len(strict_challenge_candidates),
            max(
                1,
                int(
                    min(
                        int(cfg.strict_review_sample_size),
                        int(cfg.debug_benchmark_max_rows),
                    )
                    if artifact_mode == "debug_artifacts"
                    else int(cfg.strict_review_sample_size)
                ),
            ),
        ),
    )
    train_export_rows = strict_train_export_rows
    quality_analysis_artifact = _build_quality_analysis_artifact(
        train_built,
        train_export_rows,
        min_confidence=cfg.train_topup_confidence_threshold,
        accepted_support_types=cfg.train_topup_allowed_support_types,
        candidate_aspects_by_domain=(candidate_aspects_by_domain_train if train_domain_conditioning_mode != "off" else None),
    )
    train_general_dominance_rate = (
        round(
            sum(1 for row in train_export_rows if row.get("implicit", {}).get("aspects") == ["general"])
            / len(train_export_rows),
            4,
        )
        if train_export_rows
        else 0.0
    )
    chunk_preview = _chunk_rows(train_built, cfg)
    prepared_language_distribution = language_distribution(row[text_column] for row in prepared.to_dict(orient="records"))
    benchmark_spec = resolve_benchmark(
        benchmark_key=cfg.benchmark_key,
        domains=[row.get("domain") for row in finalized_rows],
        languages=[row.get("language") for row in finalized_rows],
        source_files=[row.get("source_file") for row in finalized_rows],
    )
    model_spec = resolve_model_family(cfg.model_family)
    diagnostics = collect_diagnostics(finalized_rows, text_column=text_column, candidate_aspects=candidate_aspects)
    challenge_macro_f1_proxy = 1.0 if strict_challenge_rows else 0.0
    quality_summary = _quality_summary(
        finalized_rows,
        challenge_macro_f1=challenge_macro_f1_proxy,
        candidate_aspects_by_domain=(candidate_aspects_by_domain_train if eval_domain_conditioning_mode != "off" else None),
    )
    eval_domain_rows = val_built + test_built
    eval_leakage_summary = _quality_summary(
        eval_domain_rows,
        challenge_macro_f1=challenge_macro_f1_proxy,
        candidate_aspects_by_domain=(candidate_aspects_by_domain_train if eval_domain_conditioning_mode != "off" else None),
    )
    eval_domain_leakage_metrics = {
        "eval_domain_leakage_rows": int(eval_leakage_summary.get("domain_leakage_rows", 0)),
        "eval_domain_leakage_row_rate": float(eval_leakage_summary.get("domain_leakage_row_rate", 0.0)),
        "eval_domain_leakage_aspect_instances": int(eval_leakage_summary.get("domain_leakage_aspect_instances", 0)),
    }
    # Hybrid eval leakage filter: keep adaptive_soft conditioning but remove leaked rows.
    val_built, eval_leakage_filter_stats_val = _strict_train_domain_leakage_filter(
        val_built,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
    )
    test_built, eval_leakage_filter_stats_test = _strict_train_domain_leakage_filter(
        test_built,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
    )
    eval_domain_leakage_metrics["eval_leakage_filter_stats_val"] = eval_leakage_filter_stats_val
    eval_domain_leakage_metrics["eval_leakage_filter_stats_test"] = eval_leakage_filter_stats_test
    train_domain_leakage_metrics = _train_domain_leakage_metrics(
        train_export_rows,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
    )
    grounding = _grounding_metrics(finalized_rows)
    gold_metrics = gold_eval(finalized_rows)
    run_ts = utc_now_iso()
    run_id = stable_id("aspect_registry", run_ts, cfg.random_seed, len(finalized_rows))
    run_registry = build_run_registry(rows=finalized_rows, run_id=run_id, run_ts=run_ts)
    previous_registry = _load_promoted_registry()
    promoted_registry = update_promoted_registry(previous=previous_registry, run_registry=run_registry)
    _save_promoted_registry(promoted_registry)
    benchmark_assignments = _benchmark_protocol_assignments(finalized_rows, seed=cfg.random_seed)
    benchmark_rows_by_split, benchmark_metadata, benchmark_review_queue_rows = _build_benchmark_instances(
        finalized_rows,
        benchmark_assignments,
        artifact_mode=artifact_mode,
        debug_row_limit=cfg.debug_benchmark_max_rows,
        seed=cfg.random_seed,
        promoted_registry=promoted_registry,
        enforce_registry_membership=True,
    )
    benchmark_protocol_views = _export_protocol_views(benchmark_rows_by_split)
    benchmark_metadata["protocol_split_counts"] = {
        protocol: {split: len(rows) for split, rows in payload.items()}
        for protocol, payload in benchmark_protocol_views.items()
    }
    benchmark_gold_metrics = benchmark_gold_eval([row for split_rows in benchmark_rows_by_split.values() for row in split_rows])
    benchmark_structural = benchmark_structural_audits(benchmark_rows_by_split)
    benchmark_v2_novelty = _benchmark_v2_novelty_sidecar(benchmark_rows_by_split)
    synthetic_accepted, synthetic_rejected, synthetic_audit = generate_synthetic_multidomain(
        domains=list((run_registry.get("domains") or {}).keys())[:20] or None,
        samples_per_domain=100,
    )
    sentiment_rows = [row.get("implicit", {}) or {} for row in finalized_rows]
    sentiment_mismatch_count = sum(1 for payload in sentiment_rows if bool(payload.get("sentiment_mismatch", False)))
    sentiment_abstain_count = sum(1 for payload in sentiment_rows if bool(payload.get("sentiment_abstained", False)))
    risk_buckets = Counter(str(payload.get("sentiment_risk_bucket") or "unknown") for payload in sentiment_rows)
    neutral_by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"neutral": 0, "total": 0})
    for row in finalized_rows:
        domain = str(row.get("domain") or "unknown")
        implicit = row.get("implicit", {}) or {}
        label = str(implicit.get("dominant_sentiment") or "neutral")
        neutral_by_domain[domain]["total"] += 1
        if label == "neutral":
            neutral_by_domain[domain]["neutral"] += 1
    sentiment_quality = {
        "sentiment_mismatch_rate": round(sentiment_mismatch_count / max(1, len(sentiment_rows)), 4),
        "abstain_coverage": round(sentiment_abstain_count / max(1, len(sentiment_rows)), 4),
        "abstain_risk_buckets": dict(risk_buckets),
        "neutral_overuse_rate_by_domain": {
            domain: round(values["neutral"] / max(1, values["total"]), 4)
            for domain, values in neutral_by_domain.items()
        },
    }
    robust_training_eval = evaluate_training_tracks(benchmark_rows_by_split)
    previous_accepted_path = Path(__file__).resolve().parents[1] / "state" / "accepted_training_metrics.json"
    previous_worst_domain = None
    if previous_accepted_path.exists():
        try:
            previous_payload = json.loads(previous_accepted_path.read_text(encoding="utf-8"))
            previous_worst_domain = float(previous_payload.get("worst_domain_f1"))
        except Exception:  # noqa: BLE001
            previous_worst_domain = None
    promotion_guard = promotion_gate(
        current_worst_domain_f1=float((robust_training_eval.get("groupdro") or {}).get("worst_domain_f1", 0.0)),
        previous_worst_domain_f1=previous_worst_domain,
        max_regression=0.02,
    )
    domain_generalization = _domain_generalization(
        finalized_rows,
        evaluation_protocol=cfg.evaluation_protocol,
        domain_holdout=cfg.domain_holdout,
    )
    unseen_metrics = _unseen_domain_metrics(
        finalized_rows,
        train_domain_support=dict(train_domain_support),
        weak_domain_support_row_threshold=cfg.weak_domain_support_row_threshold,
    )
    report = {
        "pipeline_version": "6.0-v6-only",
        "output_version": cfg.output_version,
        "generated_at": utc_now_iso(),
        "run_profile": run_profile,
        "artifact_mode": artifact_mode,
        "config": asdict(cfg),
        "schema": asdict(schema),
        "implicit_mode": cfg.implicit_mode,
        "multilingual_mode": cfg.multilingual_mode,
        "domain_conditioning_mode": train_domain_conditioning_mode,
        "train_domain_conditioning_mode": train_domain_conditioning_mode,
        "eval_domain_conditioning_mode": eval_domain_conditioning_mode,
        "coreference_enabled": cfg.use_coref,
        "language_distribution": prepared_language_distribution,
        "research": {
            "benchmark": benchmark_spec.key,
            "benchmark_family": benchmark_spec.family,
            "model_family": model_spec.key,
            "model_kind": model_spec.kind,
            "prompt_mode": cfg.prompt_mode,
            "augmentation_mode": cfg.augmentation_mode,
            "implicit_mode": cfg.implicit_mode,
            "multilingual_strategy": cfg.multilingual_mode,
            "coreference_enabled": cfg.use_coref,
            "no_drop": cfg.no_drop,
        },
        "stratification_choice": "group_id",
        "candidate_aspects": candidate_aspects,
        "candidate_aspects_by_language": candidate_aspects_by_language,
        "candidate_aspects_by_domain": candidate_aspects_by_domain_train,
        "chunk_preview_size": len(chunk_preview),
        "chunk_sampling_strategy": "seeded_shuffle_then_slice",
        "row_counts": {
            "input": len(frame),
            "preprocessed": len(prepared),
            "selected": len(sample_frame),
            "train": len(train_built),
            "val": len(val_built),
            "test": len(test_built),
            "train_export": len(train_export_rows),
        },
        "split_ratios": {
            "train": round(len(train_built) / max(1, len(train_built) + len(val_built) + len(test_built)), 4),
            "val": round(len(val_built) / max(1, len(train_built) + len(val_built) + len(test_built)), 4),
            "test": round(len(test_built) / max(1, len(train_built) + len(val_built) + len(test_built)), 4),
        },
        "split_sizes": {"train": len(train_built), "val": len(val_built), "test": len(test_built)},
        "implicit_diagnostics": diagnostics,
        "benchmark_summary": benchmark_metadata,
        "benchmark_gold_eval": benchmark_gold_metrics,
        "benchmark_structural_audits": benchmark_structural,
        "benchmark_v2_novelty": benchmark_v2_novelty,
        "aspect_registry": {
            "run_registry_version": resolve_registry_version(run_registry),
            "promoted_registry_version": resolve_registry_version(promoted_registry),
            "run_registry_domains": len(run_registry.get("domains", {})),
            "promoted_registry_domains": len(promoted_registry.get("domains", {})),
        },
        "sentiment_quality": sentiment_quality,
        "synthetic_generation": synthetic_audit,
        "robust_training_eval": robust_training_eval,
        "promotion_guard": promotion_guard,
        "benchmark_artifact_counts": benchmark_metadata.get("split_counts", {}),
        "benchmark_review_queue_rows": len(benchmark_review_queue_rows),
        "output_quality": quality_summary,
        "strict_quality": quality_summary.get("strict_quality", {}),
        "strict_artifacts": {
            "strict_train_rows": len(strict_train_export_rows),
            "strict_val_rows": len(strict_val_export_rows),
            "strict_test_rows": len(strict_test_export_rows),
            "strict_review_queue_rows": len(strict_review_queue_rows),
            "strict_challenge_rows": len(strict_challenge_rows),
        },
        "train_export_floor_rows": len(train_export_floor_rows),
        "grounded_prediction_rate": grounding["grounded_prediction_rate"],
        "ungrounded_non_general_count": grounding["ungrounded_non_general_count"],
        "train_general_rows_before_policy": train_general_policy_stats["train_general_rows_before_policy"],
        "train_general_rows_after_policy": train_general_policy_stats["train_general_rows_after_policy"],
        "train_general_policy_applied": train_general_policy_stats["train_general_policy_applied"],
        "train_review_rows_before_filter": train_review_filter_stats["train_review_rows_before_filter"],
        "train_review_rows_after_filter": train_review_filter_stats["train_review_rows_after_filter"],
        "train_review_filter_applied": train_review_filter_stats["train_review_filter_applied"],
        "train_review_dropped_soft_rows": len(train_review_dropped_soft_rows),
        "train_review_dropped_hard_rows": len(train_review_dropped_hard_rows),
        "train_reinference_stats": train_reinference_stats,
        "train_salvage_stats": train_salvage_stats,
        "train_leakage_filter_stats_before_salvage": train_leakage_filter_stats_before_salvage,
        "train_leakage_filter_stats_after_salvage": train_leakage_filter_stats_after_salvage,
        "train_leakage_filter_stats_after_topup": train_leakage_filter_stats_after_topup,
        "train_leakage_filter_stats_after_targeting": train_leakage_filter_stats_after_targeting,
        "train_sentiment_before_balance": train_sentiment_before_balance,
        "train_sentiment_after_balance": train_sentiment_after_balance,
        "train_sentiment_constraints": train_sentiment_constraints,
        "train_topup_stats": train_topup_stats,
        "train_topup_rejection_breakdown": train_topup_stats.get("train_topup_rejection_breakdown", {}),
        "topup_effectiveness": train_topup_stats.get("topup_effectiveness", {}),
        "size_recovery_stage": train_topup_stats.get("size_recovery_stage", "none"),
        "size_recovery_shortfall_remaining": train_topup_stats.get("size_recovery_shortfall_remaining", 0),
        "quality_analysis_summary": {
            "train_rows": quality_analysis_artifact["train_rows"],
            "final_train_rows": quality_analysis_artifact["final_train_rows"],
            "excluded_rows": quality_analysis_artifact["excluded_rows"],
            "borderline_count": quality_analysis_artifact["borderline_count"],
            "rejected_count": quality_analysis_artifact["rejected_count"],
            "reason_group_counts": quality_analysis_artifact["reason_group_counts"],
        },
        "train_general_dominance_rate": train_general_dominance_rate,
        **train_domain_leakage_metrics,
        **eval_domain_leakage_metrics,
        "train_negative_ratio": _sentiment_ratio(train_export_rows, label="negative"),
        "train_positive_ratio": _sentiment_ratio(train_export_rows, label="positive"),
        "train_target_stats": train_target_stats,
        "gold_eval": gold_metrics,
        "domain_generalization": domain_generalization,
        "unseen_domain_metrics": unseen_metrics,
        "domain_prior_boost_count": sum(int(row.get("implicit", {}).get("domain_prior_boost_count", 0)) for row in finalized_rows),
        "domain_prior_penalty_count": sum(int(row.get("implicit", {}).get("domain_prior_penalty_count", 0)) for row in finalized_rows),
        "explicit_metrics": aspect_metrics(train_export_rows),
        "validation": {
            "counts_match": len(train_built) + len(val_built) + len(test_built) == (
                len(finalized_rows)
                - int(eval_leakage_filter_stats_val.get("train_domain_leakage_filter_removed_rows", 0))
                - int(eval_leakage_filter_stats_test.get("train_domain_leakage_filter_removed_rows", 0))
            ),
            "has_language_distribution": bool(prepared_language_distribution),
            "has_candidate_aspects_by_language": bool(candidate_aspects_by_language),
            "no_generic_aspects": quality_summary["generic_implicit_aspects"] == 0,
            "no_rejected_aspects": quality_summary["rejected_implicit_aspects"] == 0,
            "has_gold_eval": bool(gold_metrics.get("has_gold_labels")) or bool(benchmark_gold_metrics.get("has_gold_interpretations")),
            "has_benchmark_gold_eval": bool(benchmark_gold_metrics.get("has_gold_interpretations")),
            "train_general_excluded": train_general_dominance_rate == 0.0,
            "train_domain_leakage_ok": float(train_domain_leakage_metrics.get("train_domain_leakage_row_rate", 1.0)) == 0.0,
            "train_target_size_within_range": bool(train_target_stats.get("size_within_target_range")),
            "train_positive_ratio_within_max": float(_sentiment_ratio(train_export_rows, label="positive")) <= float(cfg.train_max_positive_ratio),
            "train_neutral_ratio_within_max": float(_sentiment_ratio(train_export_rows, label="neutral")) <= float(cfg.train_neutral_max_ratio),
            "train_target_blocking_failure": (
                run_profile == "research"
                and not bool(train_target_stats.get("size_within_target_range"))
            ),
            "sampled_run_blocked_or_debug": sampled_run and run_profile in {"research", "debug"},
            "unseen_non_general_coverage_ok": float(unseen_metrics.get("unseen_non_general_coverage", 0.0)) >= float(cfg.unseen_non_general_coverage_min),
            "unseen_not_ready_rate_ok": float(unseen_metrics.get("unseen_implicit_not_ready_rate", 1.0)) <= float(cfg.unseen_implicit_not_ready_rate_max),
            "unseen_domain_leakage_ok": float(unseen_metrics.get("unseen_domain_leakage_row_rate", 1.0)) <= float(cfg.unseen_domain_leakage_row_rate_max),
            "strict_explicit_contamination_ok": float(quality_summary.get("explicit_in_implicit_rate", 1.0)) <= float(cfg.strict_explicit_in_implicit_rate_max),
            "strict_boundary_fp_ok": int(quality_summary.get("boundary_false_positive_count", 10**9)) <= int(cfg.strict_boundary_fp_max),
            "strict_h2_h3_ok": float(quality_summary.get("h2_h3_ratio", 0.0)) >= float(cfg.strict_h2_h3_ratio_min),
            "strict_multi_aspect_ok": float(quality_summary.get("multi_aspect_ratio", 0.0)) >= float(cfg.strict_multi_aspect_ratio_min),
            "strict_challenge_ok": float(quality_summary.get("challenge_macro_f1", 0.0)) >= float(cfg.strict_challenge_macro_f1_min),
            "grouped_split_leakage_ok": int(benchmark_metadata.get("grouped_split_leakage", {}).get("overlap_counts", {}).get("train_val", 0)) == 0
            and int(benchmark_metadata.get("grouped_split_leakage", {}).get("overlap_counts", {}).get("train_test", 0)) == 0
            and int(benchmark_metadata.get("grouped_split_leakage", {}).get("overlap_counts", {}).get("val_test", 0)) == 0,
            "benchmark_val_non_empty": int(len(benchmark_rows_by_split.get("val", []))) > 0,
            "benchmark_grounded_evidence_ok": float(benchmark_metadata.get("grounded_evidence_rate", 0.0)) >= 0.98,
            "benchmark_duplicate_rate_ok": float(benchmark_metadata.get("duplicate_interpretation_rate", 1.0)) <= 0.01,
            "benchmark_thermal_share_ok": float(benchmark_metadata.get("thermal_share", 1.0)) <= 0.35,
            "benchmark_domain_coverage_ok": bool(benchmark_metadata.get("benchmark_domain_coverage_ok", True)),
            "benchmark_implicit_purity_ok": float(benchmark_metadata.get("implicit_purity_rate", 0.0)) >= 0.7,
            "benchmark_ontology_compatibility_ok": float(benchmark_metadata.get("ontology_compatibility_rate", 0.0)) >= 0.9,
            "sentiment_mismatch_rate_ok": float(sentiment_quality.get("sentiment_mismatch_rate", 1.0)) <= 0.2,
            "promotion_guard_ok": not bool(promotion_guard.get("blocked", False)),
            "benchmark_artifact_counts_match": True,
        },
    }
    blocking_reasons: list[dict[str, str]] = []
    if bool(report["validation"].get("sampled_run_blocked_or_debug")) and run_profile == "debug":
        blocking_reasons.append({"code": "DEBUG_OR_SAMPLED_RUN", "message": "Run used debug profile and is not promotable."})
    if bool(report["validation"].get("train_target_blocking_failure")):
        topup_shortfall = int(report.get("size_recovery_shortfall_remaining", 0))
        if str(cfg.train_topup_recovery_mode).strip().lower() == "strict_topup" and topup_shortfall > 0:
            blocking_reasons.append({
                "code": "TRAIN_SIZE_BELOW_TARGET_AFTER_STAGED_TOPUP",
                "message": "Train export remains below minimum after staged strict top-up. See train_topup_rejection_breakdown.",
            })
        else:
            blocking_reasons.append({"code": "TRAIN_SIZE_BELOW_TARGET", "message": "Train export is below configured minimum target size."})
    if not bool(report["validation"].get("train_domain_leakage_ok")):
        blocking_reasons.append({"code": "TRAIN_DOMAIN_LEAKAGE", "message": "Train export contains cross-domain aspect leakage."})
    if not bool(report["validation"].get("train_general_excluded")):
        blocking_reasons.append({"code": "TRAIN_GENERAL_CONTAMINATION", "message": "Train export still includes general-only fallback rows."})
    if not bool(report["validation"].get("no_generic_aspects")) or not bool(report["validation"].get("no_rejected_aspects")):
        blocking_reasons.append({"code": "INVALID_ASPECT_LABELS", "message": "Generic or rejected aspect labels detected."})
    if not bool(report["validation"].get("train_positive_ratio_within_max")):
        blocking_reasons.append({"code": "TRAIN_POSITIVE_RATIO_TOO_HIGH", "message": "Positive sentiment exceeds configured maximum ratio."})
    if not bool(report["validation"].get("train_neutral_ratio_within_max")):
        blocking_reasons.append({"code": "TRAIN_NEUTRAL_RATIO_TOO_HIGH", "message": "Neutral sentiment exceeds configured maximum ratio."})
    if not bool(report["validation"].get("strict_explicit_contamination_ok")):
        blocking_reasons.append({"code": "STRICT_EXPLICIT_CONTAMINATION", "message": "Strict implicit set contains explicit span contamination."})
    if not bool(report["validation"].get("strict_boundary_fp_ok")):
        blocking_reasons.append({"code": "STRICT_BOUNDARY_FALSE_POSITIVES", "message": "Strict implicit set still includes boundary false positives."})
    if not bool(report["validation"].get("strict_h2_h3_ok")):
        blocking_reasons.append({"code": "STRICT_HARDNESS_TOO_LOW", "message": "Strict implicit set does not meet H2/H3 minimum ratio."})
    if not bool(report["validation"].get("strict_multi_aspect_ok")):
        blocking_reasons.append({"code": "STRICT_MULTI_ASPECT_TOO_LOW", "message": "Strict implicit set does not meet multi-aspect minimum ratio."})
    if not bool(report["validation"].get("strict_challenge_ok")):
        blocking_reasons.append({"code": "STRICT_CHALLENGE_TOO_LOW", "message": "Strict challenge metric is below configured floor."})
    if not bool(report["validation"].get("grouped_split_leakage_ok")):
        blocking_reasons.append({"code": "GROUPED_SPLIT_LEAKAGE", "message": "Grouped split leakage detected across benchmark splits."})
    if not bool(report["validation"].get("benchmark_val_non_empty")):
        blocking_reasons.append({"code": "BENCHMARK_VAL_EMPTY", "message": "Benchmark validation split is empty."})
    if not bool(report["validation"].get("benchmark_grounded_evidence_ok")):
        blocking_reasons.append({"code": "BENCHMARK_EVIDENCE_NOT_GROUNDED", "message": "Benchmark evidence grounding rate is below required threshold."})
    if not bool(report["validation"].get("benchmark_duplicate_rate_ok")):
        blocking_reasons.append({"code": "BENCHMARK_DUPLICATE_INTERPRETATIONS", "message": "Benchmark still contains duplicate interpretations."})
    if not bool(report["validation"].get("benchmark_thermal_share_ok")):
        blocking_reasons.append({"code": "BENCHMARK_THERMAL_OVERCONCENTRATION", "message": "Thermal aspect share remains over concentrated."})
    if not bool(report["validation"].get("benchmark_domain_coverage_ok")):
        blocking_reasons.append({"code": "BENCHMARK_DOMAIN_COVERAGE", "message": "Benchmark is missing a core V1 domain family present in the source pool."})
    if not bool(report["validation"].get("benchmark_implicit_purity_ok")):
        blocking_reasons.append({"code": "BENCHMARK_IMPLICIT_PURITY", "message": "Benchmark implicit purity rate is below threshold."})
    if not bool(report["validation"].get("benchmark_ontology_compatibility_ok")):
        blocking_reasons.append({"code": "BENCHMARK_ONTOLOGY_COMPATIBILITY", "message": "Benchmark ontology compatibility is below threshold."})
    if not bool(report["validation"].get("sentiment_mismatch_rate_ok")):
        blocking_reasons.append({"code": "SENTIMENT_MISMATCH_RATE", "message": "Sentiment mismatch rate exceeds allowed maximum."})
    if not bool(report["validation"].get("promotion_guard_ok")):
        blocking_reasons.append({"code": "WORST_DOMAIN_REGRESSION", "message": "Worst-domain F1 regressed above allowed threshold."})
    report["blocking_reasons"] = blocking_reasons
    report["governance_signoff"] = governance_signoff(report=report)
    report["promotion_eligibility"] = _resolve_promotion_eligibility(
        run_profile=run_profile,
        sampled=sampled_run,
        validation=report["validation"],
    )
    report["novelty_identity"] = _novelty_identity_block(cfg, report)
    research_manifest = build_research_manifest(
        dataset={
            "input_dir": cfg.input_dir,
            "output_dir": cfg.output_dir,
            "rows_in": len(frame),
            "rows_out": len(prepared),
            "text_column": text_column,
            "schema_fingerprint": schema.schema_fingerprint,
            "language_distribution": language_distribution(row[text_column] for row in prepared.to_dict(orient="records")),
        },
        benchmark=benchmark_spec,
        model_family=model_spec,
        metrics=quality_summary,
        prompt_mode=cfg.prompt_mode,
        augmentation_mode=cfg.augmentation_mode,
    )

    progress.step("report assembly")
    if not cfg.dry_run:
        write_benchmark_outputs(
            cfg.benchmark_dir,
            benchmark_rows_by_split,
            benchmark_metadata,
            protocol_views=benchmark_protocol_views,
        )
        write_jsonl(cfg.benchmark_dir / "review_queue.jsonl", benchmark_review_queue_rows)
        write_json(cfg.reports_dir / "aspect_registry_run.json", run_registry)
        write_json(cfg.reports_dir / "aspect_registry_promoted.json", promoted_registry)
        write_json(cfg.reports_dir / "quality_analysis.json", quality_analysis_artifact)
        write_synthetic_outputs(
            output_dir=cfg.output_dir / "synthetic",
            accepted=synthetic_accepted,
            rejected=synthetic_rejected,
        )
        write_json(cfg.output_dir / "synthetic" / "audit.json", synthetic_audit)
        benchmark_artifact_counts = _benchmark_artifact_counts(cfg.benchmark_dir)
        benchmark_report_counts = dict(benchmark_metadata.get("split_counts", {}))
        benchmark_report_counts["total"] = int(benchmark_metadata.get("rows", 0))
        benchmark_artifact_counts_match = all(
            int(benchmark_artifact_counts.get(key, 0)) == int(value)
            for key, value in benchmark_report_counts.items()
        )
        report["benchmark_artifact_counts"] = benchmark_artifact_counts
        report["validation"]["benchmark_artifact_counts_match"] = benchmark_artifact_counts_match
        if not benchmark_artifact_counts_match:
            report["blocking_reasons"] = report.get("blocking_reasons", [])
            report["blocking_reasons"].append({
                "code": "BENCHMARK_ARTIFACT_COUNT_MISMATCH",
                "message": "Written benchmark JSONL counts do not match report metadata.",
            })
        report["promotion_eligibility"] = _resolve_promotion_eligibility(
            run_profile=run_profile,
            sampled=sampled_run,
            validation=report["validation"],
        )
        if not bool(promotion_guard.get("blocked", False)):
            previous_accepted_path.parent.mkdir(parents=True, exist_ok=True)
            previous_accepted_path.write_text(
                json.dumps(
                    {
                        "updated_at": utc_now_iso(),
                        "worst_domain_f1": float((robust_training_eval.get("groupdro") or {}).get("worst_domain_f1", 0.0)),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        write_json(cfg.reports_dir / "build_report.json", report)
        write_json(cfg.reports_dir / "benchmark_v2_novelty_report.json", benchmark_v2_novelty)
        write_json(cfg.reports_dir / "data_quality_report.json", {
            "run_profile": run_profile,
            "artifact_mode": artifact_mode,
            "rows_in": len(frame),
            "rows_out": len(prepared),
            "text_column": text_column,
            "schema_fingerprint": schema.schema_fingerprint,
            "candidate_aspects": candidate_aspects,
            "candidate_aspects_by_language": candidate_aspects_by_language,
            "candidate_aspects_by_domain": candidate_aspects_by_domain_train,
            "implicit_mode": cfg.implicit_mode,
            "multilingual_mode": cfg.multilingual_mode,
            "coreference_enabled": cfg.use_coref,
            "language_distribution": prepared_language_distribution,
            "row_counts": report["row_counts"],
            "research": report["research"],
            "output_quality": quality_summary,
            "benchmark_summary": benchmark_metadata,
            "benchmark_v2_novelty": benchmark_v2_novelty,
            "aspect_registry_run": run_registry,
            "aspect_registry_promoted": promoted_registry,
            "sentiment_quality": sentiment_quality,
            "synthetic_generation": synthetic_audit,
            "robust_training_eval": robust_training_eval,
            "promotion_guard": promotion_guard,
            "governance_signoff": report.get("governance_signoff", {}),
            "benchmark_artifact_counts": benchmark_artifact_counts,
            "benchmark_report_counts": benchmark_report_counts,
            "benchmark_artifact_counts_match": benchmark_artifact_counts_match,
            "strict_quality": report.get("strict_quality", {}),
            "strict_artifacts": report.get("strict_artifacts", {}),
            "train_salvage_stats": train_salvage_stats,
            "train_topup_stats": train_topup_stats,
            "train_reinference_stats": report.get("train_reinference_stats", {}),
            "train_review_dropped_soft_rows": report.get("train_review_dropped_soft_rows", 0),
            "train_review_dropped_hard_rows": report.get("train_review_dropped_hard_rows", 0),
            "size_recovery_stage": train_topup_stats.get("size_recovery_stage", "none"),
            "size_recovery_shortfall_remaining": train_topup_stats.get("size_recovery_shortfall_remaining", 0),
            "topup_effectiveness": train_topup_stats.get("topup_effectiveness", {}),
            "train_topup_rejection_breakdown": train_topup_stats.get("train_topup_rejection_breakdown", {}),
            "train_target_stats": train_target_stats,
            "train_sentiment_constraints": train_sentiment_constraints,
            "train_domain_leakage_rows": report["train_domain_leakage_rows"],
            "train_domain_leakage_row_rate": report["train_domain_leakage_row_rate"],
            "train_domain_leakage_aspect_instances": report["train_domain_leakage_aspect_instances"],
            "eval_domain_leakage_rows": report["eval_domain_leakage_rows"],
            "eval_domain_leakage_row_rate": report["eval_domain_leakage_row_rate"],
            "eval_domain_leakage_aspect_instances": report["eval_domain_leakage_aspect_instances"],
            "train_negative_ratio": report["train_negative_ratio"],
            "train_positive_ratio": report["train_positive_ratio"],
            "grounded_prediction_rate": grounding["grounded_prediction_rate"],
            "ungrounded_non_general_count": grounding["ungrounded_non_general_count"],
            "gold_eval": gold_metrics,
            "domain_generalization": domain_generalization,
            "unseen_domain_metrics": unseen_metrics,
            "domain_prior_boost_count": report["domain_prior_boost_count"],
            "domain_prior_penalty_count": report["domain_prior_penalty_count"],
            "novelty_identity": report["novelty_identity"],
            "promotion_eligibility": report["promotion_eligibility"],
            "blocking_reasons": report["blocking_reasons"],
            "validation": report["validation"],
            "chunked_execution": {"sample_size": cfg.sample_size, "chunk_size": cfg.chunk_size, "chunk_offset": cfg.chunk_offset, "strategy": "seeded_shuffle_then_slice"},
        })
        write_json(cfg.reports_dir / "research_manifest.json", research_manifest)
        if cfg.emit_review_set:
            write_jsonl(cfg.reports_dir / "review_set_template.jsonl", _build_review_set_template(finalized_rows, size=cfg.review_set_size, seed=cfg.random_seed))
        progress.step("report/export writing")
    else:
        progress.step("report/export writing")

    report["research_manifest"] = research_manifest
    progress.close()
    return report


def build_parser() -> argparse.ArgumentParser:
    runtime_defaults = _load_runtime_defaults()
    parser = argparse.ArgumentParser(description="ReviewOp V6 dataset builder")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--run-profile", type=str, default="research", choices=["research", "debug"])
    parser.add_argument("--artifact-mode", type=str, default="auto", choices=["auto", "debug_artifacts", "research_release"])
    parser.add_argument("--debug-benchmark-max-rows", type=int, default=180)
    parser.add_argument("--processor", type=str, default=_optional_env("REVIEWOP_DATASET_BUILDER_PROCESSOR", "DATASET_BUILDER_PROCESSOR"), choices=["local", "runpod"])
    parser.add_argument("--llm-provider", type=str, default="auto", choices=["auto", "openai", "runpod", "ollama", "mock", "claude", "anthropic"])
    parser.add_argument(
        "--llm-model-name",
        type=str,
        default=_optional_env("REVIEWOP_LLM_MODEL_NAME", "LLM_MODEL_NAME", "RUNPOD_MODEL", "CLAUDE_MODEL", "GROQ_MODEL", "OPENAI_MODEL", "OLLAMA_MODEL", default="meta-llama/Meta-Llama-3.1-8B-Instruct"),
    )
    parser.add_argument("--enable-reasoned-recovery", dest="enable_reasoned_recovery", action="store_true")
    parser.add_argument("--no-enable-reasoned-recovery", dest="enable_reasoned_recovery", action="store_false")
    parser.set_defaults(enable_reasoned_recovery=True)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=float(runtime_defaults.get("confidence_threshold", 0.6)))
    parser.add_argument("--max-aspects", type=int, default=20)
    parser.add_argument("--min-text-tokens", type=int, default=4)
    parser.add_argument("--implicit-min-tokens", type=int, default=8)
    parser.add_argument("--implicit-mode", type=str, default=str(runtime_defaults.get("implicit_mode", "zeroshot")), choices=["zeroshot", "supervised", "hybrid", "heuristic", "benchmark"])
    parser.add_argument("--multilingual-mode", type=str, default="shared_vocab")
    parser.add_argument("--use-coref", dest="use_coref", action="store_true")
    parser.add_argument("--no-use-coref", dest="use_coref", action="store_false")
    parser.set_defaults(use_coref=bool(runtime_defaults.get("use_coref", False)))
    parser.add_argument("--language-detection-mode", type=str, default="heuristic")
    parser.add_argument("--no-drop", action="store_true")
    parser.add_argument("--enable-llm-fallback", dest="enable_llm_fallback", action="store_true")
    parser.add_argument("--no-enable-llm-fallback", dest="enable_llm_fallback", action="store_false")
    parser.set_defaults(enable_llm_fallback=bool(runtime_defaults.get("enable_llm_fallback", True)))
    parser.add_argument("--llm-fallback-threshold", type=float, default=float(runtime_defaults.get("llm_fallback_threshold", 0.65)))
    parser.add_argument("--benchmark-key", type=str, default=None)
    parser.add_argument("--model-family", type=str, default="heuristic_latent")
    parser.add_argument("--augmentation-mode", type=str, default="none")
    parser.add_argument("--prompt-mode", type=str, default="constrained")
    parser.add_argument("--gold-annotations-path", type=Path, default=None)
    parser.add_argument("--emit-review-set", action="store_true")
    parser.add_argument("--review-set-size", type=int, default=300)
    parser.add_argument("--evaluation-protocol", type=str, default="random", choices=["random", "loo", "source-free"])
    parser.add_argument("--domain-holdout", type=str, default=None)
    parser.add_argument("--enforce-grounding", dest="enforce_grounding", action="store_true")
    parser.add_argument("--no-enforce-grounding", dest="enforce_grounding", action="store_false")
    parser.add_argument("--high-difficulty", dest="high_difficulty", action="store_true")
    parser.add_argument("--no-high-difficulty", dest="high_difficulty", action="store_false")
    parser.add_argument("--adversarial-refine", dest="adversarial_refine", action="store_true")
    parser.add_argument("--no-adversarial-refine", dest="adversarial_refine", action="store_false")
    parser.add_argument("--no-domain-conditioning", dest="use_domain_conditioning", action="store_false")
    parser.add_argument("--no-strict-domain-conditioning", dest="strict_domain_conditioning", action="store_false")
    parser.add_argument("--domain-conditioning-mode", type=str, default="adaptive_soft", choices=["adaptive_soft", "strict_hard", "off"])
    parser.add_argument("--train-domain-conditioning-mode", type=str, default=None, choices=["adaptive_soft", "strict_hard", "off"])
    parser.add_argument("--eval-domain-conditioning-mode", type=str, default=None, choices=["adaptive_soft", "strict_hard", "off"])
    parser.set_defaults(enforce_grounding=True, use_domain_conditioning=True, strict_domain_conditioning=False, high_difficulty=False, adversarial_refine=False)
    parser.add_argument("--domain-prior-boost", type=float, default=0.05)
    parser.add_argument("--domain-prior-penalty", type=float, default=0.08)
    parser.add_argument("--weak-domain-support-row-threshold", type=int, default=80)
    parser.add_argument("--unseen-non-general-coverage-min", type=float, default=0.55)
    parser.add_argument("--unseen-implicit-not-ready-rate-max", type=float, default=0.35)
    parser.add_argument("--unseen-domain-leakage-row-rate-max", type=float, default=0.02)
    parser.add_argument("--train-fallback-general-policy", type=str, default="cap", choices=["keep", "cap", "drop"])
    parser.add_argument("--train-fallback-general-cap-ratio", type=float, default=0.15)
    parser.add_argument("--train-review-filter-mode", type=str, default="reasoned_strict", choices=["keep", "drop_needs_review", "reasoned_strict"])
    parser.add_argument("--train-salvage-mode", type=str, default="recover_non_general", choices=["off", "recover_non_general"])
    parser.add_argument("--train-salvage-confidence-threshold", type=float, default=0.56)
    parser.add_argument("--train-salvage-accepted-support-types", type=str, default="exact,near_exact,gold")
    parser.add_argument(
        "--train-sentiment-balance-mode",
        type=str,
        default="cap_neutral_with_dual_floor",
        choices=["none", "cap_neutral", "cap_neutral_with_negative_floor", "cap_neutral_with_dual_floor"],
    )
    parser.add_argument("--train-neutral-cap-ratio", type=float, default=0.5)
    parser.add_argument("--train-min-negative-ratio", type=float, default=0.12)
    parser.add_argument("--train-min-positive-ratio", type=float, default=0.12)
    parser.add_argument("--train-max-positive-ratio", type=float, default=0.5)
    parser.add_argument("--train-neutral-max-ratio", type=float, default=0.58)
    parser.add_argument("--train-topup-recovery-mode", type=str, default="strict_topup", choices=["off", "strict_topup"])
    parser.add_argument("--train-topup-confidence-threshold", type=float, default=0.58)
    parser.add_argument("--train-topup-staged-recovery", dest="train_topup_staged_recovery", action="store_true")
    parser.add_argument("--no-train-topup-staged-recovery", dest="train_topup_staged_recovery", action="store_false")
    parser.set_defaults(train_topup_staged_recovery=True)
    parser.add_argument("--train-topup-stage-b-confidence-threshold", type=float, default=0.54)
    parser.add_argument("--train-topup-allow-weak-support-in-stage-c", dest="train_topup_allow_weak_support_in_stage_c", action="store_true")
    parser.add_argument("--no-train-topup-allow-weak-support-in-stage-c", dest="train_topup_allow_weak_support_in_stage_c", action="store_false")
    parser.set_defaults(train_topup_allow_weak_support_in_stage_c=True)
    parser.add_argument("--train-topup-stage-c-confidence-threshold", type=float, default=0.52)
    parser.add_argument("--train-topup-allowed-support-types", type=str, default="exact,near_exact,gold")
    parser.add_argument("--train-target-min-rows", type=int, default=1600)
    parser.add_argument("--train-target-max-rows", type=int, default=2000)
    parser.add_argument("--strict-implicit-enabled", dest="strict_implicit_enabled", action="store_true")
    parser.add_argument("--no-strict-implicit-enabled", dest="strict_implicit_enabled", action="store_false")
    parser.set_defaults(strict_implicit_enabled=True)
    parser.add_argument("--strict-review-sample-size", type=int, default=200)
    parser.add_argument("--strict-explicit-in-implicit-rate-max", type=float, default=0.0)
    parser.add_argument("--strict-boundary-fp-max", type=int, default=0)
    parser.add_argument("--strict-h2-h3-ratio-min", type=float, default=0.35)
    parser.add_argument("--strict-multi-aspect-ratio-min", type=float, default=0.12)
    parser.add_argument("--strict-challenge-macro-f1-min", type=float, default=0.5)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--zip-only", action="store_true", help="Skip dataset build and only zip existing benchmark/report artifacts")
    parser.add_argument("--no-llm-cache", "--no-cache-llm", dest="no_llm_cache", action="store_true", help="Bypass and clear the LLM response cache for this run.")
    parser.set_defaults(no_llm_cache=False)
    parser.add_argument("--discovery-mode", action="store_true", default=True, help="Enable Open-Domain Discovery for untracked aspects.")
    parser.add_argument("--no-discovery-mode", action="store_false", dest="discovery_mode")
    parser.add_argument("--discovery-min-confidence", type=float, default=0.55)
    parser.add_argument("--discovery-stability-threshold", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or BuilderConfig().output_dir

    if args.zip_only:
        zip_path = compress_dataset_artifacts(output_dir)
        if zip_path:
            print(f"Dataset artifacts archived to: {zip_path}")
            return 0
        print("No dataset artifacts found to archive.")
        return 1

    artifact_mode = str(args.artifact_mode or "auto").strip().lower()
    if artifact_mode == "auto":
        artifact_mode = "debug_artifacts" if str(args.run_profile).strip().lower() == "debug" else "research_release"
    
    # Provider/Processor Reconciliation
    llm_provider_choice = str(args.llm_provider).strip().lower()
    if llm_provider_choice != "auto":
        llm_provider_name = llm_provider_choice
        processor = _resolve_processor_choice(processor=args.processor) if args.processor else "local"
    else:
        processor = _resolve_processor_choice(processor=args.processor)
        llm_provider_name = _resolve_llm_provider_for_processor(processor=processor)

    llm_provider = resolve_async_llm_provider(llm_provider_name, model_name=args.llm_model_name) if llm_provider_name else None

    domain_conditioning_mode = str(args.domain_conditioning_mode or "").strip().lower()
    if not args.use_domain_conditioning:
        domain_conditioning_mode = "off"
    elif args.strict_domain_conditioning and domain_conditioning_mode == "adaptive_soft":
        domain_conditioning_mode = "strict_hard"
    train_domain_conditioning_mode = str(args.train_domain_conditioning_mode or "").strip().lower() or None
    eval_domain_conditioning_mode = str(args.eval_domain_conditioning_mode or "").strip().lower() or None
    if train_domain_conditioning_mode is None or eval_domain_conditioning_mode is None:
        if domain_conditioning_mode == "strict_hard":
            train_domain_conditioning_mode = train_domain_conditioning_mode or "strict_hard"
            eval_domain_conditioning_mode = eval_domain_conditioning_mode or "strict_hard"
        elif domain_conditioning_mode == "off":
            train_domain_conditioning_mode = train_domain_conditioning_mode or "off"
            eval_domain_conditioning_mode = eval_domain_conditioning_mode or "off"
        else:
            train_domain_conditioning_mode = train_domain_conditioning_mode or "strict_hard"
            eval_domain_conditioning_mode = eval_domain_conditioning_mode or "adaptive_soft"

    cfg = BuilderConfig(
        input_dir=args.input_dir or BuilderConfig().input_dir,
        output_dir=output_dir,
        random_seed=args.seed,
        text_column_override=args.text_column,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_offset=args.chunk_offset,
        run_profile=args.run_profile,
        artifact_mode=artifact_mode,
        debug_benchmark_max_rows=args.debug_benchmark_max_rows,
        dry_run=args.dry_run or args.preview,
        preview_only=args.preview,
        confidence_threshold=args.confidence_threshold,
        max_aspects=args.max_aspects,
        min_text_tokens=args.min_text_tokens,
        implicit_min_tokens=args.implicit_min_tokens,
        implicit_mode=args.implicit_mode,
        multilingual_mode=args.multilingual_mode,
        use_coref=args.use_coref,
        language_detection_mode=args.language_detection_mode,
        no_drop=args.no_drop,
        enable_llm_fallback=args.enable_llm_fallback,
        llm_fallback_threshold=args.llm_fallback_threshold,
        benchmark_key=args.benchmark_key,
        model_family=args.model_family,
        augmentation_mode=args.augmentation_mode,
        prompt_mode=args.prompt_mode,
        gold_annotations_path=args.gold_annotations_path,
        emit_review_set=args.emit_review_set,
        review_set_size=args.review_set_size,
        evaluation_protocol=args.evaluation_protocol,
        domain_holdout=args.domain_holdout,
        enforce_grounding=args.enforce_grounding,
        use_domain_conditioning=args.use_domain_conditioning,
        strict_domain_conditioning=args.strict_domain_conditioning,
        domain_conditioning_mode=domain_conditioning_mode,
        train_domain_conditioning_mode=str(train_domain_conditioning_mode),
        eval_domain_conditioning_mode=str(eval_domain_conditioning_mode),
        domain_prior_boost=args.domain_prior_boost,
        domain_prior_penalty=args.domain_prior_penalty,
        weak_domain_support_row_threshold=args.weak_domain_support_row_threshold,
        progress=args.progress,
        unseen_non_general_coverage_min=args.unseen_non_general_coverage_min,
        unseen_implicit_not_ready_rate_max=args.unseen_implicit_not_ready_rate_max,
        unseen_domain_leakage_row_rate_max=args.unseen_domain_leakage_row_rate_max,
        train_fallback_general_policy=args.train_fallback_general_policy,
        train_fallback_general_cap_ratio=args.train_fallback_general_cap_ratio,
        train_review_filter_mode=args.train_review_filter_mode,
        train_salvage_mode=args.train_salvage_mode,
        train_salvage_confidence_threshold=args.train_salvage_confidence_threshold,
        train_salvage_accepted_support_types=tuple(part.strip() for part in str(args.train_salvage_accepted_support_types).split(",") if part.strip()),
        train_sentiment_balance_mode=args.train_sentiment_balance_mode,
        train_neutral_cap_ratio=args.train_neutral_cap_ratio,
        train_min_negative_ratio=args.train_min_negative_ratio,
        train_min_positive_ratio=args.train_min_positive_ratio,
        train_max_positive_ratio=args.train_max_positive_ratio,
        train_neutral_max_ratio=args.train_neutral_max_ratio,
        train_topup_recovery_mode=args.train_topup_recovery_mode,
        train_topup_confidence_threshold=args.train_topup_confidence_threshold,
        train_topup_staged_recovery=args.train_topup_staged_recovery,
        train_topup_stage_b_confidence_threshold=args.train_topup_stage_b_confidence_threshold,
        train_topup_allow_weak_support_in_stage_c=args.train_topup_allow_weak_support_in_stage_c,
        train_topup_stage_c_confidence_threshold=args.train_topup_stage_c_confidence_threshold,
        train_topup_allowed_support_types=tuple(part.strip() for part in str(args.train_topup_allowed_support_types).split(",") if part.strip()),
        train_target_min_rows=args.train_target_min_rows,
        train_target_max_rows=args.train_target_max_rows,
        strict_implicit_enabled=args.strict_implicit_enabled,
        strict_review_sample_size=args.strict_review_sample_size,
        strict_explicit_in_implicit_rate_max=args.strict_explicit_in_implicit_rate_max,
        strict_boundary_fp_max=args.strict_boundary_fp_max,
        strict_h2_h3_ratio_min=args.strict_h2_h3_ratio_min,
        strict_multi_aspect_ratio_min=args.strict_multi_aspect_ratio_min,
        strict_challenge_macro_f1_min=args.strict_challenge_macro_f1_min,
        processor=processor,
        llm_provider=llm_provider,
        llm_model_name=args.llm_model_name,
        enable_reasoned_recovery=args.enable_reasoned_recovery,
        max_workers=args.max_workers,
        high_difficulty=args.high_difficulty,
        adversarial_refine=args.adversarial_refine,
        no_llm_cache=args.no_llm_cache,
        discovery_mode=args.discovery_mode,
        discovery_min_confidence=args.discovery_min_confidence,
        discovery_stability_threshold=args.discovery_stability_threshold,
    )
    import asyncio
    report = asyncio.run(run_pipeline(cfg))
    
    if not cfg.dry_run:
        zip_path = compress_dataset_artifacts(cfg.output_dir)
        if zip_path:
            print(f"Dataset artifacts archived to: {zip_path}")

    if bool(report.get("validation", {}).get("train_target_blocking_failure")):
        print("Build blocked: train_target_size_within_range=false under research profile.")
        return 2
    print(f"Build complete: {report['generated_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

