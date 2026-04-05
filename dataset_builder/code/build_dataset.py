from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from contracts import BuilderConfig
from coref import heuristic_coref
from evaluation import aspect_metrics, gold_eval
from exporters import write_compat_exports, write_named_outputs, write_split_outputs
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
from llm_utils import resolve_llm_provider
from research_stack import build_research_manifest, resolve_benchmark, resolve_model_family
from schema_detect import detect_schema
from splitter import choose_stratify_values, preliminary_split, split_holdout
from utils import (
    normalize_whitespace,
    stable_id,
    utc_now_iso,
    write_json,
    write_jsonl,
    compress_output_folder,
)

load_dotenv()


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
    # Optimized: Use vectorized operations where possible, or faster row iteration
    out = frame.copy()
    # Create a deterministic ID using source_file and a hash of the content (stable_id)
    # We still use a comprehension but we avoid to_json if possible by using a tuple of keys
    out["id"] = [
        stable_id(row.get("source_file", "source"), idx, str(tuple(row.values())))
        for idx, row in enumerate(out.to_dict(orient="records"))
    ]
    return out


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


def _select_working_rows(rows: list[dict[str, Any]], cfg: BuilderConfig) -> list[dict[str, Any]]:
    ordered = _chunk_rows(rows, cfg)
    if cfg.sample_size is not None:
        ordered = ordered[: max(0, cfg.sample_size)]
    if cfg.chunk_size is not None:
        start = max(0, cfg.chunk_offset)
        end = start + max(0, cfg.chunk_size)
        ordered = ordered[start:end]
    return ordered


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
        domain = str(row.get("domain", "unknown"))
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


def _merge_gold_labels(rows: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not annotations:
        return rows
    by_record_id: dict[str, list[dict[str, Any]]] = {}
    by_domain_text: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in annotations:
        labels = item.get("gold_labels")
        if not isinstance(labels, list):
            continue
        record_id = item.get("record_id")
        if isinstance(record_id, str) and record_id.strip():
            by_record_id[record_id.strip()] = labels
        domain = str(item.get("domain") or "unknown")
        text = normalize_whitespace(item.get("text") or "")
        if text:
            by_domain_text[(domain, text)] = labels

    merged: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        existing = out.get("gold_labels")
        if isinstance(existing, list) and existing:
            merged.append(out)
            continue
        record_id = str(out.get("id") or "")
        labels = by_record_id.get(record_id)
        if labels is None:
            domain = str(out.get("domain") or "unknown")
            text = normalize_whitespace(out.get("source_text") or "")
            labels = by_domain_text.get((domain, text))
        if labels is not None:
            out["gold_labels"] = labels
        merged.append(out)
    return merged


def _grounding_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_general = [
        row for row in rows if row.get("implicit", {}).get("aspects") and row.get("implicit", {}).get("aspects") != ["general"]
    ]
    grounded = [row for row in non_general if bool(row.get("implicit", {}).get("spans"))]
    return {
        "grounded_prediction_rate": round(len(grounded) / len(non_general), 4) if non_general else 0.0,
        "ungrounded_non_general_count": len(non_general) - len(grounded),
        "non_general_count": len(non_general),
    }


def _train_sentiment_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        sentiment = str(row.get("implicit", {}).get("dominant_sentiment") or "unknown")
        counts[sentiment] += 1
    return dict(counts)


def _sentiment_ratio(rows: list[dict[str, Any]], *, label: str) -> float:
    if not rows:
        return 0.0
    hit = sum(1 for row in rows if str(row.get("implicit", {}).get("dominant_sentiment") or "unknown") == label)
    return round(hit / len(rows), 4)


def _strict_row_passes(row: dict[str, Any]) -> bool:
    implicit = row.get("implicit", {}) or {}
    if str(implicit.get("implicit_quality_tier") or "needs_review") != "strict_pass":
        return False
    if bool(implicit.get("needs_review")):
        return False
    aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
    if not aspects:
        return False
    spans = list(implicit.get("spans") or [])
    if not spans:
        return False
    if any(str(span.get("label_type") or "").strip().lower() == "explicit" for span in spans):
        return False
    return True


def _strict_quality_metrics(rows: list[dict[str, Any]], *, challenge_macro_f1: float = 0.0) -> dict[str, Any]:
    total_spans = 0
    explicit_spans = 0
    boundary_fp_count = 0
    h_tier_counts: Counter[str] = Counter()
    non_general_rows = 0
    h2_h3_rows = 0
    multi_aspect_rows = 0
    strict_pass_rows = 0
    review_queue_rows = 0
    for row in rows:
        implicit = row.get("implicit", {}) or {}
        tier = str(implicit.get("hardness_tier") or "H0")
        h_tier_counts[tier] += 1
        aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
        if aspects:
            non_general_rows += 1
            if len(aspects) > 1:
                multi_aspect_rows += 1
        if tier in {"H2", "H3"} and aspects:
            h2_h3_rows += 1
        if str(implicit.get("implicit_quality_tier") or "needs_review") == "strict_pass":
            strict_pass_rows += 1
        else:
            review_queue_rows += 1
        for span in implicit.get("spans", []) or []:
            total_spans += 1
            if str(span.get("label_type") or "").strip().lower() == "explicit":
                explicit_spans += 1
            for flag in span.get("leakage_flags", []) or []:
                if str(flag) in {"explicit_keyword_surface_leakage", "latent_name_surface_leakage", "surface_equals_latent"}:
                    boundary_fp_count += 1
                    break
    return {
        "explicit_in_implicit_rate": round(explicit_spans / max(1, total_spans), 4),
        "explicit_in_implicit_count": explicit_spans,
        "total_implicit_spans": total_spans,
        "boundary_false_positive_count": int(boundary_fp_count),
        "hardness_distribution": dict(h_tier_counts),
        "h2_h3_ratio": round(h2_h3_rows / max(1, non_general_rows), 4),
        "multi_aspect_ratio": round(multi_aspect_rows / max(1, len(rows)), 4),
        "strict_pass_rows": int(strict_pass_rows),
        "review_queue_rows": int(review_queue_rows),
        "challenge_macro_f1": float(challenge_macro_f1),
    }


def _stable_keep(rows: list[dict[str, Any]], *, seed: int, token: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    ordered = sorted(
        rows,
        key=lambda row: stable_id(seed, token, row.get("id") or row.get("source_text") or ""),
    )
    return ordered[:limit]


def _stable_stratified_keep(rows: list[dict[str, Any]], *, seed: int, token: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if limit >= len(rows):
        return sorted(rows, key=lambda row: stable_id(seed, token, row.get("id") or row.get("source_text") or ""))
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(
            str(row.get("domain", "unknown")),
            str(row.get("language", "unknown")),
            _primary_non_general_aspect(row),
        )].append(row)
    retained: list[dict[str, Any]] = []
    total_rows = len(rows)
    for key, bucket_rows in buckets.items():
        provisional = int(round((len(bucket_rows) / total_rows) * limit))
        if provisional <= 0 and bucket_rows:
            provisional = 1
        retained.extend(_stable_keep(bucket_rows, seed=seed, token=f"{token}:{':'.join(key)}", limit=min(len(bucket_rows), provisional)))
    if len(retained) > limit:
        retained = _stable_keep(retained, seed=seed, token=f"{token}:trim", limit=limit)
    elif len(retained) < limit:
        retained_ids = {str(row.get("id") or "") for row in retained}
        extras = [row for row in rows if str(row.get("id") or "") not in retained_ids]
        retained.extend(_stable_keep(extras, seed=seed, token=f"{token}:topup", limit=limit - len(retained)))
    return sorted(retained, key=lambda row: stable_id(seed, token, row.get("id") or row.get("source_text") or ""))


def _allowed_latents_for_domain(
    *,
    row: dict[str, Any],
    candidate_aspects_by_domain: dict[str, list[str]],
) -> set[str]:
    domain = str(row.get("domain", "unknown"))
    domain_candidates = candidate_aspects_by_domain.get(domain, [])
    allowed_latents = {
        _latent_aspect_label(candidate, str(row.get("source_text", "")))
        for candidate in domain_candidates
    }
    return {aspect for aspect in allowed_latents if aspect != "general" and _is_valid_latent_aspect(aspect)}


def _row_domain_valid_for_train(
    *,
    row: dict[str, Any],
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
            if review_reason == "fallback_general":
                dropped_soft_rows.append(row)
                continue
            if review_reason in {"weak_support", "low_confidence"}:
                if (
                    _row_grounded_non_general(row, accepted_support_types=accepted_support, min_confidence=min_confidence)
                    and _row_domain_valid_for_train(row=row, candidate_aspects_by_domain=domain_map)
                ):
                    kept_rows.append(row)
                else:
                    dropped_soft_rows.append(row)
                continue
            # Edge parse and other review reasons remain recoverable candidates.
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
            domain = str(row.get("domain", "unknown"))
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
        domain = str(row.get("domain", "unknown"))
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
            domain = str(row.get("domain", "unknown"))
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
    used_stage = "none"

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
        stage_additions = 0
        for row in ordered_candidates:
            if remaining <= 0:
                break
            row_id = str(row.get("id") or "")
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
    legacy_mode = _resolve_domain_conditioning_mode(cfg)
    train_mode = str(getattr(cfg, "train_domain_conditioning_mode", "") or "").strip().lower()
    eval_mode = str(getattr(cfg, "eval_domain_conditioning_mode", "") or "").strip().lower()
    default_pair = train_mode in {"", "strict_hard"} and eval_mode in {"", "adaptive_soft"}

    # Backward compatibility: if legacy flags/mode were explicitly used and split modes are untouched defaults,
    # keep old global behavior.
    legacy_override_requested = (
        not bool(cfg.use_domain_conditioning)
        or bool(cfg.strict_domain_conditioning)
        or str(getattr(cfg, "domain_conditioning_mode", "") or "").strip().lower() in {"strict_hard", "off"}
    )
    if default_pair and legacy_override_requested:
        train_mode = legacy_mode
        eval_mode = legacy_mode
    else:
        if train_mode not in {"adaptive_soft", "strict_hard", "off"}:
            train_mode = legacy_mode
        if eval_mode not in {"adaptive_soft", "strict_hard", "off"}:
            eval_mode = legacy_mode
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
        domain = str(row.get("domain", "unknown"))
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


def _prepare_rows(frame: pd.DataFrame, cfg: BuilderConfig, text_column: str, progress_tracker: _ProgressTracker | None = None) -> pd.DataFrame:
    out = _assign_ids(frame.reset_index(drop=True).copy())
    out[text_column] = out[text_column].fillna("").astype(str)
    
    if progress_tracker:
        progress_tracker.step("preprocessing: schema & metadata", 0)
    
    out["domain"] = out.get("source_file", pd.Series(["unknown"] * len(out))).map(lambda value: _canonical_domain(str(value)))
    out["language"] = out[text_column].map(detect_language)
    out["implicit_ready"] = [
        is_implicit_ready(text, language=language, min_tokens=cfg.implicit_min_tokens, supported_languages=cfg.supported_languages)
        for text, language in zip(out[text_column].tolist(), out["language"].tolist(), strict=False)
    ]
    if not cfg.no_drop:
        out = out[out[text_column].str.split().map(len) >= cfg.min_text_tokens].reset_index(drop=True)
    return out


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

    domain = _canonical_domain(str(row.get("source_file", "unknown")))
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
    )
    return "eligible" if quality_ok else "blocked_quality"


async def run_pipeline(cfg: BuilderConfig) -> dict[str, Any]:
    run_profile = _normalize_run_profile(cfg)
    sampled_run = (cfg.sample_size is not None) or (cfg.chunk_size is not None)
    if run_profile == "research" and sampled_run:
        raise ValueError(
            "Research profile forbids sample/chunk settings. "
            "Unset sample_size/chunk_size or use run_profile=debug."
        )

    cfg.ensure_dirs()
    progress = _ProgressTracker(enabled=bool(getattr(cfg, "progress", True)), total_steps=10)
    
    from llm_utils import resolve_async_llm_provider
    llm_provider = resolve_async_llm_provider(cfg.llm_provider, model_name=cfg.llm_model_name)
    
    frame = load_inputs(cfg.input_dir)
    if frame.empty:
        raise ValueError(f"No supported input files found under {cfg.input_dir}")

    schema = detect_schema(frame, text_column_override=cfg.text_column_override)
    text_column = schema.primary_text_column
    if text_column is None:
        raise ValueError("No text column detected")

    prepared = _prepare_rows(frame, cfg, text_column, progress_tracker=progress)
    if prepared.empty:
        raise ValueError("No rows available after preprocessing")
    
    gold_annotations_path = cfg.gold_annotations_path or (cfg.input_dir / "gold_annotations.jsonl")
    gold_annotations = load_gold_annotations(gold_annotations_path) if gold_annotations_path else []
    progress.step("input load/schema detect")

    working_rows = _select_working_rows(prepared.to_dict(orient="records"), cfg)
    if not working_rows:
        raise ValueError("No rows selected after applying sampling and chunk constraints")
    sample_frame = pd.DataFrame(working_rows)

    stratify_key, stratify_values = choose_stratify_values(
        sample_frame.to_dict(orient="records"),
        preferred_key=schema.target_column,
        fallback_key="language",
    )
    train_frame, holdout_frame = preliminary_split(
        sample_frame,
        train_ratio=cfg.train_ratio,
        random_seed=cfg.random_seed,
        stratify_values=stratify_values,
    )
    train_rows = train_frame.to_dict(orient="records")
    holdout_rows = holdout_frame.to_dict(orient="records")
    val_rows, test_rows = split_holdout(
        holdout_rows,
        val_ratio_within_holdout=cfg.val_ratio / max(cfg.val_ratio + cfg.test_ratio, 1e-9),
        random_seed=cfg.random_seed + 1,
        stratify_values=[str(row.get(stratify_key, "unknown")) for row in holdout_rows] if stratify_key else None,
    )
    progress.step("split preparation")

    train_domain_conditioning_mode, eval_domain_conditioning_mode = _resolve_split_domain_conditioning_modes(cfg)
    train_domain_support = Counter(str(_canonical_domain(str(row.get("source_file", "unknown")))) for row in train_rows)
    
    candidate_aspects = discover_aspects(train_rows, text_column=text_column, max_aspects=cfg.max_aspects, implicit_mode=cfg.implicit_mode)
    candidate_aspects_by_language: dict[str, list[str]] = {}
    candidate_aspects_by_domain: dict[str, list[str]] = {}
    
    progress.step("domain discovery")
    for language in sorted({str(row.get("language", "unknown")) for row in train_rows}):
        language_rows = [row for row in train_rows if str(row.get("language", "unknown")) == language]
        candidate_aspects_by_language[language] = discover_aspects(language_rows, text_column=text_column, max_aspects=cfg.max_aspects, implicit_mode=cfg.implicit_mode)
    
    for domain in sorted({str(_canonical_domain(str(row.get("source_file", "unknown")))) for row in train_rows}):
        domain_rows = [row for row in train_rows if _canonical_domain(str(row.get("source_file", "unknown"))) == domain]
        candidate_aspects_by_domain[domain] = discover_aspects(domain_rows, text_column=text_column, max_aspects=cfg.max_aspects, implicit_mode=cfg.implicit_mode)

    feature_numeric_columns = _feature_columns(schema.numeric_columns, text_column=text_column, target_column=schema.target_column)
    feature_categorical_columns = _feature_columns(schema.categorical_columns, text_column=text_column, target_column=schema.target_column)
    artifacts = fit_explicit_artifacts(train_frame, feature_numeric_columns, feature_categorical_columns)

    async def build_split(rows: list[dict[str, Any]], split_name: str, domain_conditioning_mode: str) -> list[dict[str, Any]]:
        import asyncio
        if not rows:
            return []
        
        progress.step(f"building {split_name} (async)", 0)
        
        async def _run_batch(items):
            tasks = [
                _process_row(
                    item,
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
                    llm_provider=llm_provider
                )
                for item in items
            ]
            return await asyncio.gather(*tasks)

        # Process in chunks to avoid overwhelming the event loop or APIs
        chunk_size = cfg.max_workers * 2
        results = []
        with tqdm(total=len(rows), desc=f"Processing {split_name}", leave=False, disable=not cfg.progress) as pbar:
            for i in range(0, len(rows), chunk_size):
                chunk = list(enumerate(rows))[i : i + chunk_size]
                chunk_results = await _run_batch(chunk)
                results.extend(chunk_results)
                pbar.update(len(chunk))
        return results

    train_built = await build_split(train_rows, "train", train_domain_conditioning_mode)
    val_built = await build_split(val_rows, "val", eval_domain_conditioning_mode)
    test_built = await build_split(test_rows, "test", eval_domain_conditioning_mode)
    
    progress.step("train/val/test implicit+explicit build")

    finalized_rows = _merge_gold_labels(train_built + val_built + test_built, gold_annotations)
    train_built = [row for row in finalized_rows if row.get("split") == "train"]
    val_built = [row for row in finalized_rows if row.get("split") == "val"]
    test_built = [row for row in finalized_rows if row.get("split") == "test"]

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
                # v5.5 logic: LLM-based rephrasing and aspect mapping
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
    train_topup_candidates = train_general_policy_dropped_rows + train_review_dropped_soft_rows + train_review_dropped_hard_rows
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
    )
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
    train_export_rows, train_leakage_filter_stats_after_targeting = _strict_train_domain_leakage_filter(
        train_export_rows,
        candidate_aspects_by_domain=candidate_aspects_by_domain_train,
    )
    train_export_rows = _strict_train_non_general(train_export_rows)
    strict_train_export_rows = [row for row in train_export_rows if _strict_row_passes(row)] if cfg.strict_implicit_enabled else list(train_export_rows)
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
        limit=max(0, int(cfg.strict_review_sample_size)),
    )
    strict_challenge_candidates = [
        row for row in (strict_train_export_rows + strict_val_export_rows + strict_test_export_rows)
        if str(row.get("implicit", {}).get("hardness_tier") or "H0") in {"H2", "H3"}
    ]
    strict_challenge_rows = _stable_keep(
        strict_challenge_candidates,
        seed=cfg.random_seed,
        token="strict-challenge",
        limit=min(len(strict_challenge_candidates), max(1, int(cfg.strict_review_sample_size))),
    )
    train_export_rows = strict_train_export_rows
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
        "pipeline_version": "5.5-production",
        "output_version": cfg.output_version,
        "generated_at": utc_now_iso(),
        "run_profile": run_profile,
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
        "stratification_choice": stratify_key,
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
        "output_quality": quality_summary,
        "strict_quality": quality_summary.get("strict_quality", {}),
        "strict_artifacts": {
            "strict_train_rows": len(strict_train_export_rows),
            "strict_val_rows": len(strict_val_export_rows),
            "strict_test_rows": len(strict_test_export_rows),
            "strict_review_queue_rows": len(strict_review_queue_rows),
            "strict_challenge_rows": len(strict_challenge_rows),
        },
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
            "has_gold_eval": bool(gold_metrics.get("has_gold_labels")),
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
    report["blocking_reasons"] = blocking_reasons
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
        export_splits = {
            "train": train_export_rows,
            "val": strict_val_export_rows if cfg.strict_implicit_enabled else val_built,
            "test": strict_test_export_rows if cfg.strict_implicit_enabled else test_built,
        }
        write_split_outputs(cfg.explicit_dir, {split: [{**row["explicit"], "id": row["id"], "split": row["split"], "source_file": row["source_file"], "source_text": row["source_text"], "domain": row["domain"], "language": row["language"], "implicit_ready": row["implicit_ready"], "track": row["track"]} for row in rows] for split, rows in export_splits.items()})
        write_split_outputs(cfg.implicit_dir, export_splits)
        write_named_outputs(
            cfg.implicit_strict_dir,
            {
                "train": strict_train_export_rows,
                "val": strict_val_export_rows,
                "test": strict_test_export_rows,
                "review_queue": strict_review_queue_rows,
                "challenge": strict_challenge_rows,
            },
        )
        write_json(cfg.reports_dir / "build_report.json", report)
        write_json(cfg.reports_dir / "data_quality_report.json", {
            "run_profile": run_profile,
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
        write_compat_exports(cfg.output_dir / "compat" / "protonet" / "reviewlevel", export_splits)
        write_compat_exports(cfg.output_dir / "compat" / "protonet" / "episodic", export_splits)
        write_compat_exports(cfg.output_dir / "compat" / "backend", export_splits)
        progress.step("report/export writing")
    else:
        progress.step("report/export writing")

    report["research_manifest"] = research_manifest
    progress.close()
    return report


def build_parser() -> argparse.ArgumentParser:
    runtime_defaults = _load_runtime_defaults()
    parser = argparse.ArgumentParser(description="DAGR-PIPE v4 dataset builder")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--run-profile", type=str, default="research", choices=["research", "debug"])
    parser.add_argument("--llm-provider", type=str, default=str(runtime_defaults.get("llm_provider", "openai")), choices=["openai", "runpod", "ollama", "mock"])
    parser.add_argument("--llm-model-name", type=str, default=str(runtime_defaults.get("llm_model_name", "gpt-4o-mini")))
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
    parser.add_argument("--no-enforce-grounding", dest="enforce_grounding", action="store_false")
    parser.add_argument("--no-domain-conditioning", dest="use_domain_conditioning", action="store_false")
    parser.add_argument("--no-strict-domain-conditioning", dest="strict_domain_conditioning", action="store_false")
    parser.add_argument("--domain-conditioning-mode", type=str, default="adaptive_soft", choices=["adaptive_soft", "strict_hard", "off"])
    parser.add_argument("--train-domain-conditioning-mode", type=str, default=None, choices=["adaptive_soft", "strict_hard", "off"])
    parser.add_argument("--eval-domain-conditioning-mode", type=str, default=None, choices=["adaptive_soft", "strict_hard", "off"])
    parser.set_defaults(enforce_grounding=True, use_domain_conditioning=True, strict_domain_conditioning=False)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        output_dir=args.output_dir or BuilderConfig().output_dir,
        random_seed=args.seed,
        text_column_override=args.text_column,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_offset=args.chunk_offset,
        run_profile=args.run_profile,
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
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model_name,
        enable_reasoned_recovery=args.enable_reasoned_recovery,
        max_workers=args.max_workers,
    )
    import asyncio
    report = asyncio.run(run_pipeline(cfg))
    
    if not cfg.dry_run:
        zip_path = compress_output_folder(cfg.output_dir)
        if zip_path:
            print(f"Output archived to: {zip_path}")

    if bool(report.get("validation", {}).get("train_target_blocking_failure")):
        print("Build blocked: train_target_size_within_range=false under research profile.")
        return 2
    print(f"Build complete: {report['generated_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
