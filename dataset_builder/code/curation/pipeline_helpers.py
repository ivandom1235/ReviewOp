from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from ..contracts import BuilderConfig
    from ..utils.utils import stable_id
except (ImportError, ValueError):  # pragma: no cover
    from contracts import BuilderConfig
    from utils.utils import stable_id

_CORE_BENCHMARK_DOMAINS = ("electronics", "restaurant", "telecom")


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


def assign_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "source_file" not in out.columns:
        out["source_file"] = "unknown"

    def get_row_id(row_tuple):
        return stable_id(row_tuple.source_file, row_tuple.Index, str(row_tuple))

    out["id"] = [get_row_id(row) for row in out.itertuples()]
    return out


def harvest_dataset_aspects(frame: pd.DataFrame) -> list[tuple[str, set[str], set[str]]]:
    aspect_cols = [c for c in frame.columns if c.lower() in {"aspect", "gold_aspect", "target_aspect", "implicit_aspect"}]
    if not aspect_cols:
        return []
    discovered_labels = set()
    for col in aspect_cols:
        for v in frame[col].dropna().unique():
            if isinstance(v, str) and len(v.strip()) > 2:
                discovered_labels.add(v.strip().lower())
    return [(label, {label}, set()) for label in discovered_labels]


def get_row_domain(row: dict[str, Any]) -> str:
    explicit_domain = row.get("domain")
    source_file = row.get("source_file", "unknown")
    if isinstance(explicit_domain, str) and explicit_domain and explicit_domain != "unknown":
        return explicit_domain.strip().lower()
    return canonical_domain(str(source_file))


def canonical_domain(source_file: str | None) -> str:
    name = Path(str(source_file or "unknown")).stem.lower()
    for suffix in ("_train", "_test", "_val", "-train", "-test", "-val"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    domain_map = {"product_reviews_mock_data": "product", "fake reviews dataset": "general_product"}
    return domain_map.get(name, name)


def chunk_rows(rows: list[dict[str, Any]], cfg: BuilderConfig) -> list[dict[str, Any]]:
    ordered = list(rows)
    random.Random(cfg.random_seed).shuffle(ordered)
    return ordered


def benchmark_domain_family(domain: str | None) -> str:
    normalized = str(domain or "unknown").strip().lower()
    if normalized in {"laptop", "electronics"}:
        return "electronics"
    if normalized in {"restaurant", "telecom"}:
        return normalized
    return normalized or "unknown"


def benchmark_row_priority(row: dict[str, Any], *, preferred_domain: str | None = None) -> tuple[float, str]:
    implicit = row.get("implicit", {}) or {}
    aspects = [str(aspect) for aspect in implicit.get("aspects", []) if str(aspect) != "general"]
    hardness = str(implicit.get("hardness_tier") or "").strip().upper()
    review_reason = str(implicit.get("review_reason") or "").strip().lower()
    sentiment = str(implicit.get("dominant_sentiment") or row.get("sentiment") or "").strip().lower()
    domain_family = benchmark_domain_family(str(get_row_domain(row)))

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
    stable = stable_id("benchmark-selection", row.get("id") or row.get("source_text") or "", row.get("domain") or "unknown", row.get("split") or "train")
    return (-score, stable)


def select_working_rows(rows: list[dict[str, Any]], cfg: BuilderConfig) -> list[dict[str, Any]]:
    ordered = chunk_rows(rows, cfg)
    if cfg.sample_size is None and cfg.chunk_size is None:
        return ordered

    target_size = cfg.sample_size if cfg.sample_size is not None else (cfg.chunk_size if cfg.chunk_size is not None else len(ordered))
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_domain[benchmark_domain_family(str(get_row_domain(row)))].append(row)

    sorted_domains = sorted(by_domain.keys())
    quota_per_domain = max(1, target_size // max(1, len(sorted_domains)))
    prioritized: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for domain in sorted_domains:
        domain_rows = sorted(by_domain[domain], key=lambda row: benchmark_row_priority(row, preferred_domain=domain))
        for row in domain_rows[:quota_per_domain]:
            chosen_id = str(row.get("id") or "")
            if chosen_id and chosen_id not in selected_ids:
                prioritized.append(row)
                selected_ids.add(chosen_id)

    remaining = [row for row in ordered if str(row.get("id") or "") not in selected_ids]
    remaining_sorted = sorted(remaining, key=lambda row: benchmark_row_priority(row, preferred_domain=benchmark_domain_family(str(get_row_domain(row)))))
    prioritized.extend(remaining_sorted)

    if cfg.sample_size is not None:
        prioritized = prioritized[: max(0, cfg.sample_size)]
    if cfg.chunk_size is not None:
        start = max(0, cfg.chunk_offset)
        end = start + max(0, cfg.chunk_size)
        prioritized = prioritized[start:end]
    return prioritized
