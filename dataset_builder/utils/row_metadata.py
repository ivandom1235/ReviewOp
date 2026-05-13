from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ..schemas.benchmark_row import BenchmarkRow


def _derive_query_text(row: BenchmarkRow) -> str:
    # Priority: exact evidence span -> sentence/phrase evidence -> matched-term window fallback -> full review
    for interp in row.gold_interpretations or []:
        evidence_text = str(getattr(interp, "evidence_text", "") or "").strip()
        evidence_scope = str(getattr(interp, "evidence_scope", "") or "").strip()
        if evidence_text and len(evidence_text.split()) >= 3 and evidence_scope in {"exact_phrase", "token_span", "phrase_window"}:
            return evidence_text
    for interp in row.gold_interpretations or []:
        evidence_text = str(getattr(interp, "evidence_text", "") or "").strip()
        if evidence_text and len(evidence_text.split()) >= 3:
            return evidence_text
    return str(row.review_text or "")


def _gold_values(row: BenchmarkRow, field: str) -> list[str]:
    values: list[str] = []
    for interp in row.gold_interpretations or []:
        value = getattr(interp, field, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            values.append(text)
    return values


def derive_row_source_type(gold_interpretations: Iterable[object]) -> str:
    types: set[str] = set()
    for interp in gold_interpretations or []:
        source_type = str(getattr(interp, "source_type", "") or "").strip()
        if not source_type:
            continue
        if source_type == "explicit":
            types.add("explicit")
        elif source_type in {"implicit_learned", "implicit_json", "implicit_llm", "merged"}:
            types.add("implicit")
        else:
            types.add("unknown")
    if not types:
        return "unknown"
    if types == {"explicit"}:
        return "explicit_only"
    if types == {"implicit"}:
        return "implicit_only"
    if "explicit" in types and "implicit" in types:
        return "hybrid"
    return "unknown"


def derive_row_mapping_scope(gold_interpretations: Iterable[object]) -> str:
    scopes: list[str] = []
    for interp in gold_interpretations or []:
        scope = str(getattr(interp, "mapping_scope", "") or "").strip()
        if scope:
            scopes.append(scope)
    if not scopes:
        return "none"
    unique = sorted(set(scopes))
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def derive_row_mapping_sources(gold_interpretations: Iterable[object]) -> tuple[str, ...]:
    sources: list[str] = []
    for interp in gold_interpretations or []:
        source = str(getattr(interp, "mapping_source", "") or "").strip()
        if source:
            sources.append(source)
    return tuple(sorted(set(sources)))


def derive_row_metadata(row: BenchmarkRow) -> BenchmarkRow:
    query_text = _derive_query_text(row)
    if bool(getattr(row, "abstain_acceptable", False)) and not tuple(row.gold_interpretations or ()):
        return replace(
            row,
            query_text=query_text,
            source_type="abstain",
            mapping_source="abstain",
            mapping_scope="abstain",
            row_source_type="abstain",
            row_mapping_scope="abstain",
            row_mapping_sources=("abstain",),
        )
    mapping_sources = derive_row_mapping_sources(row.gold_interpretations)
    mapping_scope = derive_row_mapping_scope(row.gold_interpretations)
    source_type = derive_row_source_type(row.gold_interpretations)
    mapping_source = "|".join(mapping_sources) if mapping_sources else "unknown"
    return replace(
        row,
        query_text=query_text,
        source_type=source_type,
        mapping_source=mapping_source,
        mapping_scope=mapping_scope,
        row_source_type=source_type,
        row_mapping_scope=mapping_scope,
        row_mapping_sources=mapping_sources,
    )
