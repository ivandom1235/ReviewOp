from __future__ import annotations

from .merge_candidates import dedupe_merged_candidates
from ..schemas.interpretation import Interpretation


def resolve_same_evidence_duplicates(items: list[Interpretation]) -> list[Interpretation]:
    return dedupe_merged_candidates(items)


def resolve_same_aspect_duplicates(items: list[Interpretation]) -> list[Interpretation]:
    return dedupe_merged_candidates(items)
