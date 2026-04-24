from __future__ import annotations

from ..schemas.interpretation import Interpretation


def merge_explicit_implicit(explicit: list[Interpretation], implicit: list[Interpretation]) -> list[Interpretation]:
    return dedupe_merged_candidates([*explicit, *implicit])


def dedupe_merged_candidates(items: list[Interpretation]) -> list[Interpretation]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Interpretation] = []
    for item in sorted(items, key=lambda i: (i.canonical_confidence, 1 if i.pattern_id else 0), reverse=True):
        key = (item.aspect_canonical, item.sentiment, item.evidence_text)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
