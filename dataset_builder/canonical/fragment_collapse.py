from __future__ import annotations

from ..schemas.interpretation import Interpretation


def collapse_same_evidence_fragments(items: list[Interpretation]) -> tuple[list[Interpretation], dict[str, int]]:
    buckets: dict[tuple[str, tuple[int, int], str], list[Interpretation]] = {}
    for item in items:
        key = (item.evidence_text.lower(), tuple(item.evidence_span), item.sentiment.lower())
        buckets.setdefault(key, []).append(item)
    kept: list[Interpretation] = []
    dropped = 0
    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda item: (1 if item.pattern_id else 0, len(item.aspect_canonical), item.canonical_confidence), reverse=True)
        winner = ordered[0]
        kept.append(winner)
        for loser in ordered[1:]:
            if loser.aspect_canonical in winner.aspect_canonical or loser.aspect_raw.lower() in winner.aspect_raw.lower():
                dropped += 1
            else:
                kept.append(loser)
    return kept, {"dropped_fragments": dropped}
