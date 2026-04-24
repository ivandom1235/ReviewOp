from __future__ import annotations

from dataclasses import dataclass

from ..schemas.interpretation import Interpretation


@dataclass(frozen=True)
class ReviewQueueRecord:
    row_id: str
    interpretations: list[Interpretation]
    reason_codes: list[str]


def queue_for_review(row_id: str, interpretations: list[Interpretation], reason_codes: list[str]) -> ReviewQueueRecord:
    return ReviewQueueRecord(row_id=row_id, interpretations=interpretations, reason_codes=list(reason_codes))


def review_reason_breakdown(records: list[ReviewQueueRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for reason in record.reason_codes:
            counts[reason] = counts.get(reason, 0) + 1
    return counts
