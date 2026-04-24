from __future__ import annotations

from .review_queue import ReviewQueueRecord


def topup_from_review_queue(records: list[ReviewQueueRecord], limit: int) -> list[ReviewQueueRecord]:
    return records[: max(0, limit)]
