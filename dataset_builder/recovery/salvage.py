from __future__ import annotations

from .review_queue import ReviewQueueRecord


def salvage_one_issue_rows(records: list[ReviewQueueRecord]) -> list[ReviewQueueRecord]:
    return [record for record in records if len(record.reason_codes) == 1]
