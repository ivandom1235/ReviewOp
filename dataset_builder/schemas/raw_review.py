from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawReview:
    review_id: str
    group_id: str
    domain: str
    domain_family: str
    text: str
    source_name: str = "unknown"
    source_split: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.review_id).strip():
            raise ValueError("review_id is required")
        if not str(self.group_id).strip():
            raise ValueError("group_id is required")
        if not str(self.text).strip():
            raise ValueError("text is required")
