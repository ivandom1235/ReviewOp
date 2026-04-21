from __future__ import annotations

from typing import Any


def build_pipeline_state(
    *,
    train: dict[str, Any],
    benchmark: dict[str, Any],
    evaluation: dict[str, Any],
    governance: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "train": dict(train),
        "benchmark": dict(benchmark),
        "evaluation": dict(evaluation),
        "governance": dict(governance),
    }
