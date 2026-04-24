from __future__ import annotations

from ..schemas.interpretation import Interpretation


def heuristic_verify(items: list[Interpretation]) -> list[dict[str, object]]:
    return [{"action": "keep", "aspect_canonical": item.aspect_canonical} for item in items]
