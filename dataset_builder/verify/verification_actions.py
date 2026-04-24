from __future__ import annotations

from ..schemas.interpretation import Interpretation


def apply_keep_drop_merge_add(items: list[Interpretation], actions: list[dict]) -> list[Interpretation]:
    drop = {str(action.get("aspect_canonical")) for action in actions if action.get("action") == "drop"}
    return [item for item in items if item.aspect_canonical not in drop]
