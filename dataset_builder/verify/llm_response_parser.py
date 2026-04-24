from __future__ import annotations

import json


def parse_keep_drop_merge_add(text: str) -> list[dict]:
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("verifier response must be a list")
    return payload


def validate_verifier_json(payload: list[dict]) -> bool:
    return all(item.get("action") in {"keep", "drop", "merge", "add"} for item in payload)
