from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_runtime_defaults() -> dict[str, Any]:
    defaults_path = Path(__file__).resolve().parent / "runtime_defaults.json"
    if not defaults_path.exists():
        return {}
    try:
        payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    defaults = payload.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def optional_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def resolve_artifact_mode(*, run_profile: str, artifact_mode: str | None) -> str:
    raw = str(artifact_mode or "auto").strip().lower()
    if raw in {"debug_artifacts", "research_release"}:
        return raw
    return "debug_artifacts" if str(run_profile).strip().lower() == "debug" else "research_release"


def resolve_domain_conditioning_modes(
    *,
    domain_conditioning_mode: str | None,
    use_domain_conditioning: bool,
    strict_domain_conditioning: bool,
    train_domain_conditioning_mode: str | None,
    eval_domain_conditioning_mode: str | None,
) -> tuple[str, str]:
    mode = str(domain_conditioning_mode or "").strip().lower()
    if not use_domain_conditioning:
        mode = "off"
    elif strict_domain_conditioning and mode == "adaptive_soft":
        mode = "strict_hard"

    train_mode = str(train_domain_conditioning_mode or "").strip().lower() or None
    eval_mode = str(eval_domain_conditioning_mode or "").strip().lower() or None
    if train_mode is None or eval_mode is None:
        if mode == "strict_hard":
            train_mode = train_mode or "strict_hard"
            eval_mode = eval_mode or "strict_hard"
        elif mode == "off":
            train_mode = train_mode or "off"
            eval_mode = eval_mode or "off"
        else:
            train_mode = train_mode or "strict_hard"
            eval_mode = eval_mode or "adaptive_soft"

    return train_mode or "strict_hard", eval_mode or "adaptive_soft"
