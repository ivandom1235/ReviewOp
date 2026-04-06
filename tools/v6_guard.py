from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "V6_CONTRACT_MANIFEST.json"
SEARCH_DIRS = ["dataset_builder/code", "protonet/code", "backend", "frontend/src"]
ALLOWED_TEXT_EXT = {".py", ".md", ".json", ".jsx", ".js", ".ts", ".tsx"}


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Missing V6_CONTRACT_MANIFEST.json")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _scan_for_blocked_tokens(blocked_tokens: list[str]) -> list[str]:
    violations: list[str] = []
    for rel in SEARCH_DIRS:
        base = ROOT / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOWED_TEXT_EXT:
                continue
            if (
                "node_modules" in path.parts
                or "dist" in path.parts
                or "venv" in path.parts
                or ".venv" in path.parts
                or "output" in path.parts
                or "metadata" in path.parts
                or "tests" in path.parts
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for token in blocked_tokens:
                if token.lower() in text:
                    violations.append(f"{path.relative_to(ROOT)} contains blocked token '{token}'")
    return violations


def _check_required_paths(manifest: dict) -> list[str]:
    failures: list[str] = []
    paths = manifest.get("runtime_contracts", {}).get("dataset_builder", {}).get("primary_artifacts", [])
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"missing expected artifact path: {rel}")
    return failures


def main() -> int:
    manifest = _load_manifest()
    blocked_tokens = list(manifest.get("deprecated_signals", {}).get("blocked_tokens", []))
    violations = _scan_for_blocked_tokens(blocked_tokens)
    violations.extend(_check_required_paths(manifest))
    if violations:
        for item in violations:
            print(item)
        return 1
    print("v6_guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
