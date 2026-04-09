from __future__ import annotations

import subprocess
import sys
import unittest
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "dataset_builder" / "code", REPO_ROOT / "protonet" / "code"):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)


def _count_tests(suite: unittest.TestSuite) -> int:
    total = 0
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            total += _count_tests(item)
        else:
            total += 1
    return total


def _run_unittest_suite(name: str, start_dir: Path) -> int:
    print(f"\n==> {name}")
    if not start_dir.exists():
        print(f"[skip] {name}: directory not found at {start_dir}")
        return 0

    suite = unittest.defaultTestLoader.discover(str(start_dir), pattern="test_*.py")
    test_count = _count_tests(suite)
    if test_count == 0:
        print(f"[skip] {name}: no test files found")
        return 0

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _run_frontend_build() -> int:
    print("\n==> frontend")
    npm_cmd = shutil.which("npm.cmd") or shutil.which("npm") or "npm"
    result = subprocess.run([npm_cmd, "--prefix", "frontend", "run", "build"], cwd=REPO_ROOT)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    targets = [arg.strip().lower() for arg in (argv if argv is not None else sys.argv[1:]) if arg.strip()]
    if not targets:
        targets = ["backend", "dataset-builder", "protonet", "frontend"]

    exit_code = 0
    for target in targets:
        if target == "backend":
            exit_code |= _run_unittest_suite("backend", REPO_ROOT / "backend")
        elif target in {"dataset-builder", "dataset_builder"}:
            exit_code |= _run_unittest_suite("dataset_builder", REPO_ROOT / "dataset_builder" / "tests")
        elif target == "protonet":
            exit_code |= _run_unittest_suite("protonet", REPO_ROOT / "protonet" / "tests")
        elif target == "frontend":
            exit_code |= _run_frontend_build()
        else:
            print(f"[warn] unknown target skipped: {target}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
