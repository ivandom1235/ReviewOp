from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import requests
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
DATASETS_DIR = ROOT / "datasets"
DEFAULT_LAPTOP = DATASETS_DIR / "Laptop_train.csv"
DEFAULT_RESTAURANT = DATASETS_DIR / "Restaurant_train.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run /infer/review on laptop and restaurant datasets.")
    parser.add_argument("--limit", type=int, default=100, help="Rows per dataset to process (<=0 means all).")
    parser.add_argument("--persist", action="store_true", help="Persist inference output.")
    return parser.parse_args(argv)


def _load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def resolve_config(args: argparse.Namespace) -> dict:
    env = _load_env(ENV_PATH)
    api_base = env.get("VITE_PROXY_TARGET", "http://127.0.0.1:8000")
    username = os.getenv("REVIEWOP_ADMIN_USERNAME") or env.get("REVIEWOP_ADMIN_USERNAME", "admin")
    password = os.getenv("REVIEWOP_ADMIN_PASSWORD") or env.get("REVIEWOP_ADMIN_PASSWORD", "admin123")
    return {
        "api_base": api_base,
        "username": username,
        "password": password,
        "limit": args.limit,
        "persist": bool(args.persist),
        "laptop_path": DEFAULT_LAPTOP,
        "restaurant_path": DEFAULT_RESTAURANT,
    }


def _read_records(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    raise ValueError(f"Unsupported dataset format: {path}")


def _detect_text_key(sample: dict) -> str:
    options = ["review_text", "text", "review", "sentence", "content"]
    lowered = {str(k).lower(): k for k in sample.keys()}
    for k in options:
        if k in lowered:
            return lowered[k]
    raise RuntimeError(f"No review text field found. Keys: {list(sample.keys())}")


def _login_token(api_base: str, username: str, password: str) -> str:
    candidates = [
        f"{api_base.rstrip('/')}/user/auth/login",
        f"{api_base.rstrip('/')}/auth/login",
    ]
    last_error = None
    for url in candidates:
        try:
            r = requests.post(url, json={"username": username, "password": password}, timeout=20)
        except Exception as ex:  # noqa: BLE001
            last_error = f"{url}: request failed: {ex}"
            continue
        if r.status_code >= 400:
            last_error = f"{url}: {r.status_code} {r.text[:200]}"
            continue
        payload = r.json()
        token = payload.get("token")
        if token:
            return token
        last_error = f"{url}: login response missing token"
    raise RuntimeError(f"Login failed. {last_error}")


def _run_dataset(path: Path, domain: str, cfg: dict, token: str, progress: Progress) -> tuple[int, int]:
    sent = failed = attempted = 0
    infer_url = f"{cfg['api_base'].rstrip('/')}/infer/review"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    text_key = None
    total_target = cfg["limit"] if cfg["limit"] > 0 else None
    task = progress.add_task(
        f"[cyan]{domain}[/cyan]",
        total=total_target,
        stats="ok=0 fail=0",
    )

    for idx, row in enumerate(_read_records(path), start=1):
        if cfg["limit"] > 0 and attempted >= cfg["limit"]:
            break
        if text_key is None:
            text_key = _detect_text_key(row)
        text = str(row.get(text_key) or "").strip()
        if not text:
            continue
        attempted += 1
        payload = {"text": text, "domain": domain, "persist": cfg["persist"]}
        res = requests.post(infer_url, headers=headers, json=payload, timeout=60)
        if res.status_code >= 400:
            failed += 1
            progress.console.print(f"[{path.name}] row {idx}: HTTP {res.status_code} {res.text[:120]}")
        else:
            sent += 1
        progress.update(task, advance=1 if total_target else 0, completed=attempted if not total_target else None, stats=f"ok={sent} fail={failed}")
    return sent, failed


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    for p in [cfg["laptop_path"], cfg["restaurant_path"]]:
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
    token = _login_token(cfg["api_base"], cfg["username"], cfg["password"])
    print(f"API: {cfg['api_base']} | limit={cfg['limit']} | persist={cfg['persist']}")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}" if cfg["limit"] > 0 else "{task.completed}"),
        TextColumn("{task.fields[stats]}"),
        TimeElapsedColumn(),
    ) as progress:
        l_ok, l_fail = _run_dataset(cfg["laptop_path"], "laptop", cfg, token, progress)
        r_ok, r_fail = _run_dataset(cfg["restaurant_path"], "restaurant", cfg, token, progress)
    print(f"Laptop: sent={l_ok} failed={l_fail}")
    print(f"Restaurant: sent={r_ok} failed={r_fail}")


if __name__ == "__main__":
    main()
