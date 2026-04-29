from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import requests

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


TEXT_CANDIDATE_FIELDS = [
    "text",
    "review_text",
    "review",
    "sentence",
    "content",
    "comment",
    "body",
]


def _detect_text_key(record: dict) -> str | None:
    lowered = {str(k).lower(): k for k in record.keys()}
    for cand in TEXT_CANDIDATE_FIELDS:
        if cand in lowered:
            return lowered[cand]
    return None


def _read_records(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
        return
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
        elif isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("data") or []
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        yield row
        return
    raise ValueError(f"Unsupported file type: {path.suffix}")


def run_dataset(
    *,
    dataset_path: Path,
    api_base: str,
    token: str,
    domain: str,
    limit: int,
    persist: bool,
    timeout_seconds: int,
) -> tuple[int, int]:
    sent = 0
    failed = 0
    attempted = 0
    infer_url = f"{api_base.rstrip('/')}/infer/review"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    chosen_key = None
    for idx, record in enumerate(_read_records(dataset_path), start=1):
        if limit > 0 and attempted >= limit:
            break

        if chosen_key is None:
            chosen_key = _detect_text_key(record)
            if not chosen_key:
                raise RuntimeError(
                    f"Could not detect text field in {dataset_path}. "
                    f"Expected one of: {', '.join(TEXT_CANDIDATE_FIELDS)}"
                )

        text = str(record.get(chosen_key) or "").strip()
        if not text:
            continue
        attempted += 1

        payload = {
            "text": text,
            "domain": domain,
            "persist": persist,
        }
        try:
            response = requests.post(infer_url, headers=headers, json=payload, timeout=timeout_seconds)
            if response.status_code >= 400:
                failed += 1
                print(f"[{dataset_path.name}] row {idx}: HTTP {response.status_code} - {response.text[:200]}")
            else:
                sent += 1
                if attempted % 25 == 0:
                    print(f"[{dataset_path.name}] attempted {attempted} rows...")
        except Exception as ex:  # noqa: BLE001
            failed += 1
            print(f"[{dataset_path.name}] row {idx}: request failed: {ex}")

    return sent, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call /infer/review one-by-one for laptop and restaurant datasets."
    )
    parser.add_argument("--laptop-path", required=True, help="Path to laptop dataset (.csv/.jsonl/.json)")
    parser.add_argument("--restaurant-path", required=True, help="Path to restaurant dataset (.csv/.jsonl/.json)")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000", help="Backend API base URL")
    parser.add_argument("--token", required=True, help="Admin bearer token")
    parser.add_argument("--limit-per-dataset", type=int, default=100, help="Max rows to send for each dataset (<=0 means no limit)")
    parser.add_argument("--persist", action="store_true", help="Persist inference results into backend (default false)")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="HTTP timeout seconds per request")
    args = parser.parse_args()

    laptop_path = Path(args.laptop_path).resolve()
    restaurant_path = Path(args.restaurant_path).resolve()
    if not laptop_path.exists():
        raise FileNotFoundError(f"Laptop dataset not found: {laptop_path}")
    if not restaurant_path.exists():
        raise FileNotFoundError(f"Restaurant dataset not found: {restaurant_path}")

    print("Starting inference run...")
    print(f"  laptop: {laptop_path}")
    print(f"  restaurant: {restaurant_path}")
    print(f"  api: {args.api_base}")
    print(f"  limit per dataset: {args.limit_per_dataset}")
    print(f"  persist: {args.persist}")

    lap_sent, lap_failed = run_dataset(
        dataset_path=laptop_path,
        api_base=args.api_base,
        token=args.token,
        domain="laptop",
        limit=args.limit_per_dataset,
        persist=args.persist,
        timeout_seconds=args.timeout_seconds,
    )
    rest_sent, rest_failed = run_dataset(
        dataset_path=restaurant_path,
        api_base=args.api_base,
        token=args.token,
        domain="restaurant",
        limit=args.limit_per_dataset,
        persist=args.persist,
        timeout_seconds=args.timeout_seconds,
    )

    print("Done.")
    print(f"  laptop -> sent: {lap_sent}, failed: {lap_failed}")
    print(f"  restaurant -> sent: {rest_sent}, failed: {rest_failed}")


if __name__ == "__main__":
    main()
