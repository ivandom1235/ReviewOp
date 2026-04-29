from __future__ import annotations

import argparse
import json
from statistics import median

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find weird/outlier nodes from /graph/aspects response.")
    p.add_argument("--api-base", default="http://127.0.0.1:8000")
    p.add_argument("--token", required=True, help="Admin bearer token")
    p.add_argument("--domain", default=None)
    p.add_argument("--product-id", default=None)
    p.add_argument("--min-edge-weight", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params: dict[str, str | int] = {"min_edge_weight": args.min_edge_weight}
    if args.domain:
        params["domain"] = args.domain
    if args.product_id:
        params["product_id"] = args.product_id

    url = f"{args.api_base.rstrip('/')}/graph/aspects"
    headers = {"Authorization": f"Bearer {args.token}"}
    res = requests.get(url, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    payload = res.json()

    nodes = payload.get("nodes", [])
    freqs = [float(n.get("frequency", 0) or 0) for n in nodes]
    freq_median = median(freqs) if freqs else 0.0

    weird: list[dict] = []
    for n in nodes:
        explicit_count = int(n.get("explicit_count", 0) or 0)
        implicit_count = int(n.get("implicit_count", 0) or 0)
        mentions = explicit_count + implicit_count
        if mentions <= 0:
            continue

        implicit_ratio = implicit_count / mentions
        frequency = int(n.get("frequency", 0) or 0)
        negative_ratio = float(n.get("negative_ratio", 0) or 0.0)
        avg_sentiment = float(n.get("avg_sentiment", 0) or 0.0)
        node_id = str(n.get("id", "") or "")
        dominant = str(n.get("dominant_sentiment", "neutral") or "neutral")

        flags: list[str] = []
        if mentions >= 20 and implicit_ratio >= 0.95:
            flags.append("implicit_dominant_high_support")
        if mentions >= 20 and implicit_ratio <= 0.05 and dominant == "negative" and negative_ratio < 0.2:
            flags.append("sentiment_inconsistency")
        if frequency >= max(10, int(4 * freq_median)) and node_id.strip().lower() in {"", "none", "null", "n/a"}:
            flags.append("invalid_id_high_frequency")

        if flags:
            weird.append(
                {
                    "id": node_id,
                    "frequency": frequency,
                    "mentions": mentions,
                    "explicit_count": explicit_count,
                    "implicit_count": implicit_count,
                    "implicit_ratio": round(implicit_ratio, 4),
                    "dominant_sentiment": dominant,
                    "negative_ratio": round(negative_ratio, 4),
                    "avg_sentiment": round(avg_sentiment, 4),
                    "flags": flags,
                }
            )

    weird.sort(key=lambda x: (-x["frequency"], -x["mentions"], x["id"]))
    print(json.dumps({"node_count": len(nodes), "weird_count": len(weird), "weird_nodes": weird[:50]}, indent=2))


if __name__ == "__main__":
    main()
