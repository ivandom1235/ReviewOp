from __future__ import annotations

import argparse
import random
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db import SessionLocal, init_db
from models.tables import Review, Prediction  # noqa: E402
from services.analytics import dashboard_kpis, aspect_leaderboard, segment_drilldown, aspect_detail  # noqa: E402


ASPECTS = [
    "battery",
    "performance",
    "quality",
    "price",
    "display",
    "support",
    "durability",
    "service",
]
SENTIMENTS = ["positive", "neutral", "negative"]
DOMAINS = ["electronics", "food", "transport"]


def _maybe_seed_fixture(db, target_reviews: int, seed: int) -> None:
    existing = int(db.query(Review.id).count())
    missing = max(0, target_reviews - existing)
    if missing == 0:
        return

    rand = random.Random(seed)
    now = datetime.utcnow()
    reviews: list[Review] = []
    for idx in range(missing):
        reviews.append(
            Review(
                text=f"Benchmark review {idx} about {ASPECTS[idx % len(ASPECTS)]}",
                domain=DOMAINS[idx % len(DOMAINS)],
                product_id=f"P{(idx % 250) + 1}",
                created_at=now - timedelta(minutes=idx % 60_000),
            )
        )
    db.add_all(reviews)
    db.flush()

    preds: list[Prediction] = []
    for review in reviews:
        sample_size = rand.randint(2, 5)
        sampled_aspects = rand.sample(ASPECTS, k=sample_size)
        for aspect in sampled_aspects:
            preds.append(
                Prediction(
                    review_id=review.id,
                    aspect_raw=aspect,
                    aspect_cluster=aspect,
                    sentiment=rand.choice(SENTIMENTS),
                    confidence=round(rand.uniform(0.55, 0.98), 4),
                    rationale=None,
                )
            )
    db.add_all(preds)
    db.commit()


def _bench(label: str, fn, iterations: int) -> list[float]:
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    med = statistics.median(timings)
    p95 = sorted(timings)[max(0, int(round(iterations * 0.95)) - 1)]
    print(f"{label:24} median={med:8.2f}ms p95={p95:8.2f}ms")
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark analytics service latency on a 5k+ review fixture")
    parser.add_argument("--target-reviews", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-ms", type=float, default=200.0)
    parser.add_argument("--assert-threshold", action="store_true")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        _maybe_seed_fixture(db, target_reviews=args.target_reviews, seed=args.seed)

        results = {
            "dashboard_kpis": _bench("dashboard_kpis", lambda: dashboard_kpis(db, None, None, None), args.iterations),
            "aspect_leaderboard": _bench("aspect_leaderboard", lambda: aspect_leaderboard(db, limit=25, domain=None), args.iterations),
            "segment_drilldown": _bench("segment_drilldown", lambda: segment_drilldown(db, domain=None, limit=20), args.iterations),
            "aspect_detail": _bench("aspect_detail", lambda: aspect_detail(db, "battery", interval="day", domain=None), args.iterations),
        }
    finally:
        db.close()

    if args.assert_threshold:
        failures: list[str] = []
        for name, timings in results.items():
            med = statistics.median(timings)
            p95 = sorted(timings)[max(0, int(round(len(timings) * 0.95)) - 1)]
            if med > args.threshold_ms or p95 > args.threshold_ms:
                failures.append(f"{name} median={med:.2f}ms p95={p95:.2f}ms")
        if failures:
            print("FAIL: threshold exceeded")
            for item in failures:
                print(f"- {item}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
