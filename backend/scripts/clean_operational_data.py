from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.db import SessionLocal


TARGET_TABLES = [
    "evidence_spans",
    "predictions",
    "abstained_predictions",
    "novel_candidates",
    "job_items",
    "jobs",
    "aspect_edges",
    "aspect_nodes",
    "alerts",
    "admin_dismissed_alerts",
    "reviews",
]


def main() -> None:
    with SessionLocal() as db:
        before = {}
        for table in TARGET_TABLES:
            before[table] = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0

        # Keep user-generated review records, but detach links to inferred review rows.
        db.execute(
            text("UPDATE user_product_reviews SET linked_review_id = NULL WHERE linked_review_id IS NOT NULL")
        )

        for table in TARGET_TABLES:
            db.execute(text(f"DELETE FROM {table}"))

        db.commit()

        after = {}
        for table in TARGET_TABLES:
            after[table] = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0

    print("Operational cleanup complete.")
    print("Before:")
    for table, count in before.items():
        print(f"  {table}: {count}")
    print("After:")
    for table, count in after.items():
        print(f"  {table}: {count}")
    print("Preserved: users, user_sessions, products, user_product_reviews, llm_cache (untouched).")


if __name__ == "__main__":
    main()
