from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
for extra_path in (ROOT, ROOT / "backend"):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from services import analytics


class AdminExportTests(unittest.TestCase):
    def test_export_payload_uses_expected_sections(self) -> None:
        sentinel_db = object()
        fixed_now = datetime(2026, 4, 10, 12, 0, 0)
        with (
            patch.object(analytics, "datetime") as mock_datetime,
            patch.object(analytics, "dashboard_kpis", return_value={"total_reviews": 10}) as dashboard_kpis,
            patch.object(analytics, "aspect_leaderboard", return_value=[{"aspect": "battery"}]) as aspect_leaderboard,
            patch.object(analytics, "aspect_trends", return_value=[{"bucket": "2026-04-10"}]) as aspect_trends,
            patch.object(analytics, "emerging_aspects", return_value=[{"aspect": "battery"}]) as emerging_aspects,
            patch.object(analytics, "evidence_drilldown", return_value=[{"review_id": 1}]) as evidence_drilldown,
            patch.object(analytics, "alerts", return_value=[{"id": 1}]) as alerts,
            patch.object(analytics, "impact_matrix", return_value=[{"aspect": "battery"}]) as impact_matrix,
            patch.object(analytics, "segment_drilldown", return_value=[{"segment_type": "domain"}]) as segment_drilldown,
            patch.object(analytics, "weekly_summary", return_value={"period_label": "this week"}) as weekly_summary,
            patch.object(analytics, "user_reviews_summary", return_value={"total_user_reviews": 3}) as user_reviews_summary,
            patch.object(analytics, "user_reviews_list", return_value={"total": 1, "rows": [{"review_id": 1}]}) as user_reviews_list,
        ):
            mock_datetime.utcnow.return_value = fixed_now
            payload = analytics.export_payload(sentinel_db, domain="electronics", limit=7, offset=2)

        self.assertEqual(payload["dashboard_kpis"], {"total_reviews": 10})
        self.assertEqual(payload["aspect_leaderboard"], [{"aspect": "battery"}])
        self.assertEqual(payload["user_reviews"], {"total": 1, "rows": [{"review_id": 1}]})
        self.assertEqual(payload["generated_at"], fixed_now.isoformat())

        dashboard_kpis.assert_called_once_with(sentinel_db, None, None, "electronics")
        aspect_leaderboard.assert_called_once_with(sentinel_db, limit=25, domain="electronics")
        aspect_trends.assert_called_once_with(sentinel_db, interval="day", domain="electronics", limit=500)
        emerging_aspects.assert_called_once_with(sentinel_db, interval="day", lookback_buckets=7, domain="electronics")
        evidence_drilldown.assert_called_once_with(sentinel_db, aspect=None, sentiment=None, limit=50, domain="electronics")
        alerts.assert_called_once_with(sentinel_db, domain="electronics")
        impact_matrix.assert_called_once_with(sentinel_db, domain="electronics", limit=20)
        segment_drilldown.assert_called_once_with(sentinel_db, domain="electronics", limit=20)
        weekly_summary.assert_called_once_with(sentinel_db, domain="electronics")
        user_reviews_summary.assert_called_once_with(sentinel_db, domain="electronics")
        user_reviews_list.assert_called_once_with(sentinel_db, domain="electronics", limit=7, offset=2)


if __name__ == "__main__":
    unittest.main()
