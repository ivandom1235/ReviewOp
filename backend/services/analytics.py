from __future__ import annotations

from services.analytics_alerts import alerts, clear_alert, sync_alerts
from services.analytics_exports import export_payload, export_pdf_bytes
from services.analytics_kpis import dashboard_kpis, overview
from services.analytics_operational import needs_review_queue, novel_candidates_queue
from services.analytics_segments import impact_matrix, segment_drilldown, weekly_summary
from services.analytics_user_reviews import user_reviews_list, user_reviews_summary
from services.analytics_aspects import (
    aspect_detail,
    aspect_leaderboard,
    aspect_sentiment_distribution,
    aspect_trends,
    evidence_drilldown,
    emerging_aspects,
    trends,
    top_aspects,
)
