import DataGridTable from "../components/DataGridTable";

function Stat({ label, value, isDark }) {
  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
      <p className="text-xs uppercase tracking-[0.14em] opacity-70">{label}</p>
      <p className="mt-2 text-3xl font-bold">{value}</p>
    </div>
  );
}

export default function Dashboard({ kpis, alerts, leaderboardRows, impactRows = [], segmentRows = [], weeklySummary, isDark, onSeeMoreAlerts }) {
  return (
    <section className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <Stat label="Total Reviews" value={kpis?.total_reviews ?? "-"} isDark={isDark} />
        <Stat label="Total Aspects" value={kpis?.total_aspects ?? "-"} isDark={isDark} />
        <Stat label="Most Negative" value={kpis?.most_negative_aspect ?? "-"} isDark={isDark} />
        <Stat label="% Negative" value={kpis ? `${kpis.negative_sentiment_pct}%` : "-"} isDark={isDark} />
        <Stat label="Emerging Issues" value={kpis?.emerging_issues_count ?? 0} isDark={isDark} />
      </div>
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-2 text-lg font-semibold">Weekly Summary</h3>
        {weeklySummary ? (
          <div className={`rounded-lg p-3 text-sm ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
            <p className="font-semibold">{weeklySummary.period_label}</p>
            <p className="mt-1">Top Drivers: {(weeklySummary.top_drivers || []).join(", ") || "-"}</p>
            <p>Biggest Increase: {weeklySummary.biggest_increase_aspect || "-"} ({Number(weeklySummary.biggest_increase_pct || 0).toFixed(1)}%)</p>
            <p>Emerging Issues: {weeklySummary.emerging_count || 0}</p>
          </div>
        ) : (
          <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No summary available.</p>
        )}
      </div>
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Alerts</h3>
          {alerts?.length > 10 && (
            <button
              onClick={onSeeMoreAlerts}
              className="text-sm font-semibold text-indigo-500 hover:text-indigo-400"
            >
              See More
            </button>
          )}
        </div>
        <div className="space-y-2">
          {(alerts || []).length ? (
            alerts.slice(0, 10).map((a, idx) => (
              <div key={`${a.aspect}-${idx}`} className={`rounded-lg p-3 text-sm ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
                [{a.severity}] {a.message}
              </div>
            ))
          ) : (
            <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No active alerts.</p>
          )}
        </div>
      </div>
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">Aspect Leaderboard</h3>
        <DataGridTable
          isDark={isDark}
          height={420}
          rows={leaderboardRows}
          columns={[
            { field: "aspect", headerName: "Aspect", flex: 1.1, minWidth: 160 },
            { field: "frequency", headerName: "Freq", type: "number", flex: 0.5, minWidth: 80 },
            { field: "sample_size", headerName: "N", type: "number", flex: 0.4, minWidth: 70 },
            { field: "mentions_per_100_reviews", headerName: "Mentions/100", type: "number", flex: 0.7, minWidth: 110 },
            { field: "positive_pct", headerName: "Positive %", type: "number", flex: 0.6, minWidth: 100 },
            { field: "neutral_pct", headerName: "Neutral %", type: "number", flex: 0.6, minWidth: 100 },
            { field: "negative_pct", headerName: "Negative %", type: "number", flex: 0.6, minWidth: 100 },
            { field: "negative_ci_low", headerName: "Neg CI Low", type: "number", flex: 0.6, minWidth: 100 },
            { field: "negative_ci_high", headerName: "Neg CI High", type: "number", flex: 0.6, minWidth: 100 },
            { field: "net_sentiment", headerName: "Net Sent", type: "number", flex: 0.6, minWidth: 100 },
            { field: "change_vs_previous_period", headerName: "Change %", type: "number", flex: 0.6, minWidth: 100 },
            { field: "change_7d_vs_prev_7d", headerName: "7d vs prev 7d %", type: "number", flex: 0.8, minWidth: 130 },
            { field: "implicit_pct", headerName: "Implicit %", type: "number", flex: 0.6, minWidth: 100 },
          ]}
        />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="mb-3 text-lg font-semibold">Impact Matrix (Fix First)</h3>
          <DataGridTable
            isDark={isDark}
            height={320}
            rows={impactRows.map((r, i) => ({ id: `${r.aspect}-${i}`, ...r }))}
            columns={[
              { field: "aspect", headerName: "Aspect", flex: 1, minWidth: 140 },
              { field: "volume", headerName: "Volume", type: "number", flex: 0.5, minWidth: 80 },
              { field: "negative_rate", headerName: "Neg Rate", type: "number", flex: 0.6, minWidth: 90 },
              { field: "growth_pct", headerName: "Growth %", type: "number", flex: 0.6, minWidth: 90 },
              { field: "priority_score", headerName: "Priority", type: "number", flex: 0.6, minWidth: 90 },
              { field: "action_tier", headerName: "Tier", flex: 0.6, minWidth: 80 },
            ]}
          />
        </div>
        <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="mb-3 text-lg font-semibold">Segment Drilldown</h3>
          <DataGridTable
            isDark={isDark}
            height={320}
            rows={segmentRows.map((r, i) => ({ id: `${r.segment_type}-${r.segment_value}-${i}`, ...r }))}
            columns={[
              { field: "segment_type", headerName: "Segment Type", flex: 0.7, minWidth: 100 },
              { field: "segment_value", headerName: "Segment", flex: 1, minWidth: 120 },
              { field: "review_count", headerName: "Reviews", type: "number", flex: 0.5, minWidth: 80 },
              { field: "mention_count", headerName: "Mentions", type: "number", flex: 0.5, minWidth: 80 },
              { field: "negative_pct", headerName: "Negative %", type: "number", flex: 0.6, minWidth: 90 },
              { field: "top_negative_aspect", headerName: "Top Negative Aspect", flex: 1, minWidth: 140 },
            ]}
          />
        </div>
      </div>
    </section>
  );
}
