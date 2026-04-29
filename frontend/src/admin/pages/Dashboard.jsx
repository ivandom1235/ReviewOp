function Stat({ label, value, isDark }) {
  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
      <p className="text-xs uppercase tracking-[0.14em] opacity-70">{label}</p>
      <p className="mt-2 text-3xl font-bold">{value}</p>
    </div>
  );
}

export default function Dashboard({ kpis, alerts, impactRows = [], weeklySummary, isDark, onSeeMoreAlerts, onOpenGraph, onAlertClick }) {
  const criticalOpenAlerts = (alerts || []).filter(
    (a) => (a.severity || "").toLowerCase() === "critical" && (a.status || "open").toLowerCase() !== "cleared"
  ).length;
  const fixFirst = (impactRows || []).slice(0, 3);

  return (
    <section className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Stat label="Total Reviews" value={kpis?.total_reviews ?? "-"} isDark={isDark} />
        <Stat label="Open Critical Alerts" value={criticalOpenAlerts} isDark={isDark} />
        <Stat label="Most Negative Driver" value={kpis?.most_negative_aspect ?? "-"} isDark={isDark} />
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
            <p className="mt-2 text-xs opacity-70">Summary is evidence-gated and intended for operational review, not narrative reporting.</p>
          </div>
        ) : (
          <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No summary available.</p>
        )}
      </div>
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">Fix First Queue</h3>
        <div className="space-y-2">
          {fixFirst.length ? (
            fixFirst.map((r, idx) => (
              <div key={`${r.aspect}-${idx}`} className={`rounded-lg p-3 text-sm ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
                <p className="font-semibold">{idx + 1}. {r.aspect}</p>
                <p className="mt-1">Priority: {Math.round(Number(r.priority_score || 0))}</p>
                <p>Reason: High volume + high negative rate + recent growth</p>
                <p>Action: Investigate recent negative reviews</p>
              </div>
            ))
          ) : (
            <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No prioritized issues available.</p>
          )}
        </div>
      </div>

      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Active Alerts</h3>
          {alerts?.length > 5 && (
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
            alerts.slice(0, 5).map((a, idx) => (
              <button type="button" onClick={() => onAlertClick?.(a)} key={`${a.aspect}-${idx}`} className={`w-full rounded-lg p-3 text-left text-sm ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
                <div className="flex items-center justify-between gap-3">
                  <span className="font-semibold">[{a.severity}] {a.aspect}</span>
                  <span className="rounded-full bg-slate-200 px-2 py-0.5 text-[11px] text-slate-700 dark:bg-slate-800 dark:text-slate-200">{a.status || "Open"}</span>
                </div>
                <p className="mt-1">{a.message}</p>
              </button>
            ))
          ) : (
            <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No active alerts.</p>
          )}
        </div>
      </div>

      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-2 text-lg font-semibold">Mini Graph Preview</h3>
        <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
          Explore clustered relationships, edge strengths, and sentiment direction in Graph Intelligence.
        </p>
        <button type="button" onClick={onOpenGraph} className="mt-3 rounded-xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-950">
          Open Graph Intelligence
        </button>
      </div>
    </section>
  );
}
