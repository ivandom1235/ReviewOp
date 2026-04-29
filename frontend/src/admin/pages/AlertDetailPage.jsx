export default function AlertDetailPage({ alert, isDark, onBack, onOpenGraphNode }) {
  if (!alert) return null;
  const severity = String(alert.severity || "").toLowerCase();

  return (
    <section className="space-y-6">
      <button type="button" onClick={onBack} className="flex items-center gap-2 text-sm font-semibold text-indigo-500 hover:text-indigo-400">
        {"\u2190"} Back to Alerts
      </button>

      <div className={`rounded-2xl border p-8 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-3xl font-bold">{alert.aspect}</h2>
          <span
            className={`rounded-xl border px-4 py-2 text-sm font-bold ${
              severity === "critical"
                ? "border-red-500/20 bg-red-500/10 text-red-500"
                : severity === "warning"
                  ? "border-amber-500/20 bg-amber-500/10 text-amber-500"
                  : "border-blue-500/20 bg-blue-500/10 text-blue-500"
            }`}
          >
            {alert.severity}
          </span>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="space-y-4">
            <div>
              <p className="mb-1 text-xs uppercase tracking-[0.14em] opacity-60">Message</p>
              <p className="text-lg leading-relaxed">{alert.message}</p>
            </div>
            <div>
              <p className="mb-1 text-xs uppercase tracking-[0.14em] opacity-60">Detected At</p>
              <p className="text-sm">{alert.detected_at ? new Date(alert.detected_at).toLocaleString() : "Unavailable"}</p>
            </div>
            <div>
              <p className="mb-1 text-xs uppercase tracking-[0.14em] opacity-60">Related Evidence</p>
              <p className="text-sm">{alert.evidence || alert.example_review || "No direct evidence snippet available."}</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className={`rounded-xl p-4 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
              <p className="mb-2 text-xs uppercase tracking-[0.14em] opacity-60">Contextual Details</p>
              <ul className="space-y-2 text-sm">
                <li>{"\u2022"} Aspect: <span className="font-semibold">{alert.aspect}</span></li>
                <li>{"\u2022"} Status: {alert.status || "Open"}</li>
                <li>{"\u2022"} Impact Score: {alert.priority_score ?? "Pending"}</li>
                <li>{"\u2022"} Trend Change: {alert.trend_change_pct != null ? `${Number(alert.trend_change_pct).toFixed(1)}%` : "Unavailable"}</li>
                <li>{"\u2022"} Affected Segment: {alert.segment || alert.segment_value || "Unavailable"}</li>
                <li>{"\u2022"} Product: {alert.product_id || "Unavailable"}</li>
              </ul>
            </div>
            <div className={`rounded-xl p-4 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
              <p className="mb-2 text-xs uppercase tracking-[0.14em] opacity-60">Suggested Admin Action</p>
              <p className="text-sm">
                {alert.suggested_action || `Prioritize ${alert.aspect || "this aspect"} in triage, confirm evidence, then assign owner and ETA.`}
              </p>
            </div>
            <button type="button" onClick={() => onOpenGraphNode?.(alert)} className="rounded-xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-950">
              Open Related Graph Node
            </button>
          </div>
        </div>

        <div className="mt-8 border-t border-slate-800/50 pt-8">
          <h3 className="mb-4 text-lg font-semibold">Recommended Next Steps</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className={`rounded-xl p-4 ${isDark ? "border border-emerald-500/10 bg-emerald-500/5" : "border border-emerald-100 bg-emerald-50"}`}>
              <p className="mb-1 font-bold text-emerald-500">Investigation</p>
              <p className="text-sm opacity-80">Review recent logs and evidence related to {alert.aspect} in the Evidence Panel.</p>
            </div>
            <div className={`rounded-xl p-4 ${isDark ? "border border-indigo-500/10 bg-indigo-500/5" : "border border-indigo-100 bg-indigo-50"}`}>
              <p className="mb-1 font-bold text-indigo-500">Action</p>
              <p className="text-sm opacity-80">Capture ownership, acknowledge the issue, and move it toward resolution.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
