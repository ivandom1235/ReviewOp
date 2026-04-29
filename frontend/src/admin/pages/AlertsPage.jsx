import { useState, useMemo } from "react";

export default function AlertsPage({ alerts = [], isDark, onAlertClick, onAlertClear }) {
  const [searchTerm, setSearchTerm] = useState("");
  const [severityFilter, setSeverityFilter] = useState("All");
  const [statusFilter, setStatusFilter] = useState("All");

  const filteredAlerts = useMemo(() => {
    return alerts.filter((a) => {
      const message = String(a?.message || "").toLowerCase();
      const aspect = String(a?.aspect || "").toLowerCase();
      const status = String(a?.status || "open").toLowerCase();
      const matchesSearch = 
        message.includes(searchTerm.toLowerCase()) || 
        aspect.includes(searchTerm.toLowerCase());
      const matchesSeverity = severityFilter === "All" || String(a?.severity || "").toLowerCase() === severityFilter.toLowerCase();
      const matchesStatus = statusFilter === "All" || status === statusFilter.toLowerCase();
      return matchesSearch && matchesSeverity && matchesStatus;
    });
  }, [alerts, searchTerm, severityFilter, statusFilter]);

  const summary = useMemo(() => {
    const critical = alerts.filter((a) => ["critical", "high"].includes(String(a?.severity || "").toLowerCase())).length;
    const warning = alerts.filter((a) => ["warning", "medium"].includes(String(a?.severity || "").toLowerCase())).length;
    const open = alerts.filter((a) => String(a?.status || "open").toLowerCase() === "open").length;
    return { critical, warning, open };
  }, [alerts]);
  const severityOptions = useMemo(() => {
    const values = Array.from(new Set((alerts || []).map((a) => String(a?.severity || "").toLowerCase()).filter(Boolean)));
    return values.length ? values : ["critical", "warning", "low"];
  }, [alerts]);

  const sortedAlerts = useMemo(() => {
    const rank = { critical: 3, high: 3, warning: 2, medium: 2, low: 1, info: 1 };
    return [...filteredAlerts].sort((a, b) => {
      const ar = rank[String(a?.severity || "").toLowerCase()] || 0;
      const br = rank[String(b?.severity || "").toLowerCase()] || 0;
      if (br !== ar) return br - ar;
      return Number(b?.priority_score || 0) - Number(a?.priority_score || 0);
    });
  }, [filteredAlerts]);

  function severityBadge(severity) {
    const s = String(severity || "").toLowerCase();
    if (["critical", "high"].includes(s)) return isDark ? "bg-red-500/20 text-red-300" : "bg-red-100 text-red-700";
    if (["warning", "medium"].includes(s)) return isDark ? "bg-amber-500/20 text-amber-300" : "bg-amber-100 text-amber-700";
    return isDark ? "bg-blue-500/20 text-blue-300" : "bg-blue-100 text-blue-700";
  }

  return (
    <section className="space-y-4">
      <div className={`rounded-2xl border p-6 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <h2 className="text-2xl font-bold">All System Alerts</h2>
          <div className="flex flex-wrap gap-3">
            <input 
              type="text" 
              placeholder="Search alerts..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`rounded-xl border px-4 py-2 w-64 ${isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-slate-50 text-slate-800"}`}
            />
            <select 
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className={`rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-slate-50 text-slate-800"}`}
            >
              <option value="All">All Severities</option>
              {severityOptions.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className={`rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-slate-50 text-slate-800"}`}
            >
              <option value="All">All Statuses</option>
              <option value="open">Open</option>
              <option value="acknowledged">Acknowledged</option>
              <option value="resolved">Resolved</option>
            </select>
          </div>
        </div>

        <div className="mb-5 grid gap-3 sm:grid-cols-3">
          <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs uppercase opacity-70">Critical</p><p className="text-2xl font-bold">{summary.critical}</p></div>
          <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs uppercase opacity-70">Warning</p><p className="text-2xl font-bold">{summary.warning}</p></div>
          <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs uppercase opacity-70">Open</p><p className="text-2xl font-bold">{summary.open}</p></div>
        </div>

        {sortedAlerts.length ? (
          <div className="space-y-3">
            {sortedAlerts.map((a, idx) => (
              <div key={a.id ?? `${a.aspect}-${idx}`} className={`rounded-xl border p-4 ${isDark ? "border-slate-800 bg-slate-900" : "border-slate-200 bg-slate-50"}`}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className={`rounded-full px-2 py-1 text-xs font-bold ${severityBadge(a.severity)}`}>{a.severity || "info"}</span>
                      <span className="text-sm font-semibold">{a.aspect || "Unknown aspect"}</span>
                      <span className={`rounded-full px-2 py-0.5 text-[11px] ${isDark ? "bg-slate-800 text-slate-300" : "bg-slate-200 text-slate-700"}`}>{a.status || "Open"}</span>
                    </div>
                    <p className="text-sm">{a.message}</p>
                    <p className={`text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                      Detected: {a.detected_at ? new Date(a.detected_at).toLocaleString() : "Unavailable"} | Priority: {a.priority_score ?? "Pending"}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={() => onAlertClick(a)} className="rounded-lg border border-indigo-300 px-3 py-1.5 text-sm font-semibold text-indigo-600 hover:bg-indigo-50 dark:border-indigo-700 dark:text-indigo-300 dark:hover:bg-indigo-950">
                      View details
                    </button>
                    <button onClick={() => onAlertClear?.(a)} className="rounded-lg border border-rose-300 px-3 py-1.5 text-sm font-semibold text-rose-600 hover:bg-rose-50 dark:border-rose-700 dark:text-rose-300 dark:hover:bg-rose-950">
                      Clear
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className={`grid h-[600px] place-items-center rounded-xl border ${isDark ? "border-slate-800 bg-slate-950 text-slate-400" : "border-slate-200 bg-slate-50 text-slate-500"}`}>
            No alerts match the current filters.
          </div>
        )}
      </div>
    </section>
  );
}
