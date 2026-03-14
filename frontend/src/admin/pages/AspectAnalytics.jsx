import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Line, Legend } from "recharts";
import EvidencePanel from "../components/EvidencePanel";
import AspectDetailPanel from "../components/AspectDetailPanel";

export default function AspectAnalytics({ trends = [], emerging = [], evidence = [], aspectDetail, weeklySummary, isDark = false }) {
  return (
    <section className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-2">
        <div className={`h-80 rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="mb-2 text-lg font-semibold">Aspect Frequency Trend</h3>
          <ResponsiveContainer width="100%" height="88%">
            <LineChart data={trends}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#334155" : "#e2e8f0"} />
              <XAxis dataKey="bucket" stroke={isDark ? "#94a3b8" : "#475569"} />
              <YAxis stroke={isDark ? "#94a3b8" : "#475569"} />
              <Tooltip />
              <Legend />
              <Line dataKey="mentions" stroke="#0ea5e9" strokeWidth={2} />
              <Line dataKey="negative_pct" stroke="#ef4444" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <AspectDetailPanel detail={aspectDetail} isDark={isDark} />
      </div>
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="text-lg font-semibold">Emerging Aspects</h3>
        <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {emerging.length ? emerging.map((e, idx) => (
            <div key={`${e.aspect}-${idx}`} className={`rounded-lg p-3 text-sm ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
              <p className="font-semibold">{e.aspect}</p>
              <p>Recent: {e.recent_mentions} | Baseline: {e.baseline_mentions}</p>
            </div>
          )) : <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No emerging aspects.</p>}
        </div>
      </div>
      <EvidencePanel rows={evidence} isDark={isDark} />
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-2 text-lg font-semibold">Recommended Actions</h3>
        {(weeklySummary?.action_recommendations || []).length ? (
          <ul className="space-y-2 text-sm">
            {weeklySummary.action_recommendations.map((item, idx) => (
              <li key={`${idx}-${item}`} className={`rounded-lg p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
                {item}
              </li>
            ))}
          </ul>
        ) : (
          <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No recommendations yet.</p>
        )}
      </div>
    </section>
  );
}
