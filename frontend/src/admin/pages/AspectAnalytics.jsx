import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Line, Legend, PieChart, Pie, Cell, BarChart, Bar } from "recharts";
import EvidencePanel from "../components/EvidencePanel";
import AspectDetailPanel from "../components/AspectDetailPanel";

export default function AspectAnalytics({ trends = [], emerging = [], evidence = [], aspectDetail, weeklySummary, isDark = false }) {
  const trendMentions = (trends || []).reduce((sum, t) => sum + Number(t.mentions || 0), 0);
  const trendNegRatio = trendMentions > 0 ? (trends || []).reduce((sum, t) => sum + (Number(t.mentions || 0) * Number(t.negative_pct || 0)), 0) / trendMentions : 0;

  const mentions = Number(aspectDetail?.frequency || 0);
  const positive = Number(aspectDetail?.positive || 0);
  const neutral = Number(aspectDetail?.neutral || 0);
  const negative = Number(aspectDetail?.negative || 0);
  const fallbackNegative = Math.round((trendNegRatio || 0) * Math.max(mentions || trendMentions || 1, 1));
  const fallbackPositive = Math.max((mentions || trendMentions || 1) - fallbackNegative, 0);
  const safePositive = positive + neutral + negative > 0 ? positive : fallbackPositive;
  const safeNeutral = positive + neutral + negative > 0 ? neutral : 0;
  const safeNegative = positive + neutral + negative > 0 ? negative : fallbackNegative;
  const totalSentiment = Math.max(safePositive + safeNeutral + safeNegative, 1);
  const negativePct = ((safeNegative / totalSentiment) * 100).toFixed(1);
  const implicitPct = (((Number(aspectDetail?.implicit_count || 0)) / Math.max(mentions, 1)) * 100).toFixed(1);
  const trendChange = Number(weeklySummary?.biggest_increase_pct || 0).toFixed(1);

  const sentimentData = [
    { name: "Positive", value: safePositive, color: "#22c55e" },
    { name: "Neutral", value: safeNeutral, color: "#94a3b8" },
    { name: "Negative", value: safeNegative, color: "#ef4444" },
  ];

  const segmentComparison = (aspectDetail?.segment_breakdown || aspectDetail?.connected_aspects || []).slice(0, 5).map((s, idx) => ({
    key: `${s.segment || s.segment_value || "segment"}-${idx}`,
    segment: s.segment || s.segment_value || s.aspect || `Segment ${idx + 1}`,
    negative_pct: Number(s.negative_pct || s.weight || 0),
  }));

  return (
    <section className="space-y-4">
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">Aspect Health Summary</h3>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs uppercase opacity-70">Mentions</p><p className="text-2xl font-bold">{mentions}</p></div>
          <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs uppercase opacity-70">Negative %</p><p className="text-2xl font-bold">{negativePct}%</p></div>
          <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs uppercase opacity-70">Trend Change</p><p className="text-2xl font-bold">{trendChange}%</p></div>
          <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs uppercase opacity-70">Implicit %</p><p className="text-2xl font-bold">{implicitPct}%</p></div>
          <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs uppercase opacity-70">Priority Score</p><p className="text-2xl font-bold">{Math.round((Number(negativePct) * 0.6) + (Number(trendChange) * 0.4))}</p></div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className={`h-80 rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="mb-2 text-lg font-semibold">Visual Summary: Trend</h3>
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
        <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="mb-2 text-lg font-semibold">Visual Summary: Sentiment Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={sentimentData} dataKey="value" nameKey="name" outerRadius={90}>
                  {sentimentData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2">
            <h4 className="mb-2 text-sm font-semibold opacity-80">Segment Comparison</h4>
            <div className="h-36">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={segmentComparison}>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#334155" : "#e2e8f0"} />
                  <XAxis dataKey="segment" hide />
                  <YAxis stroke={isDark ? "#94a3b8" : "#475569"} />
                  <Tooltip />
                  <Bar dataKey="negative_pct" fill="#f97316" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <AspectDetailPanel detail={aspectDetail} isDark={isDark} />

      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="text-lg font-semibold">Emerging Aspects</h3>
        <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {emerging.length ? emerging.map((e, idx) => (
            <div key={`${e.aspect}-${idx}`} className={`rounded-lg p-3 text-sm ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
              <p className="font-semibold">{e.aspect}</p>
              <p className="mt-1 font-medium text-amber-500">+{Number(e.growth_pct || ((Number(e.recent_mentions || 0) - Number(e.baseline_mentions || 0)) * 100) / Math.max(Number(e.baseline_mentions || 1), 1)).toFixed(0)}% increase</p>
              <p>Recent mentions: {e.recent_mentions}</p>
              <p>Baseline: {e.baseline_mentions}</p>
              <p>Risk: {Number(e.recent_mentions || 0) > Number(e.baseline_mentions || 0) ? "High" : "Moderate"}</p>
            </div>
          )) : <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No emerging aspects.</p>}
        </div>
      </div>

      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">Evidence Strip</h3>
        <EvidencePanel rows={evidence} isDark={isDark} />
      </div>

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
