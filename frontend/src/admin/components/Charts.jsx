import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  LineChart,
  Line,
  Legend,
} from "recharts";

export function SingleSentimentChart({ predictions = [], isDark = false }) {
  const counts = predictions.reduce(
    (acc, item) => {
      const key = (item.sentiment || "").toLowerCase();
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    },
    { positive: 0, neutral: 0, negative: 0 }
  );

  const data = [
    { sentiment: "Positive", count: counts.positive },
    { sentiment: "Neutral", count: counts.neutral },
    { sentiment: "Negative", count: counts.negative },
  ];

  return (
    <div className={`h-72 rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
      <h3 className={`mb-3 text-sm font-semibold uppercase tracking-[0.14em] ${isDark ? "text-slate-300" : "text-slate-700"}`}>
        Single Review Sentiment Mix
      </h3>
      <ResponsiveContainer width="100%" height="86%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#334155" : "#e2e8f0"} />
          <XAxis dataKey="sentiment" stroke={isDark ? "#94a3b8" : "#475569"} />
          <YAxis stroke={isDark ? "#94a3b8" : "#475569"} allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="#6366f1" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrendsChart({ data = [], isDark = false }) {
  const chartData = (data || []).map((item) => ({
    ...item,
    negative_pct: Number(item.negative_pct ?? 0),
    mentions: Number(item.mentions ?? 0),
  }));

  return (
    <div className={`h-72 rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
      <h3 className={`mb-3 text-sm font-semibold uppercase tracking-[0.14em] ${isDark ? "text-slate-300" : "text-slate-700"}`}>
        Batch Trends
      </h3>
      {chartData.length ? (
        <ResponsiveContainer width="100%" height="86%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#334155" : "#e2e8f0"} />
            <XAxis dataKey="bucket" stroke={isDark ? "#94a3b8" : "#475569"} />
            <YAxis yAxisId="left" stroke={isDark ? "#94a3b8" : "#475569"} />
            <YAxis yAxisId="right" orientation="right" stroke={isDark ? "#94a3b8" : "#475569"} />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="mentions" stroke="#0ea5e9" strokeWidth={2} />
            <Line yAxisId="right" type="monotone" dataKey="negative_pct" stroke="#f43f5e" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className={`grid h-[86%] place-items-center text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
          No batch trend data yet. Upload a CSV to populate this chart.
        </div>
      )}
    </div>
  );
}
