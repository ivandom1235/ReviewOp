import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Line } from "recharts";

export default function AspectDetailPanel({ detail, isDark = false }) {
  if (!detail) {
    return (
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>Select an aspect to view detail.</p>
      </div>
    );
  }

  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
      <h3 className="text-lg font-semibold">{detail.aspect}</h3>
      <p className="mt-2 text-sm">Frequency: {detail.frequency} | Explicit: {detail.explicit_count} | Implicit: {detail.implicit_count}</p>
      <p className="mt-1 text-sm">Sentiment: +{detail.positive} / ={detail.neutral} / -{detail.negative}</p>
      <div className="mt-4 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={detail.trend || []}>
            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#334155" : "#e2e8f0"} />
            <XAxis dataKey="bucket" stroke={isDark ? "#94a3b8" : "#475569"} />
            <YAxis stroke={isDark ? "#94a3b8" : "#475569"} />
            <Tooltip />
            <Line type="monotone" dataKey="mentions" stroke="#0ea5e9" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
