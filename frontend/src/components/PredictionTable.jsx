const sentimentClass = {
  positive: "text-success",
  neutral: "text-warning",
  negative: "text-danger",
};

export default function PredictionTable({ predictions = [], isDark = false }) {
  if (!predictions.length) {
    return (
      <div className={`rounded-xl border p-4 text-sm ${isDark ? "border-slate-800 bg-slate-950 text-slate-400" : "border-slate-300 bg-slate-50 text-slate-500"}`}>
        No prediction rows yet.
      </div>
    );
  }

  return (
    <div className={`overflow-x-auto rounded-xl border ${isDark ? "border-slate-800 bg-slate-950" : "border-slate-300 bg-white"}`}>
      <table className="min-w-full text-sm">
        <thead className={`text-left text-xs uppercase tracking-[0.18em] ${isDark ? "bg-slate-900 text-slate-400" : "bg-slate-100 text-slate-600"}`}>
          <tr>
            <th className="px-4 py-3">Aspect</th>
            <th className="px-4 py-3">Sentiment</th>
            <th className="px-4 py-3">Confidence</th>
            <th className="px-4 py-3">Evidence Snippet</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((row, idx) => (
            <tr key={`${row.aspect_raw}-${idx}`} className={`border-t ${isDark ? "border-slate-800 text-slate-200" : "border-slate-200 text-slate-800"}`}>
              <td className="px-4 py-3">{row.aspect_cluster || row.aspect_raw}</td>
              <td className={`px-4 py-3 font-medium ${sentimentClass[row.sentiment] || "text-slate-200"}`}>
                {row.sentiment}
              </td>
              <td className="px-4 py-3">{(row.confidence * 100).toFixed(1)}%</td>
              <td className={`max-w-[520px] truncate px-4 py-3 ${isDark ? "text-slate-300" : "text-slate-600"}`}>
                {row.evidence_spans?.[0]?.snippet || "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
