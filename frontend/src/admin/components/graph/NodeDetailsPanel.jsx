function metricLabel(label, value) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-medium text-inherit">{value ?? "-"}</p>
    </div>
  );
}

export default function NodeDetailsPanel({ node, scope = "batch", isDark = false }) {
  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-500">Node Details</p>
          <h4 className="mt-1 text-xl font-semibold">{node?.label || "Select a node"}</h4>
        </div>
        {node?.sentiment || node?.dominant_sentiment ? (
          <span className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] ${
            isDark ? "bg-slate-800 text-slate-200" : "bg-slate-100 text-slate-700"
          }`}>
            {node.sentiment || node.dominant_sentiment}
          </span>
        ) : null}
      </div>

      {node ? (
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          {scope === "single_review"
            ? metricLabel("Confidence", node.confidence ? `${(Number(node.confidence) * 100).toFixed(1)}%` : "-")
            : metricLabel("Frequency", node.frequency)}
          {scope === "single_review"
            ? metricLabel("Origin", node.origin)
            : metricLabel("Avg Sentiment", Number(node.avg_sentiment ?? 0).toFixed(2))}
          {scope === "single_review"
            ? metricLabel("Explicit Count", node.explicit_count)
            : metricLabel("Negative Ratio", `${(Number(node.negative_ratio ?? 0) * 100).toFixed(1)}%`)}
          {scope === "single_review"
            ? metricLabel("Implicit Count", node.implicit_count)
            : metricLabel("Dominant", node.dominant_sentiment)}
          {scope === "batch" ? metricLabel("Explicit Count", node.explicit_count) : null}
          {scope === "batch" ? metricLabel("Implicit Count", node.implicit_count) : null}
        </div>
      ) : (
        <p className={`mt-4 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
          Click a graph node to inspect its aspect statistics and evidence metadata.
        </p>
      )}

      {node?.evidence ? (
        <div className={`mt-4 rounded-xl p-3 text-sm ${isDark ? "bg-slate-900 text-slate-200" : "bg-slate-50 text-slate-700"}`}>
          {node.evidence}
        </div>
      ) : null}
    </div>
  );
}
