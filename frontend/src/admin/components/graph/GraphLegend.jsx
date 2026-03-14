const sentimentItems = [
  { label: "Positive", color: "#22c55e" },
  { label: "Neutral", color: "#94a3b8" },
  { label: "Negative", color: "#ef4444" },
];

export default function GraphLegend({ scope = "batch", isDark = false }) {
  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#08101f]" : "border-slate-200 bg-slate-50"}`}>
      <div className="flex flex-wrap items-center gap-5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-500">Legend</p>
          <p className={`mt-1 text-sm ${isDark ? "text-slate-300" : "text-slate-700"}`}>
            {scope === "single_review"
              ? "Directed review path graph ordered by evidence position."
              : "Undirected corpus co-occurrence graph aggregated across reviews."}
          </p>
        </div>

        <div className="flex flex-wrap gap-4">
          {sentimentItems.map((item) => (
            <div key={item.label} className="flex items-center gap-2 text-sm">
              <span className="h-3 w-3 rounded-full" style={{ backgroundColor: item.color }} />
              <span>{item.label}</span>
            </div>
          ))}
        </div>

        <div className={`text-sm ${isDark ? "text-slate-400" : "text-slate-600"}`}>
          {scope === "single_review" ? "Node size: near-constant / confidence" : "Node size: review frequency"}
        </div>

        <div className={`text-sm ${isDark ? "text-slate-400" : "text-slate-600"}`}>
          {scope === "single_review" ? "Edge arrows: on" : "Edge arrows: off"}
        </div>
      </div>
    </div>
  );
}
