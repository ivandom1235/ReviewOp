export default function EvidencePanel({ rows = [], isDark = false, title = "Evidence Drill-down" }) {
  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
      <h3 className="text-lg font-semibold">{title}</h3>
      <div className="mt-3 max-h-[420px] space-y-3 overflow-auto pr-1">
        {rows.length ? (
          rows.map((row, idx) => (
            <div key={`${row.review_id}-${idx}`} className={`rounded-xl border p-3 ${isDark ? "border-slate-700 bg-slate-900/60" : "border-slate-200 bg-slate-50"}`}>
              <p className="text-sm font-semibold">
                {row.aspect} <span className="text-xs font-normal opacity-80">({row.sentiment}, {row.origin})</span>
              </p>
              <p className="mt-1 text-sm opacity-90">{row.evidence || "-"}</p>
              {row.quarantine_status || Number.isFinite(Number(row.contradiction_score)) ? (
                <p className="mt-1 text-xs opacity-80">
                  {row.quarantine_status ? `Quarantine: ${row.quarantine_status}` : ""}
                  {row.quarantine_status && Number.isFinite(Number(row.contradiction_score)) ? " · " : ""}
                  {Number.isFinite(Number(row.contradiction_score)) ? `Contradiction: ${Number(row.contradiction_score).toFixed(2)}` : ""}
                </p>
              ) : null}
              <p className="mt-2 text-xs opacity-70">{row.review_text}</p>
            </div>
          ))
        ) : (
          <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No evidence rows yet.</p>
        )}
      </div>
    </div>
  );
}
