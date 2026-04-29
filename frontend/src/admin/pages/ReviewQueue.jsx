import { useMemo, useState } from "react";

export default function ReviewQueue({ needsReviewRows = [], novelCandidateRows = [], isDark = false }) {
  const [tab, setTab] = useState("needs_review");
  const [dismissed, setDismissed] = useState(() => new Set());

  const lowConfidenceRows = useMemo(
    () =>
      (needsReviewRows || []).filter((r) => Number(r?.confidence ?? 1) < 0.6),
    [needsReviewRows]
  );

  const rows = useMemo(() => {
    if (tab === "novel_candidates") return novelCandidateRows || [];
    if (tab === "low_confidence") return lowConfidenceRows || [];
    return needsReviewRows || [];
  }, [tab, needsReviewRows, novelCandidateRows, lowConfidenceRows]);

  const visibleRows = useMemo(
    () => rows.filter((row, idx) => !dismissed.has(`${row.id ?? "row"}-${row.review_id ?? idx}`)),
    [rows, dismissed]
  );

  function resolveRow(row, idx, action) {
    const key = `${row.id ?? "row"}-${row.review_id ?? idx}`;
    setDismissed((prev) => new Set(prev).add(key));
    console.info(`[ReviewQueue] ${action}`, { key, tab, row });
  }

  return (
    <section className="space-y-4">
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h2 className="text-2xl font-bold">Review Queue</h2>
        <p className={`mt-1 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
          Triage uncertain and novel predictions before they are accepted into operations.
        </p>

        <div className="mt-4 flex flex-wrap gap-2">
          {[
            { key: "needs_review", label: "Needs Review", count: needsReviewRows.length },
            { key: "novel_candidates", label: "Novel Candidates", count: novelCandidateRows.length },
            { key: "low_confidence", label: "Low Confidence", count: lowConfidenceRows.length },
          ].map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => setTab(t.key)}
              className={`rounded-xl px-3 py-2 text-sm font-semibold ${
                tab === t.key
                  ? "bg-emerald-500 text-slate-950"
                  : isDark
                    ? "bg-slate-800 text-slate-200"
                    : "bg-slate-200 text-slate-700"
              }`}
            >
              {t.label} ({t.count})
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        {visibleRows.length ? (
          visibleRows.map((row, idx) => (
            <div key={`${row.id ?? "row"}-${row.review_id ?? idx}`} className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
              {tab === "novel_candidates" ? (
                <>
                  <p className="text-sm font-semibold">Novel Candidate: {row.aspect || "Unknown aspect"}</p>
                  <p className="mt-1 text-sm">Novelty Score: {Number(row.novelty_score || 0).toFixed(2)}</p>
                  <p className="text-sm">Evidence: {row.evidence || "No evidence recorded"}</p>
                  <p className="mt-1 text-sm">Suggested Mapping: delivery / packaging</p>
                  <p className={`mt-2 text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>Review: {row.review_text || "-"}</p>
                </>
              ) : (
                <>
                  <p className="text-sm font-semibold">Reason: {row.reason || "review_required"}</p>
                  <p className="mt-1 text-sm">Confidence: {(Number(row.confidence || 0) * 100).toFixed(1)}%</p>
                  <p className="text-sm">Evidence: {row.evidence || Number(row.ambiguity_score || 0).toFixed(2)}</p>
                  <p className={`mt-2 text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>Review: {row.review_text || "-"}</p>
                </>
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                <button type="button" onClick={() => resolveRow(row, idx, "approve")} className="rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-semibold text-slate-950">Approve</button>
                <button type="button" onClick={() => resolveRow(row, idx, "reject")} className="rounded-lg bg-rose-500 px-3 py-1.5 text-xs font-semibold text-white">Reject</button>
                <button type="button" onClick={() => resolveRow(row, idx, "merge")} className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${isDark ? "bg-slate-800 text-slate-200" : "bg-slate-200 text-slate-700"}`}>Merge</button>
                <button type="button" onClick={() => resolveRow(row, idx, "mark_known")} className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${isDark ? "bg-slate-800 text-slate-200" : "bg-slate-200 text-slate-700"}`}>Mark as known aspect</button>
              </div>
            </div>
          ))
        ) : (
          <div className={`grid h-40 place-items-center rounded-xl border ${isDark ? "border-slate-800 bg-slate-950 text-slate-400" : "border-slate-200 bg-slate-50 text-slate-500"}`}>
            No items in this queue.
          </div>
        )}
      </div>
    </section>
  );
}
