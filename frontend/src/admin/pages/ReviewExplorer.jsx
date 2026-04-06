import { useMemo } from "react";
import DataGridTable from "../components/DataGridTable";
import AspectGraphView from "../components/graph/AspectGraphView";

const IMPACT_COLUMNS = [
  { field: "aspect", headerName: "Aspect", flex: 1, minWidth: 130 },
  { field: "priority_score", headerName: "Priority", type: "number", flex: 0.6, minWidth: 90 },
  { field: "volume", headerName: "Vol", type: "number", flex: 0.5, minWidth: 70 },
  { field: "negative_rate", headerName: "Neg Rate", type: "number", flex: 0.6, minWidth: 90 },
  { field: "growth_pct", headerName: "Growth %", type: "number", flex: 0.6, minWidth: 90 },
  { field: "action_tier", headerName: "Tier", flex: 0.5, minWidth: 80 },
];

const SEGMENT_COLUMNS = [
  { field: "segment_type", headerName: "Type", flex: 0.6, minWidth: 90 },
  { field: "segment_value", headerName: "Segment", flex: 1, minWidth: 130 },
  { field: "negative_pct", headerName: "Neg %", type: "number", flex: 0.6, minWidth: 90 },
  { field: "review_count", headerName: "Reviews", type: "number", flex: 0.5, minWidth: 80 },
  { field: "top_negative_aspect", headerName: "Top Negative", flex: 0.9, minWidth: 120 },
];

const LEADERBOARD_COLUMNS = [
  { field: "aspect", headerName: "Aspect", flex: 1, minWidth: 130 },
  { field: "sample_size", headerName: "N", type: "number", flex: 0.4, minWidth: 60 },
  { field: "mentions_per_100_reviews", headerName: "Mentions/100", type: "number", flex: 0.7, minWidth: 100 },
  { field: "negative_pct", headerName: "Neg %", type: "number", flex: 0.5, minWidth: 80 },
  { field: "negative_ci_low", headerName: "CI Low", type: "number", flex: 0.5, minWidth: 80 },
  { field: "negative_ci_high", headerName: "CI High", type: "number", flex: 0.5, minWidth: 80 },
];

const EXTRACTED_COLUMNS = [
  { field: "aspect", headerName: "Aspect", flex: 1.1, minWidth: 150 },
  { field: "sentiment", headerName: "Sentiment", flex: 0.7, minWidth: 110 },
  { field: "confidence", headerName: "Confidence", flex: 0.7, minWidth: 110, valueFormatter: (v) => `${(Number(v || 0) * 100).toFixed(1)}%` },
  { field: "origin", headerName: "Origin", flex: 0.7, minWidth: 110 },
  { field: "decision", headerName: "Decision", flex: 0.7, minWidth: 110 },
  { field: "routing", headerName: "Routing", flex: 0.7, minWidth: 110 },
  { field: "ambiguity_score", headerName: "Ambiguity", flex: 0.7, minWidth: 100, valueFormatter: (v) => Number(v || 0).toFixed(2) },
  { field: "novelty_score", headerName: "Novelty", flex: 0.7, minWidth: 100, valueFormatter: (v) => Number(v || 0).toFixed(2) },
  { field: "evidence", headerName: "Evidence", flex: 1.5, minWidth: 200 },
];

const ACCEPTED_COLUMNS = [
  { field: "aspect", headerName: "Aspect", flex: 1, minWidth: 120 },
  { field: "sentiment", headerName: "Sentiment", flex: 0.8, minWidth: 100 },
  { field: "confidence", headerName: "Confidence", flex: 0.8, minWidth: 100, valueFormatter: (v) => `${(Number(v || 0) * 100).toFixed(1)}%` },
  { field: "routing", headerName: "Routing", flex: 0.7, minWidth: 90 },
];

const ABSTAIN_COLUMNS = [
  { field: "reason", headerName: "Reason", flex: 1.2, minWidth: 140 },
  { field: "confidence", headerName: "Confidence", flex: 0.8, minWidth: 100, valueFormatter: (v) => Number(v || 0).toFixed(2) },
  { field: "ambiguity_score", headerName: "Ambiguity", flex: 0.8, minWidth: 100, valueFormatter: (v) => Number(v || 0).toFixed(2) },
];

const NOVEL_COLUMNS = [
  { field: "aspect", headerName: "Aspect", flex: 1.2, minWidth: 140 },
  { field: "novelty_score", headerName: "Novelty", flex: 0.8, minWidth: 100, valueFormatter: (v) => Number(v || 0).toFixed(2) },
  { field: "confidence", headerName: "Confidence", flex: 0.8, minWidth: 100, valueFormatter: (v) => Number(v || 0).toFixed(2) },
];

export default function ReviewExplorer({
  reviewText,
  setReviewText,
  onSubmit,
  loading,
  output,
  reviewGraph,
  batchFile,
  setBatchFile,
  onBatchSubmit,
  jobStatus,
  kpis,
  leaderboardRows = [],
  impactRows = [],
  segmentRows = [],
  weeklySummary,
  alerts = [],
  evidenceRows = [],
  onOpenGraph,
  onOpenAnalytics,
  isDark = false,
}) {
  const rows = useMemo(
    () =>
      (output?.predictions || []).map((p, idx) => ({
        id: `${idx}-${p.aspect_raw || p.aspect_cluster || "row"}`,
        aspect: p.aspect_cluster || p.aspect_raw,
        sentiment: p.sentiment,
        confidence: Number(p.confidence || 0),
        origin: p.origin || (p.is_implicit ? "implicit" : p.source || "explicit"),
        evidence: p.evidence_spans?.[0]?.snippet || "-",
        decision: p.decision || "single_label",
        routing: p.routing || "known",
        ambiguity_score: Number(p.ambiguity_score || 0),
        novelty_score: Number(p.novelty_score || 0),
      })),
    [output?.predictions]
  );

  const acceptedRows = useMemo(
    () =>
      (output?.accepted_predictions || []).map((p, idx) => ({
        id: `accepted-${p.aspect || "a"}-${idx}`,
        aspect: p.aspect,
        sentiment: p.sentiment,
        confidence: Number(p.confidence || 0),
        routing: p.routing || "known",
        evidence: p.evidence || "-",
      })),
    [output?.accepted_predictions]
  );

  const abstainedRows = useMemo(
    () =>
      (output?.abstained_predictions || []).map((p, idx) => ({
        id: `abstained-${p.reason || "r"}-${idx}`,
        reason: p.reason || "low_selective_confidence",
        confidence: Number(p.confidence || 0),
        ambiguity_score: Number(p.ambiguity_score || 0),
      })),
    [output?.abstained_predictions]
  );

  const novelRows = useMemo(
    () =>
      (output?.novel_candidates || []).map((p, idx) => ({
        id: `novel-${p.aspect || "n"}-${idx}`,
        aspect: p.aspect,
        novelty_score: Number(p.novelty_score || 0),
        confidence: Number(p.confidence || 0),
      })),
    [output?.novel_candidates]
  );

  const impactRowsWithId = useMemo(() => impactRows.map((r, i) => ({ id: `${r.aspect || "aspect"}-${i}`, ...r })), [impactRows]);
  const segmentRowsWithId = useMemo(
    () => segmentRows.map((r, i) => ({ id: `${r.segment_type || "seg"}-${r.segment_value || "value"}-${i}`, ...r })),
    [segmentRows]
  );

  return (
    <section className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <form onSubmit={onSubmit} className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-2">Single Review Inference</h3>
          <textarea value={reviewText} onChange={(e) => setReviewText(e.target.value)} required className="app-input h-28 p-3" placeholder="Enter a review" />
          <button disabled={loading} className="app-btn app-btn-primary mt-3">{loading ? "Processing..." : "Analyze Review"}</button>
        </form>

        <form onSubmit={onBatchSubmit} className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-2">Batch CSV Inference</h3>
          <input type="file" accept=".csv" onChange={(e) => setBatchFile(e.target.files?.[0] || null)} className="app-input w-full p-3" />
          <button disabled={loading || !batchFile} className="app-btn app-btn-secondary mt-3">{loading ? "Processing..." : "Run Batch Inference"}</button>
          {jobStatus ? (
            <div className={`mt-3 rounded-xl p-3 text-sm ${isDark ? "bg-slate-900 text-slate-200" : "bg-slate-50 text-slate-700"}`}>
              Job {jobStatus.job_id}: {jobStatus.status} | {jobStatus.processed || 0}/{jobStatus.total || 0} processed | failed: {jobStatus.failed || 0}
            </div>
          ) : null}
        </form>
      </div>

      {kpis ? (
        <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="app-section-title text-xl">Batch Results Snapshot</h3>
              <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>Start analysis here before opening corpus graph.</p>
            </div>
            <div className="flex gap-2">
              <button type="button" onClick={onOpenGraph} className="app-btn app-btn-secondary text-sm">Open Corpus Graph</button>
              <button type="button" onClick={onOpenAnalytics} className="app-btn app-btn-secondary text-sm">Open Aspect Analytics</button>
            </div>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Reviews</p><p className="text-2xl font-bold">{kpis.total_reviews ?? 0}</p></div>
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Aspect Mentions</p><p className="text-2xl font-bold">{kpis.total_aspects ?? 0}</p></div>
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Most Negative Aspect</p><p className="text-2xl font-bold">{kpis.most_negative_aspect || "-"}</p></div>
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Negative %</p><p className="text-2xl font-bold">{kpis.negative_sentiment_pct ?? 0}%</p></div>
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Emerging Issues</p><p className="text-2xl font-bold">{kpis.emerging_issues_count ?? 0}</p></div>
          </div>

          {weeklySummary ? (
            <div className={`mt-4 rounded-xl p-3 text-sm ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
              <p className="font-semibold">Weekly Brief: {weeklySummary.period_label}</p>
              <p className="mt-1">Top drivers: {(weeklySummary.top_drivers || []).join(", ") || "-"}</p>
              <p>Biggest increase: {weeklySummary.biggest_increase_aspect || "-"} ({Number(weeklySummary.biggest_increase_pct || 0).toFixed(1)}%)</p>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-3">Fix First Queue (Impact Matrix)</h3>
          <DataGridTable isDark={isDark} height={280} rows={impactRowsWithId} columns={IMPACT_COLUMNS} />
        </div>
        <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-3">Segment Risk Drilldown</h3>
          <DataGridTable isDark={isDark} height={280} rows={segmentRowsWithId} columns={SEGMENT_COLUMNS} />
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-3">Batch QA + Distribution</h3>
          <DataGridTable isDark={isDark} height={260} rows={leaderboardRows.slice(0, 12)} columns={LEADERBOARD_COLUMNS} />
        </div>
        <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-3">Alert Feed + Evidence Preview</h3>
          <div className="max-h-[250px] space-y-2 overflow-auto pr-1">
            {alerts.slice(0, 6).map((a, idx) => (
              <div key={`${a.id || a.aspect || "alert"}-${idx}`} className={`rounded-lg p-2 text-sm ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
                [{a.severity}] {a.message}
              </div>
            ))}
            {evidenceRows.slice(0, 4).map((r, idx) => (
              <div key={`${r.review_id || "review"}-${r.aspect || "aspect"}-${idx}`} className={`rounded-lg p-2 text-xs ${isDark ? "bg-slate-900 text-slate-300" : "bg-slate-50 text-slate-700"}`}>
                <span className="font-semibold">{r.aspect}</span> ({r.sentiment}): {r.evidence || "-"}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="app-section-title mb-3">Extracted Aspects</h3>
        <DataGridTable isDark={isDark} height={300} rows={rows} columns={EXTRACTED_COLUMNS} />
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-3">Accepted Predictions</h3>
          <DataGridTable isDark={isDark} height={240} rows={acceptedRows} columns={ACCEPTED_COLUMNS} />
        </div>
        <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-3">Abstained Cases</h3>
          <DataGridTable isDark={isDark} height={240} rows={abstainedRows} columns={ABSTAIN_COLUMNS} />
        </div>
        <div className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="app-section-title mb-3">Novel Candidates</h3>
          <DataGridTable isDark={isDark} height={240} rows={novelRows} columns={NOVEL_COLUMNS} />
        </div>
      </div>

      <AspectGraphView graph={reviewGraph} scope="single_review" isDark={isDark} reviewText={reviewText} emptyMessage="Run single review inference to view explanation graph." />
    </section>
  );
}
