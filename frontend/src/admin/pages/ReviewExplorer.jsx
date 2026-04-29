import AspectGraphView from "../components/graph/AspectGraphView";

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
  onOpenGraph,
  onOpenAnalytics,
  isDark = false,
}) {
  const rows = (output?.predictions || []).map((p, idx) => ({
    id: `${idx}-${p.aspect_raw}`,
    aspect: p.aspect_cluster || p.aspect_raw,
    sentiment: p.sentiment,
    confidence: Number(p.confidence || 0),
    origin: p.origin || (p.is_implicit ? "implicit" : p.source || "explicit"),
    evidence: p.evidence_spans?.[0]?.snippet || "-",
  }));

  return (
    <section className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <form onSubmit={onSubmit} className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="mb-2 text-lg font-semibold">Single Review Inference</h3>
          <textarea value={reviewText} onChange={(e) => setReviewText(e.target.value)} required className={`h-28 w-full rounded-xl border p-3 ${isDark ? "border-slate-700 bg-slate-900" : "border-slate-200 bg-slate-50"}`} placeholder="Enter a review" />
          <button disabled={loading} className="mt-3 rounded-xl bg-indigo-600 px-4 py-2 font-semibold text-white">{loading ? "Processing..." : "Analyze Review"}</button>
        </form>

        <form onSubmit={onBatchSubmit} className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <h3 className="mb-2 text-lg font-semibold">Batch CSV Inference</h3>
          <input type="file" accept=".csv" onChange={(e) => setBatchFile(e.target.files?.[0] || null)} className={`w-full rounded-xl border p-3 ${isDark ? "border-slate-700 bg-slate-900" : "border-slate-200 bg-slate-50"}`} />
          <button disabled={loading || !batchFile} className="mt-3 rounded-xl bg-amber-500 px-4 py-2 font-semibold text-slate-900">{loading ? "Processing..." : "Run Batch Inference"}</button>
          {jobStatus ? (
            <div className={`mt-3 rounded-xl p-3 text-sm ${isDark ? "bg-slate-900 text-slate-200" : "bg-slate-50 text-slate-700"}`}>
              Job {jobStatus.job_id}: {jobStatus.status} | {jobStatus.processed || 0}/{jobStatus.total || 0} processed | failed: {jobStatus.failed || 0}
            </div>
          ) : null}
        </form>
      </div>

      {kpis ? (
        <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-xl font-semibold">Batch Results Snapshot</h3>
              <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>Start analysis here before opening corpus graph.</p>
            </div>
            <div className="flex gap-2">
              <button type="button" onClick={onOpenGraph} className="rounded-xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-950">Open Corpus Graph</button>
              <button type="button" onClick={onOpenAnalytics} className="rounded-xl bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950">Open Aspect Analytics</button>
            </div>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Reviews</p><p className="text-2xl font-bold">{kpis.total_reviews ?? 0}</p></div>
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Aspect Mentions</p><p className="text-2xl font-bold">{kpis.total_aspects ?? 0}</p></div>
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Most Negative Aspect</p><p className="text-2xl font-bold">{kpis.most_negative_aspect || "-"}</p></div>
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Negative %</p><p className="text-2xl font-bold">{kpis.negative_sentiment_pct ?? 0}%</p></div>
            <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}><p className="text-xs opacity-70">Emerging Issues</p><p className="text-2xl font-bold">{kpis.emerging_issues_count ?? 0}</p></div>
          </div>

        </div>
      ) : null}

      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">Single Review Result</h3>
        <div className="grid gap-3 md:grid-cols-2">
          {rows.length ? rows.map((r) => (
            <div key={r.id} className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
              <p className="font-semibold">Aspect: {r.aspect}</p>
              <p className="text-sm">Sentiment: {r.sentiment}</p>
              <p className="text-sm">Type: {r.origin}</p>
              <p className="text-sm">Confidence: {(Number(r.confidence || 0) * 100).toFixed(1)}%</p>
              <p className="mt-1 text-sm">Evidence: {r.evidence}</p>
            </div>
          )) : <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>Analyze a review to see extracted aspects.</p>}
        </div>
      </div>

      <AspectGraphView graph={reviewGraph} scope="single_review" isDark={isDark} reviewText={reviewText} emptyMessage="Run single review inference to view explanation graph." />
    </section>
  );
}
