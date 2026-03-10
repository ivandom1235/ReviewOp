import { useEffect, useMemo, useState } from "react";
import {
  inferSingleReview,
  inferBatchCsv,
  getJob,
  getOverview,
  getTopAspects,
  getAspectSentimentDistribution,
  getTrends,
  getKgCentrality,
  rebuildKg,
  getReviewGraph,
  getBatchAspectGraph,
} from "./api/client";
import DataGridTable from "./components/DataGridTable";
import { SingleSentimentChart, TrendsChart } from "./components/Charts";
import AspectGraphView from "./components/graph/AspectGraphView";
import GraphModeToggle from "./components/graph/GraphModeToggle";

function short(n) {
  if (n === null || n === undefined) return "-";
  return new Intl.NumberFormat().format(n);
}

function buildSingleAnalytics(singleOut) {
  const predictions = singleOut?.predictions || [];
  const sentiments = { positive: 0, neutral: 0, negative: 0 };
  const byAspect = new Map();
  let confSum = 0;

  predictions.forEach((p) => {
    const sent = (p.sentiment || "").toLowerCase();
    if (sentiments[sent] !== undefined) sentiments[sent] += 1;
    confSum += Number(p.confidence || 0);

    const key = p.aspect_cluster || p.aspect_raw || "unknown";
    if (!byAspect.has(key)) {
      byAspect.set(key, { aspect: key, count: 0, positive: 0, neutral: 0, negative: 0 });
    }
    const row = byAspect.get(key);
    row.count += 1;
    if (row[sent] !== undefined) row[sent] += 1;
  });

  const topAspects = Array.from(byAspect.values())
    .map((r) => ({ aspect: r.aspect, count: r.count }))
    .sort((a, b) => b.count - a.count);

  const aspectDist = Array.from(byAspect.values()).map((r) => ({
    aspect: r.aspect,
    positive: r.positive,
    neutral: r.neutral,
    negative: r.negative,
  }));

  return {
    overview: {
      total_reviews: 1,
      total_aspect_mentions: predictions.length,
      unique_aspects_raw: byAspect.size,
      avg_confidence: predictions.length ? confSum / predictions.length : 0,
      sentiment_counts: sentiments,
    },
    topAspects,
    aspectDist,
  };
}

function buildSingleGraphFallback(singleOut) {
  if (!singleOut?.review_id) return null;
  const predictions = Array.isArray(singleOut?.predictions) ? singleOut.predictions : [];
  if (!predictions.length) return { scope: "single_review", review_id: singleOut.review_id, nodes: [], edges: [] };

  const nodesById = new Map();
  const orderedMentions = [];

  predictions.forEach((prediction, index) => {
    const aspectId = (prediction?.aspect_cluster || prediction?.aspect_raw || `aspect-${index}`).toString().trim() || `aspect-${index}`;
    const evidence = prediction?.evidence_spans?.[0] || null;
    const start = Number.isFinite(Number(evidence?.start_char)) ? Number(evidence.start_char) : index;
    const end = Number.isFinite(Number(evidence?.end_char)) ? Number(evidence.end_char) : start;
    const sentiment = (prediction?.sentiment || "neutral").toString().toLowerCase();
    const confidence = Number(prediction?.confidence || 0);

    orderedMentions.push({ start, aspectId });

    const current = nodesById.get(aspectId);
    if (!current) {
      nodesById.set(aspectId, {
        id: aspectId,
        label: aspectId,
        sentiment,
        confidence: Number.isFinite(confidence) ? confidence : 0,
        explicit_count: 0,
        implicit_count: 0,
        evidence: evidence?.snippet || null,
        evidence_start: Number.isFinite(start) ? start : null,
        evidence_end: Number.isFinite(end) ? end : null,
        _mentions: 1,
      });
      return;
    }

    current._mentions += 1;
    current.confidence =
      ((Number(current.confidence || 0) * (current._mentions - 1)) + (Number.isFinite(confidence) ? confidence : 0)) /
      current._mentions;
    if ((current.evidence_start ?? Number.POSITIVE_INFINITY) > start) {
      current.evidence = evidence?.snippet || current.evidence;
      current.evidence_start = Number.isFinite(start) ? start : current.evidence_start;
      current.evidence_end = Number.isFinite(end) ? end : current.evidence_end;
    }
  });

  const orderedAspects = orderedMentions
    .sort((a, b) => a.start - b.start)
    .map((item) => item.aspectId);

  const transitionWeights = new Map();
  for (let i = 0; i < orderedAspects.length - 1; i += 1) {
    const source = orderedAspects[i];
    const target = orderedAspects[i + 1];
    if (!source || !target || source === target) continue;
    const key = `${source}__${target}`;
    transitionWeights.set(key, (transitionWeights.get(key) || 0) + 1);
  }

  const nodes = Array.from(nodesById.values()).map((node) => {
    const mentions = Math.max(Number(node._mentions || 1), 1);
    return {
      id: node.id,
      label: node.label,
      sentiment: node.sentiment || "neutral",
      confidence: Number(node.confidence || 0),
      explicit_count: Number(node.explicit_count || 0),
      implicit_count: Number(node.implicit_count || 0),
      evidence: node.evidence || null,
      evidence_start: node.evidence_start,
      evidence_end: node.evidence_end,
      origin: mentions > 1 ? "mixed" : "explicit",
    };
  });

  const sentimentById = new Map(nodes.map((node) => [node.id, node.sentiment]));
  const edges = Array.from(transitionWeights.entries()).map(([key, weight]) => {
    const [source, target] = key.split("__");
    const src = sentimentById.get(source) || "neutral";
    const dst = sentimentById.get(target) || "neutral";
    const polarityHint = src === dst ? src : src === "negative" || dst === "negative" ? "negative" : "mixed";
    return {
      source,
      target,
      type: "review_transition",
      weight: Number(weight || 1),
      directional: true,
      pair_count: Number(weight || 1),
      polarity_hint: polarityHint,
    };
  });

  return {
    scope: "single_review",
    review_id: singleOut.review_id,
    nodes,
    edges,
  };
}

function StatTile({ title, value, subtitle, tone }) {
  const tones = {
    violet: "from-violet-700 via-indigo-600 to-indigo-500",
    blue: "from-sky-700 via-blue-600 to-blue-500",
    amber: "from-amber-500 via-orange-500 to-yellow-500",
    rose: "from-rose-600 via-red-500 to-pink-500",
  };

  return (
    <div className={`rounded-2xl bg-gradient-to-br ${tones[tone]} p-6 text-white shadow-lg`}>
      <p className="text-5xl font-bold leading-none">{value}</p>
      <p className="mt-3 text-3xl font-semibold">{title}</p>
      <p className="mt-2 text-white/85">{subtitle}</p>
    </div>
  );
}

function cardClass(isDark) {
  return `rounded-2xl border p-5 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`;
}

const initialGraphFilters = {
  domain: "",
  product_id: "",
  from: "",
  to: "",
  min_edge_weight: 2,
};

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("reviewop-theme") || "dark");
  const isDark = theme === "dark";

  const [reviewText, setReviewText] = useState("");
  const [singleOutput, setSingleOutput] = useState(null);
  const [reviewGraph, setReviewGraph] = useState(null);
  const [batchGraph, setBatchGraph] = useState(null);
  const [graphMode, setGraphMode] = useState("batch");
  const [graphFilters, setGraphFilters] = useState(initialGraphFilters);
  const [batchFile, setBatchFile] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [overview, setOverview] = useState(null);
  const [topAspects, setTopAspects] = useState([]);
  const [aspectDist, setAspectDist] = useState([]);
  const [trends, setTrends] = useState([]);
  const [centrality, setCentrality] = useState([]);
  const [loading, setLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const [kgLoading, setKgLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem("reviewop-theme");
    if (!stored) {
      const systemPrefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      setTheme(systemPrefersDark ? "dark" : "light");
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("reviewop-theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    const load = async () => {
      try {
        await refreshAnalytics();
        await refreshBatchGraph(initialGraphFilters);
      } catch {
        // Keep initial render usable even if backend has no data yet.
      }
    };

    load();
  }, []);

  const sentimentCounts = useMemo(() => {
    if (!overview?.sentiment_counts) return { positive: 0, neutral: 0, negative: 0 };
    return {
      positive: overview.sentiment_counts.positive || 0,
      neutral: overview.sentiment_counts.neutral || 0,
      negative: overview.sentiment_counts.negative || 0,
    };
  }, [overview]);

  function clearDashboardData() {
    setError("");
    setSingleOutput(null);
    setReviewGraph(null);
    setJobStatus(null);
    setOverview(null);
    setTopAspects([]);
    setAspectDist([]);
    setTrends([]);
    setCentrality([]);
  }

  async function refreshAnalytics(domain = "") {
    const kgPromise = domain ? getKgCentrality(20, domain) : getKgCentrality();
    const [o, ta, ad, tr, c] = await Promise.all([
      getOverview(),
      getTopAspects(),
      getAspectSentimentDistribution(),
      getTrends("day"),
      kgPromise,
    ]);

    setOverview(o);
    setTopAspects(ta || []);
    setAspectDist(ad || []);
    setTrends(tr || []);
    setCentrality(c || []);
  }

  async function handleRebuildKg() {
    setError("");
    setKgLoading(true);
    try {
      const domain = (graphFilters.domain || "").trim();
      await rebuildKg(domain);
      await refreshAnalytics(domain);
    } catch (ex) {
      setError(ex.message || "KG rebuild failed");
    } finally {
      setKgLoading(false);
    }
  }

  async function refreshBatchGraph(nextFilters = graphFilters) {
    setGraphLoading(true);
    try {
      const graph = await getBatchAspectGraph(nextFilters);
      setBatchGraph(graph);
    } finally {
      setGraphLoading(false);
    }
  }

  async function handleSingleSubmit(e) {
    e.preventDefault();
    clearDashboardData();
    setLoading(true);
    try {
      const out = await inferSingleReview(reviewText.trim());
      setSingleOutput(out);
      const local = buildSingleAnalytics(out);
      setOverview(local.overview);
      setTopAspects(local.topAspects);
      setAspectDist(local.aspectDist);
      setGraphMode("single_review");

      try {
        const graph = await getReviewGraph(out.review_id);
        setReviewGraph(graph);
      } catch (graphEx) {
        setError(`Review processed, but graph load failed: ${graphEx.message || "Unknown error"}`);
      }
    } catch (ex) {
      setError(ex.message || "Single review inference failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleBatchSubmit(e) {
    e.preventDefault();
    if (!batchFile) return;

    clearDashboardData();
    setLoading(true);
    try {
      const created = await inferBatchCsv(batchFile);
      let latest = created;
      setJobStatus(created);

      for (let i = 0; i < 12; i += 1) {
        const status = await getJob(created.job_id);
        latest = status;
        setJobStatus(status);
        if (["done", "failed"].includes((status.status || "").toLowerCase())) {
          break;
        }
        await new Promise((r) => setTimeout(r, 800));
      }

      if ((latest.status || "").toLowerCase() === "failed") {
        throw new Error(latest.error || "Batch job failed");
      }

      await refreshAnalytics();
      await refreshBatchGraph(graphFilters);
      setGraphMode("batch");
      setOverview((prev) => (prev ? { ...prev, total_reviews: latest.total ?? prev.total_reviews } : prev));
    } catch (ex) {
      setError(ex.message || "Batch CSV inference failed");
    } finally {
      setLoading(false);
    }
  }

  async function applyBatchGraphFilters(e) {
    e.preventDefault();
    setError("");
    setGraphMode("batch");
    try {
      await refreshBatchGraph(graphFilters);
    } catch (ex) {
      setError(ex.message || "Batch graph load failed");
    }
  }

  const predictionRows = (singleOutput?.predictions || []).map((p, idx) => ({
    id: `${p.aspect_raw}-${idx}`,
    aspect: p.aspect_cluster || p.aspect_raw,
    sentiment: p.sentiment,
    confidence: Number(p.confidence || 0),
    evidence: p.evidence_spans?.[0]?.snippet || "-",
  }));

  const predictionColumns = [
    { field: "aspect", headerName: "Aspect", flex: 1.1, minWidth: 180 },
    { field: "sentiment", headerName: "Sentiment", flex: 0.7, minWidth: 120 },
    {
      field: "confidence",
      headerName: "Confidence",
      flex: 0.7,
      minWidth: 130,
      valueFormatter: (v) => `${(Number(v || 0) * 100).toFixed(1)}%`,
    },
    { field: "evidence", headerName: "Evidence", flex: 2, minWidth: 260 },
  ];

  const topAspectRows = topAspects.map((r, i) => ({ id: `${r.aspect}-${i}`, ...r }));
  const aspectDistRows = aspectDist.map((r, i) => ({ id: `${r.aspect}-${i}`, ...r }));
  const centralityRows = centrality.map((r, i) => ({ id: `${r.aspect}-${i}`, ...r }));
  const singleGraphFallback = useMemo(() => buildSingleGraphFallback(singleOutput), [singleOutput]);
  const activeGraph = graphMode === "single_review" ? reviewGraph || singleGraphFallback : batchGraph;

  return (
    <div className={`min-h-screen ${isDark ? "bg-[#060b18] text-slate-100" : "bg-[#f1f5f9] text-slate-800"}`}>
      <header className={`border-b ${isDark ? "border-slate-800 bg-[#050a16]" : "border-slate-200 bg-white"}`}>
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-10">
            <div className="text-3xl font-bold tracking-[0.08em]">REVIEWOP</div>
            <nav className="hidden items-center gap-8 md:flex">
              <span className="font-semibold">Dashboard</span>
              <span className={isDark ? "text-slate-400" : "text-slate-500"}>Users</span>
              <span className={isDark ? "text-slate-400" : "text-slate-500"}>Settings</span>
            </nav>
          </div>

          <label className="inline-flex items-center gap-3">
            <span className={isDark ? "text-slate-300" : "text-slate-600"}>Day</span>
            <span className="relative inline-flex items-center">
              <input
                type="checkbox"
                className="peer sr-only"
                checked={isDark}
                onChange={(e) => setTheme(e.target.checked ? "dark" : "light")}
                aria-label="Theme toggle"
              />
              <span className="h-8 w-14 rounded-full bg-slate-300 transition peer-checked:bg-indigo-600" />
              <span className="absolute left-1 top-1 h-6 w-6 rounded-full bg-white shadow transition peer-checked:left-7" />
            </span>
            <span className={isDark ? "text-slate-300" : "text-slate-600"}>Night</span>
          </label>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        <p className={`mb-7 text-2xl ${isDark ? "text-slate-400" : "text-slate-500"}`}>Home / Dashboard</p>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatTile title="Total Reviews" value={short(overview?.total_reviews)} subtitle="Reviews in current scope" tone="violet" />
          <StatTile title="Aspect Mentions" value={short(overview?.total_aspect_mentions)} subtitle="Extracted aspect mentions" tone="blue" />
          <StatTile title="Average Confidence" value={overview ? `${(overview.avg_confidence * 100).toFixed(1)}%` : "-"} subtitle="Model confidence" tone="amber" />
          <StatTile title="Negative Signals" value={short(sentimentCounts.negative)} subtitle="Negative sentiment count" tone="rose" />
        </section>

        <section className="mt-6 grid gap-4 lg:grid-cols-2">
          <form onSubmit={handleSingleSubmit} className={cardClass(isDark)}>
            <h2 className="text-2xl font-semibold">Single Review</h2>
            <textarea
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              placeholder="Paste a customer review"
              className={`mt-4 h-36 w-full rounded-xl border p-4 outline-none ${
                isDark ? "border-slate-700 bg-[#0f172a]" : "border-slate-200 bg-slate-50"
              }`}
              required
            />
            <button disabled={loading} className="mt-4 rounded-xl bg-indigo-600 px-6 py-3 font-medium text-white hover:bg-indigo-500 disabled:opacity-60">
              {loading ? "Processing..." : "Send Single Review"}
            </button>
          </form>

          <form onSubmit={handleBatchSubmit} className={cardClass(isDark)}>
            <h2 className="text-2xl font-semibold">Batch CSV</h2>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
              className={`mt-4 block w-full rounded-xl border p-3 ${
                isDark ? "border-slate-700 bg-[#0f172a]" : "border-slate-200 bg-slate-50"
              }`}
            />
            <button
              disabled={loading || !batchFile}
              className="mt-4 rounded-xl bg-amber-500 px-6 py-3 font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-60"
            >
              {loading ? "Processing..." : "Send Batch CSV"}
            </button>
            {jobStatus ? (
              <div className={`mt-4 rounded-xl p-3 text-sm ${isDark ? "bg-[#0f172a] text-slate-300" : "bg-slate-100 text-slate-700"}`}>
                Job {jobStatus.job_id} | {jobStatus.status} | {short(jobStatus.processed || 0)} / {short(jobStatus.total || 0)}
              </div>
            ) : null}
          </form>
        </section>

        {error ? (
          <div className={`mt-4 rounded-xl p-4 ${isDark ? "bg-red-950 text-red-300" : "bg-red-100 text-red-700"}`}>{error}</div>
        ) : null}

        <section className="mt-6 grid gap-4 lg:grid-cols-2">
          <SingleSentimentChart predictions={singleOutput?.predictions || []} isDark={isDark} />
          <TrendsChart data={trends} isDark={isDark} />
        </section>

        <section className={`mt-6 ${cardClass(isDark)}`}>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-500">Aspect Graphs</p>
              <h3 className="mt-1 text-2xl font-semibold">Interactive Aspect Relationship Views</h3>
              <p className={`mt-2 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                Shared visual language, separate graph semantics for single-review explanation and corpus analytics.
              </p>
            </div>
            <GraphModeToggle value={graphMode} onChange={setGraphMode} isDark={isDark} />
          </div>

          <form onSubmit={applyBatchGraphFilters} className={`mt-5 rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#08101f]" : "border-slate-200 bg-slate-50"}`}>
            <div className="flex flex-col gap-3 lg:flex-row lg:flex-wrap lg:items-end">
              <label className="flex-1 min-w-[140px] text-sm">
                <span className="mb-1 block font-medium">Domain</span>
                <input
                  value={graphFilters.domain}
                  onChange={(e) => setGraphFilters((prev) => ({ ...prev, domain: e.target.value }))}
                  className={`w-full rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-[#0f172a]" : "border-slate-200 bg-white"}`}
                  placeholder="electronics"
                />
              </label>
              <label className="flex-1 min-w-[140px] text-sm">
                <span className="mb-1 block font-medium">Product ID</span>
                <input
                  value={graphFilters.product_id}
                  onChange={(e) => setGraphFilters((prev) => ({ ...prev, product_id: e.target.value }))}
                  className={`w-full rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-[#0f172a]" : "border-slate-200 bg-white"}`}
                  placeholder="sku-42"
                />
              </label>
              <label className="text-sm">
                <span className="mb-1 block font-medium">From</span>
                <input
                  type="date"
                  value={graphFilters.from}
                  onChange={(e) => setGraphFilters((prev) => ({ ...prev, from: e.target.value }))}
                  className={`rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-[#0f172a]" : "border-slate-200 bg-white"}`}
                />
              </label>
              <label className="text-sm">
                <span className="mb-1 block font-medium">To</span>
                <input
                  type="date"
                  value={graphFilters.to}
                  onChange={(e) => setGraphFilters((prev) => ({ ...prev, to: e.target.value }))}
                  className={`rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-[#0f172a]" : "border-slate-200 bg-white"}`}
                />
              </label>
              <label className="text-sm">
                <span className="mb-1 block font-medium">Min Edge Weight</span>
                <input
                  type="number"
                  min="1"
                  value={graphFilters.min_edge_weight}
                  onChange={(e) => setGraphFilters((prev) => ({ ...prev, min_edge_weight: Number(e.target.value || 1) }))}
                  className={`w-28 rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-[#0f172a]" : "border-slate-200 bg-white"}`}
                />
              </label>
              <button type="submit" disabled={graphLoading} className="rounded-xl bg-emerald-500 px-5 py-2.5 font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-60">
                {graphLoading ? "Loading..." : "Apply Batch Filters"}
              </button>
            </div>
          </form>

          <div className="mt-5">
            <AspectGraphView
              graph={activeGraph}
              scope={graphMode}
              isDark={isDark}
              reviewText={singleOutput?.review_id ? reviewText : ""}
              emptyMessage={
                graphMode === "single_review"
                  ? "Run single-review inference to render the explanation graph."
                  : "Upload a batch CSV or apply filters to render the corpus graph."
              }
            />
          </div>
        </section>

        <section className={`mt-6 ${cardClass(isDark)}`}>
          <h3 className="mb-3 text-lg font-semibold">Single Review Predictions</h3>
          <DataGridTable columns={predictionColumns} rows={predictionRows} isDark={isDark} height={360} />
        </section>

        <section className="mt-6 grid gap-4 lg:grid-cols-2">
          <div className={cardClass(isDark)}>
            <h3 className="mb-3 text-lg font-semibold">Top Aspects</h3>
            <DataGridTable
              isDark={isDark}
              height={350}
              columns={[
                { field: "aspect", headerName: "Aspect", flex: 1.2, minWidth: 180 },
                { field: "count", headerName: "Count", type: "number", flex: 0.6, minWidth: 120 },
              ]}
              rows={topAspectRows}
            />
          </div>

          <div className={cardClass(isDark)}>
            <div className="mb-3 flex items-center justify-between gap-3">
              <h3 className="text-lg font-semibold">KG Centrality</h3>
              <button
                type="button"
                onClick={handleRebuildKg}
                disabled={kgLoading}
                className="rounded-lg bg-cyan-500 px-3 py-1.5 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-60"
              >
                {kgLoading ? "Rebuilding..." : "Rebuild KG"}
              </button>
            </div>
            <DataGridTable
              isDark={isDark}
              height={350}
              columns={[
                { field: "aspect", headerName: "Aspect", flex: 1.1, minWidth: 180 },
                {
                  field: "centrality",
                  headerName: "Centrality",
                  type: "number",
                  flex: 0.8,
                  minWidth: 130,
                  valueFormatter: (v) => Number(v || 0).toFixed(4),
                },
                { field: "df", headerName: "DF", type: "number", flex: 0.5, minWidth: 100 },
              ]}
              rows={centralityRows}
            />
            {!centralityRows.length ? (
              <p className={`mt-3 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                No KG centrality yet. Click <span className="font-semibold">Rebuild KG</span> to compute graph centrality values.
              </p>
            ) : null}
          </div>
        </section>

        <section className={`mt-6 ${cardClass(isDark)}`}>
          <h3 className="mb-3 text-lg font-semibold">Aspect Sentiment Distribution</h3>
          <DataGridTable
            isDark={isDark}
            height={360}
            rows={aspectDistRows}
            columns={[
              { field: "aspect", headerName: "Aspect", flex: 1.2, minWidth: 180 },
              { field: "positive", headerName: "Positive", type: "number", flex: 0.6, minWidth: 100 },
              { field: "neutral", headerName: "Neutral", type: "number", flex: 0.6, minWidth: 100 },
              { field: "negative", headerName: "Negative", type: "number", flex: 0.6, minWidth: 100 },
            ]}
          />
        </section>
      </main>
    </div>
  );
}
