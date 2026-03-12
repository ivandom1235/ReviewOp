import { useEffect, useMemo, useState } from "react";
import {
  inferSingleReview,
  inferBatchCsv,
  getJob,
  getReviewGraph,
  getBatchAspectGraph,
  getDashboardKpis,
  getAspectLeaderboard,
  getAspectTrends,
  getEmergingAspects,
  getEvidence,
  getAspectDetail,
  getAlerts,
  getImpactMatrix,
  getSegments,
  getWeeklySummary,
} from "./api/client";
import Dashboard from "./pages/Dashboard";
import AspectAnalytics from "./pages/AspectAnalytics";
import GraphExplorer from "./pages/GraphExplorer";
import ReviewExplorer from "./pages/ReviewExplorer";

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
  const [activePage, setActivePage] = useState("Dashboard");

  const [reviewText, setReviewText] = useState("");
  const [singleOutput, setSingleOutput] = useState(null);
  const [reviewGraph, setReviewGraph] = useState(null);
  const [batchGraph, setBatchGraph] = useState(null);
  const [graphFilters, setGraphFilters] = useState(initialGraphFilters);
  const [batchFile, setBatchFile] = useState(null);

  const [kpis, setKpis] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);
  const [aspectTrends, setAspectTrends] = useState([]);
  const [emergingAspects, setEmergingAspects] = useState([]);
  const [evidenceRows, setEvidenceRows] = useState([]);
  const [aspectDetail, setAspectDetail] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [impactMatrix, setImpactMatrix] = useState([]);
  const [segmentRows, setSegmentRows] = useState([]);
  const [weeklySummary, setWeeklySummary] = useState(null);

  const [loading, setLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const [error, setError] = useState("");
  const [jobStatus, setJobStatus] = useState(null);

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
        // allow empty startup
      }
    };
    load();
  }, []);

  async function refreshAnalytics() {
    const [k, l, t, e, ev, a, impact, segments, weekly] = await Promise.all([
      getDashboardKpis(),
      getAspectLeaderboard(),
      getAspectTrends("day"),
      getEmergingAspects("day", 7),
      getEvidence("", "", 30),
      getAlerts(),
      getImpactMatrix(),
      getSegments(),
      getWeeklySummary(),
    ]);
    setKpis(k);
    setLeaderboard(l || []);
    setAspectTrends(t || []);
    setEmergingAspects(e || []);
    setEvidenceRows(ev || []);
    setAlerts(a || []);
    setImpactMatrix(impact || []);
    setSegmentRows(segments || []);
    setWeeklySummary(weekly || null);

    if (l?.length) {
      const detail = await getAspectDetail(l[0].aspect);
      setAspectDetail(detail);
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
    setError("");
    setLoading(true);
    try {
      const out = await inferSingleReview(reviewText.trim());
      setSingleOutput(out);
      const graph = await getReviewGraph(out.review_id);
      setReviewGraph(graph);
      setActivePage("ReviewExplorer");
    } catch (ex) {
      setError(ex.message || "Single review inference failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleBatchSubmit(e) {
    e.preventDefault();
    if (!batchFile) return;

    setError("");
    setLoading(true);
    try {
      const created = await inferBatchCsv(batchFile);
      let latest = created;
      setJobStatus(created);
      for (let i = 0; i < 15; i += 1) {
        const status = await getJob(created.job_id);
        latest = status;
        setJobStatus(status);
        if (["done", "failed"].includes((status.status || "").toLowerCase())) break;
        await new Promise((r) => setTimeout(r, 800));
      }
      if ((latest.status || "").toLowerCase() === "failed") throw new Error(latest.error || "Batch job failed");
      await Promise.all([refreshAnalytics(), refreshBatchGraph(graphFilters)]);
      setActivePage("ReviewExplorer");
    } catch (ex) {
      setError(ex.message || "Batch CSV inference failed");
    } finally {
      setLoading(false);
    }
  }

  async function applyBatchGraphFilters(e) {
    e.preventDefault();
    setError("");
    try {
      await refreshBatchGraph(graphFilters);
    } catch (ex) {
      setError(ex.message || "Batch graph load failed");
    }
  }

  const leaderboardRows = useMemo(() => leaderboard.map((row, idx) => ({ id: `${row.aspect}-${idx}`, ...row })), [leaderboard]);
  const pageNav = ["Dashboard", "AspectAnalytics", "GraphExplorer", "ReviewExplorer"];

  return (
    <div className={`min-h-screen ${isDark ? "bg-[#060b18] text-slate-100" : "bg-[#f1f5f9] text-slate-800"}`}>
      <header className={`border-b ${isDark ? "border-slate-800 bg-[#050a16]" : "border-slate-200 bg-white"}`}>
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-5">
          <div className="text-3xl font-bold tracking-[0.08em]">REVIEWOP</div>
          <div className="flex flex-wrap items-center gap-2">
            {pageNav.map((name) => (
              <button key={name} type="button" onClick={() => setActivePage(name)} className={`rounded-xl px-3 py-2 text-sm font-semibold ${activePage === name ? "bg-emerald-500 text-slate-950" : isDark ? "bg-slate-800 text-slate-200" : "bg-slate-200 text-slate-700"}`}>
                {name}
              </button>
            ))}
          </div>
          <label className="inline-flex items-center gap-3 text-sm">
            <span className={isDark ? "text-slate-300" : "text-slate-600"}>Day</span>
            <span className="relative inline-flex items-center">
              <input
                type="checkbox"
                className="peer sr-only"
                checked={isDark}
                onChange={(e) => setTheme(e.target.checked ? "dark" : "light")}
                aria-label="Theme toggle"
              />
              <span className="h-8 w-14 rounded-full bg-slate-300 transition-all duration-300 peer-checked:bg-indigo-600" />
              <span className="absolute left-1 top-1 h-6 w-6 rounded-full bg-white shadow transition-all duration-300 peer-checked:left-7" />
            </span>
            <span className={isDark ? "text-slate-300" : "text-slate-600"}>Night</span>
          </label>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-4 px-6 py-6">
        {error ? <div className={`rounded-xl p-3 ${isDark ? "bg-red-950 text-red-300" : "bg-red-100 text-red-700"}`}>{error}</div> : null}

        <div key={activePage} className="page-fade-in">
          {activePage === "Dashboard" ? <Dashboard kpis={kpis} alerts={alerts} leaderboardRows={leaderboardRows} impactRows={impactMatrix} segmentRows={segmentRows} weeklySummary={weeklySummary} isDark={isDark} /> : null}
          {activePage === "AspectAnalytics" ? <AspectAnalytics trends={aspectTrends} emerging={emergingAspects} evidence={evidenceRows} aspectDetail={aspectDetail} weeklySummary={weeklySummary} isDark={isDark} /> : null}
          {activePage === "GraphExplorer" ? (
            <GraphExplorer
              graph={batchGraph}
              graphFilters={graphFilters}
              setGraphFilters={setGraphFilters}
              onApplyFilters={applyBatchGraphFilters}
              graphLoading={graphLoading}
              isDark={isDark}
            />
          ) : null}
          {activePage === "ReviewExplorer" ? (
            <ReviewExplorer
              reviewText={reviewText}
              setReviewText={setReviewText}
              onSubmit={handleSingleSubmit}
              loading={loading}
              output={singleOutput}
              reviewGraph={reviewGraph}
              batchFile={batchFile}
              setBatchFile={setBatchFile}
              onBatchSubmit={handleBatchSubmit}
              jobStatus={jobStatus}
              kpis={kpis}
              leaderboardRows={leaderboardRows}
              impactRows={impactMatrix}
              segmentRows={segmentRows}
              weeklySummary={weeklySummary}
              alerts={alerts}
              evidenceRows={evidenceRows}
              onOpenGraph={() => setActivePage("GraphExplorer")}
              onOpenAnalytics={() => setActivePage("AspectAnalytics")}
              isDark={isDark}
            />
          ) : null}
        </div>
      </main>
    </div>
  );
}
