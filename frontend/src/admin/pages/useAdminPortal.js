import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  inferSingleReview,
  inferBatchCsv,
  getJob,
  getReviewGraph,
  getGraphFilterOptions,
  getBatchAspectGraph,
  getDashboardKpis,
  getAspectLeaderboard,
  getAspectTrends,
  getEmergingAspects,
  getEvidence,
  getAspectDetail,
  getAlerts,
  getNeedsReview,
  getNovelCandidates,
  getImpactMatrix,
  getSegments,
  getWeeklySummary,
  clearAlert,
  exportAdminJson,
  exportAdminPdf,
  getAdminExport,
  getUserReviewsList,
  getUserReviewsSummary,
} from "../../api/client";
import { resetGraphFilters } from "./graphFilterUtils";
import { useAuth } from "../../auth/AuthContext";

const initialGraphFilters = {
  domain: "",
  product_id: "",
  from: "",
  to: "",
  min_edge_weight: 1,
  graph_mode: "accepted",
};

export function useAdminPortal() {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [theme, setTheme] = useState(() => localStorage.getItem("reviewop-theme") || "dark");
  const isDark = theme === "dark";
  const [activePage, setActivePage] = useState("Dashboard");
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [reviewText, setReviewText] = useState("");
  const [singleReviewPersist, setSingleReviewPersist] = useState(true);
  const [singleOutput, setSingleOutput] = useState(null);
  const [reviewGraph, setReviewGraph] = useState(null);
  const [batchGraph, setBatchGraph] = useState(null);
  const [graphFilters, setGraphFilters] = useState(initialGraphFilters);
  const [graphFilterOptions, setGraphFilterOptions] = useState({ domains: [], product_ids: [] });
  const [batchFile, setBatchFile] = useState(null);
  const [kpis, setKpis] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);
  const [aspectTrends, setAspectTrends] = useState([]);
  const [emergingAspects, setEmergingAspects] = useState([]);
  const [evidenceRows, setEvidenceRows] = useState([]);
  const [aspectDetail, setAspectDetail] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [needsReviewRows, setNeedsReviewRows] = useState([]);
  const [novelCandidateRows, setNovelCandidateRows] = useState([]);
  const [impactMatrix, setImpactMatrix] = useState([]);
  const [segmentRows, setSegmentRows] = useState([]);
  const [weeklySummary, setWeeklySummary] = useState(null);
  const [userReviewSummary, setUserReviewSummary] = useState(null);
  const [userReviewList, setUserReviewList] = useState({ total: 0, limit: 50, offset: 0, rows: [] });
  const [exportFilters, setExportFilters] = useState({ domain: "", limit: 100, offset: 0 });
  const [exportPayload, setExportPayload] = useState(null);
  const [exportLoading, setExportLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const [error, setError] = useState("");
  const [startupWarnings, setStartupWarnings] = useState([]);
  const [jobStatus, setJobStatus] = useState(null);

  useEffect(() => {
    localStorage.setItem("reviewop-theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  async function refreshAnalytics() {
    const [k, l, t, e, ev, a, nr, nc, impact, segments, weekly, urs, url] = await Promise.all([
      getDashboardKpis(),
      getAspectLeaderboard(),
      getAspectTrends("day"),
      getEmergingAspects("day", 7),
      getEvidence("", "", 30),
      getAlerts(),
      getNeedsReview(),
      getNovelCandidates(),
      getImpactMatrix(),
      getSegments(),
      getWeeklySummary(),
      getUserReviewsSummary(),
      getUserReviewsList({ limit: 100, offset: 0 }),
    ]);
    setKpis(k);
    setLeaderboard(l || []);
    setAspectTrends(t || []);
    setEmergingAspects(e || []);
    setEvidenceRows(ev || []);
    setAlerts(a || []);
    setNeedsReviewRows(nr || []);
    setNovelCandidateRows(nc || []);
    setImpactMatrix(impact || []);
    setSegmentRows(segments || []);
    setWeeklySummary(weekly || null);
    setUserReviewSummary(urs || null);
    setUserReviewList(url || { total: 0, limit: 50, offset: 0, rows: [] });
    if (l?.length) {
      const detail = await getAspectDetail(l[0].aspect);
      setAspectDetail(detail);
    }
  }

  async function refreshBatchGraph(nextFilters = graphFilters) {
    setGraphLoading(true);
    try {
      setBatchGraph(await getBatchAspectGraph(nextFilters));
    } finally {
      setGraphLoading(false);
    }
  }

  async function refreshExportPreview(nextFilters = exportFilters) {
    setExportLoading(true);
    try {
      setExportPayload(await getAdminExport(nextFilters));
    } finally {
      setExportLoading(false);
    }
  }

  useEffect(() => {
    const load = async () => {
      const warnings = [];
      const options = await getGraphFilterOptions().catch((ex) => {
        warnings.push(`Graph filters: ${ex.message || "failed to load"}`);
        return { domains: [], product_ids: [] };
      });
      await refreshAnalytics().catch((ex) => warnings.push(`Analytics: ${ex.message || "failed to load"}`));
      await refreshBatchGraph(initialGraphFilters).catch((ex) => warnings.push(`Graph: ${ex.message || "failed to load"}`));
      await refreshExportPreview({ domain: "", limit: 100, offset: 0 }).catch((ex) => warnings.push(`Export preview: ${ex.message || "failed to load"}`));
      setGraphFilterOptions(options || { domains: [], product_ids: [] });
      setStartupWarnings(warnings);
    };
    load();
  }, []);

  async function handleSingleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const out = await inferSingleReview(reviewText.trim(), null, null, singleReviewPersist);
      setSingleOutput(out);
      if (singleReviewPersist && out.review_id) {
        setReviewGraph(await getReviewGraph(out.review_id));
      } else {
        setReviewGraph(null);
      }
      setActivePage("ReviewExplorer");
    } catch (ex) {
      setError(ex.message || "Single review inference failed");
    } finally {
      setSingleReviewPersist(true);
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
      const options = await getGraphFilterOptions().catch(() => graphFilterOptions);
      setGraphFilterOptions(options || { domains: [], product_ids: [] });
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

  async function resetBatchGraphFilters() {
    setError("");
    const resetFilters = resetGraphFilters(initialGraphFilters);
    setGraphFilters(resetFilters);
    try {
      await refreshBatchGraph(resetFilters);
    } catch (ex) {
      setError(ex.message || "Batch graph reset failed");
    }
  }

  const leaderboardRows = useMemo(() => leaderboard.map((row, idx) => ({ id: `${row.aspect}-${idx}`, ...row })), [leaderboard]);
  const pageNav = ["Dashboard", "AspectAnalytics", "GraphExplorer", "ReviewExplorer", "Alerts", "ReviewQueue", "UserReviews"];

  const handleAlertClick = (alert) => {
    setSelectedAlert(alert);
    setActivePage("AlertDetail");
  };

  async function handleAlertClear(alert) {
    if (!alert?.id) return;
    try {
      await clearAlert(alert.id);
      await refreshAnalytics();
      if (selectedAlert?.id === alert.id) {
        setSelectedAlert(null);
        setActivePage("Alerts");
      }
    } catch (ex) {
      setError(ex.message || "Failed to clear alert");
    }
  }

  async function handleExportJson() {
    try {
      await exportAdminJson(exportFilters);
    } catch (ex) {
      setError(ex.message || "Failed to export JSON");
    }
  }

  async function handleExportPdf() {
    try {
      await exportAdminPdf(exportFilters);
    } catch (ex) {
      setError(ex.message || "Failed to export PDF");
    }
  }

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return {
    isDark,
    theme,
    setTheme,
    activePage,
    setActivePage,
    selectedAlert,
    reviewText,
    setReviewText,
    singleReviewPersist,
    setSingleReviewPersist,
    singleOutput,
    reviewGraph,
    batchGraph,
    graphFilters,
    setGraphFilters,
    graphFilterOptions,
    batchFile,
    setBatchFile,
    kpis,
    leaderboardRows,
    aspectTrends,
    emergingAspects,
    evidenceRows,
    aspectDetail,
    alerts,
    needsReviewRows,
    novelCandidateRows,
    impactMatrix,
    segmentRows,
    weeklySummary,
    userReviewSummary,
    userReviewList,
    exportFilters,
    setExportFilters,
    exportPayload,
    exportLoading,
    loading,
    graphLoading,
    error,
    startupWarnings,
    jobStatus,
    pageNav,
    refreshExportPreview,
    refreshAnalytics,
    refreshBatchGraph,
    handleSingleSubmit,
    handleBatchSubmit,
    applyBatchGraphFilters,
    resetBatchGraphFilters,
    handleAlertClick,
    handleAlertClear,
    handleExportJson,
    handleExportPdf,
    handleLogout,
  };
}
