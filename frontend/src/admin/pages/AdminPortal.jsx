import Dashboard from "./Dashboard";
import AspectAnalytics from "./AspectAnalytics";
import GraphExplorer from "./GraphExplorer";
import ReviewExplorer from "./ReviewExplorer";
import AlertsPage from "./AlertsPage";
import AlertDetailPage from "./AlertDetailPage";
import ReviewQueue from "./ReviewQueue";
import UserReviewsInsights from "./UserReviewsInsights";
import ExportsPage from "./ExportsPage";
import { useAdminPortal } from "./useAdminPortal";
import { useState } from "react";

export default function AdminPortal() {
  const [exportOpen, setExportOpen] = useState(false);
  const state = useAdminPortal();
  const {
    isDark,
    theme,
    setTheme,
    activePage,
    setActivePage,
    selectedAlert,
    reviewText,
    setReviewText,
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
    handleSingleSubmit,
    handleBatchSubmit,
    applyBatchGraphFilters,
    resetBatchGraphFilters,
    handleAlertClick,
    handleAlertClear,
    handleExportJson,
    handleExportPdf,
    handleLogout,
  } = state;

  return (
    <div className={`min-h-screen ${isDark ? "bg-[#060b18] text-slate-100" : "bg-[#f1f5f9] text-slate-800"}`}>
      <header className={`border-b ${isDark ? "border-slate-800 bg-[#050a16]" : "border-slate-200 bg-white"}`}>
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-5">
          <div className="text-3xl font-bold tracking-[0.08em]">REVIEWOP</div>
          <div className="flex flex-wrap items-center gap-2">
            {pageNav.map((name) => (
              <button key={name} type="button" onClick={() => setActivePage(name)} className={`rounded-xl px-3 py-2 text-sm font-semibold ${activePage === name ? "bg-emerald-500 text-slate-950" : isDark ? "bg-slate-800 text-slate-200" : "bg-slate-200 text-slate-700"}`}>
                {name === "AspectAnalytics" ? "Analytics" : name === "ReviewQueue" ? "Review Queue" : name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <label className="inline-flex items-center gap-3 text-sm">
              <span className={isDark ? "text-slate-300" : "text-slate-600"}>Day</span>
              <span className="relative inline-flex items-center">
                <input type="checkbox" className="peer sr-only" checked={isDark} onChange={(e) => setTheme(e.target.checked ? "dark" : "light")} aria-label="Theme toggle" />
                <span className="h-8 w-14 rounded-full bg-slate-300 transition-all duration-300 peer-checked:bg-indigo-600" />
                <span className="absolute left-1 top-1 h-6 w-6 rounded-full bg-white shadow transition-all duration-300 peer-checked:left-7" />
              </span>
              <span className={isDark ? "text-slate-300" : "text-slate-600"}>Night</span>
            </label>
            <button type="button" onClick={handleLogout} className={`rounded-xl px-3 py-2 text-sm font-semibold ${isDark ? "bg-slate-800 text-slate-200" : "bg-slate-200 text-slate-700"}`}>
              Logout
            </button>
            <button type="button" onClick={() => setExportOpen(true)} className="rounded-xl bg-emerald-500 px-3 py-2 text-sm font-semibold text-slate-950">
              Export
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-4 px-6 py-6">
        {error ? <div className={`rounded-xl p-3 ${isDark ? "bg-red-950 text-red-300" : "bg-red-100 text-red-700"}`}>{error}</div> : null}
        {startupWarnings.length ? (
          <div className={`rounded-xl p-3 text-sm ${isDark ? "bg-amber-950 text-amber-200" : "bg-amber-100 text-amber-800"}`}>
            Some admin data did not load: {startupWarnings.join("; ")}
          </div>
        ) : null}
        <div key={activePage} className="page-fade-in">
          {activePage === "Dashboard" ? <Dashboard kpis={kpis} alerts={alerts} leaderboardRows={leaderboardRows} impactRows={impactMatrix} segmentRows={segmentRows} weeklySummary={weeklySummary} isDark={isDark} onSeeMoreAlerts={() => setActivePage("Alerts")} onOpenGraph={() => setActivePage("GraphExplorer")} onAlertClick={handleAlertClick} /> : null}
          {activePage === "AspectAnalytics" ? <AspectAnalytics trends={aspectTrends} emerging={emergingAspects} evidence={evidenceRows} aspectDetail={aspectDetail} weeklySummary={weeklySummary} isDark={isDark} /> : null}
          {activePage === "GraphExplorer" ? <GraphExplorer graph={batchGraph} graphFilters={graphFilters} setGraphFilters={setGraphFilters} onApplyFilters={applyBatchGraphFilters} onResetFilters={resetBatchGraphFilters} graphLoading={graphLoading} filterOptions={graphFilterOptions} isDark={isDark} /> : null}
          {activePage === "ReviewExplorer" ? <ReviewExplorer reviewText={reviewText} setReviewText={setReviewText} onSubmit={handleSingleSubmit} loading={loading} output={singleOutput} reviewGraph={reviewGraph} batchFile={batchFile} setBatchFile={setBatchFile} onBatchSubmit={handleBatchSubmit} jobStatus={jobStatus} kpis={kpis} leaderboardRows={leaderboardRows} impactRows={impactMatrix} segmentRows={segmentRows} weeklySummary={weeklySummary} alerts={alerts} evidenceRows={evidenceRows} onOpenGraph={() => setActivePage("GraphExplorer")} onOpenAnalytics={() => setActivePage("AspectAnalytics")} isDark={isDark} /> : null}
          {activePage === "Alerts" ? <AlertsPage alerts={alerts} isDark={isDark} onAlertClick={handleAlertClick} onAlertClear={handleAlertClear} /> : null}
          {activePage === "ReviewQueue" ? <ReviewQueue needsReviewRows={needsReviewRows} novelCandidateRows={novelCandidateRows} isDark={isDark} /> : null}
          {activePage === "UserReviews" ? <UserReviewsInsights summary={userReviewSummary} list={userReviewList} isDark={isDark} onAnalyzeReview={(text) => { setReviewText(text); setSingleReviewPersist(false); setActivePage("ReviewExplorer"); }} /> : null}
          {activePage === "AlertDetail" ? <AlertDetailPage alert={selectedAlert} isDark={isDark} onBack={() => setActivePage("Alerts")} onOpenGraphNode={() => setActivePage("GraphExplorer")} /> : null}
        </div>
      </main>

      {exportOpen ? (
        <div className="fixed inset-0 z-50 flex items-stretch justify-end bg-black/40">
          <div className={`h-full w-full max-w-4xl overflow-auto border-l p-4 ${isDark ? "border-slate-800 bg-[#060b18]" : "border-slate-200 bg-white"}`}>
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-lg font-semibold">Export</h3>
              <button type="button" onClick={() => setExportOpen(false)} className={`rounded-xl px-3 py-2 text-sm font-semibold ${isDark ? "bg-slate-800 text-slate-200" : "bg-slate-200 text-slate-700"}`}>
                Close
              </button>
            </div>
            <ExportsPage isDark={isDark} onExportJson={handleExportJson} onExportPdf={handleExportPdf} exportPayload={exportPayload} exportFilters={exportFilters} setExportFilters={setExportFilters} loading={exportLoading} onRefreshExport={() => refreshExportPreview(exportFilters)} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
