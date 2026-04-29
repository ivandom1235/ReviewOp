import { useEffect, useMemo, useState } from "react";
import DataGridTable from "../components/DataGridTable";

const PRESETS = [
  {
    id: "summary",
    name: "Summary export",
    description: "A compact overview of KPIs, alerts, and weekly trends.",
    include: ["dashboard_kpis", "alerts", "weekly_summary", "impact_matrix"],
  },
  {
    id: "alerts",
    name: "Alerts export",
    description: "Filtered alert records with severity and evidence context.",
    include: ["alerts", "evidence"],
  },
  {
    id: "reviews",
    name: "Reviews export",
    description: "User review rows with product, rating, and reply metadata.",
    include: ["user_reviews_summary", "user_reviews"],
  },
  {
    id: "product-performance",
    name: "Product performance export",
    description: "Review and sentiment summary by product for analysis.",
    include: ["aspect_leaderboard", "aspect_trends", "impact_matrix", "segments"],
  },
];

export default function ExportsPage({ isDark, onExportJson, onExportPdf, exportPayload, exportFilters, setExportFilters, loading = false, onRefreshExport }) {
  const [exporting, setExporting] = useState(null);
  const previewRows = useMemo(() => {
    const rows = exportPayload?.user_reviews?.rows || [];
    return rows.slice(0, 8).map((row) => ({
      id: row.review_id,
      ...row,
    }));
  }, [exportPayload]);

  async function handleExport(kind) {
    setExporting(kind);
    try {
      if (kind === "json") {
        await onExportJson(exportFilters);
      } else {
        await onExportPdf(exportFilters);
      }
    } finally {
      setExporting(null);
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      onRefreshExport?.();
    }, 250);
    return () => window.clearTimeout(timer);
  }, [exportFilters.domain, exportFilters.limit, exportFilters.offset]);

  return (
    <section className="space-y-4">
      <div className={`rounded-2xl border p-6 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold">Export Center</h2>
            <p className="mt-1 text-sm text-muted-main">Curated exports for analysis, reporting, and downstream review.</p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button type="button" onClick={onRefreshExport} disabled={loading} className="rounded-xl border border-border-subtle bg-app/30 px-4 py-2 text-sm font-semibold text-brand-primary hover:bg-brand-primary/10 disabled:opacity-50">
              {loading ? "Loading..." : "Refresh preview"}
            </button>
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <label className="space-y-2 text-sm">
            <span className="font-semibold">Domain filter</span>
            <input
              value={exportFilters.domain}
              onChange={(e) => setExportFilters((prev) => ({ ...prev, domain: e.target.value }))}
              placeholder="electronics"
              className={`w-full rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-slate-900" : "border-slate-200 bg-white"}`}
            />
          </label>
          <label className="space-y-2 text-sm">
            <span className="font-semibold">Review limit</span>
            <input
              type="number"
              min="1"
              max="200"
              value={exportFilters.limit}
              onChange={(e) => setExportFilters((prev) => ({ ...prev, limit: Number(e.target.value || 100) }))}
              className={`w-full rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-slate-900" : "border-slate-200 bg-white"}`}
            />
          </label>
          <div className="flex items-end gap-3">
            <button type="button" onClick={() => handleExport("json")} disabled={exporting !== null} className="rounded-xl border border-border-subtle bg-app/30 px-4 py-2 text-sm font-semibold text-brand-primary hover:bg-brand-primary/10 disabled:opacity-50">
              {exporting === "json" ? "Exporting..." : "Export JSON"}
            </button>
            <button type="button" onClick={() => handleExport("pdf")} disabled={exporting !== null} className="rounded-xl border border-border-subtle bg-app/30 px-4 py-2 text-sm font-semibold text-brand-primary hover:bg-brand-primary/10 disabled:opacity-50">
              {exporting === "pdf" ? "Exporting..." : "Export PDF"}
            </button>
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {PRESETS.map((preset) => (
            <div key={preset.name} className="rounded-xl border border-border-subtle bg-app/30 p-4">
              <h3 className="font-semibold">{preset.name}</h3>
              <p className="mt-1 text-sm text-muted-main">{preset.description}</p>
              <p className="mt-3 text-xs uppercase tracking-[0.14em] text-muted-main">Includes</p>
              <p className="mt-1 text-xs text-muted-main">{preset.include.join(", ")}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 grid gap-4 xl:grid-cols-2">
          <div className={`rounded-xl border p-4 ${isDark ? "border-slate-800 bg-[#0f172a]" : "border-slate-200 bg-slate-50"}`}>
            <h3 className="text-lg font-semibold">Export Snapshot</h3>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg bg-white/60 p-3 dark:bg-slate-900">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-main">Generated at</p>
                <p className="mt-1 text-sm font-semibold">{exportPayload?.generated_at ? new Date(exportPayload.generated_at).toLocaleString() : "-"}</p>
              </div>
              <div className="rounded-lg bg-white/60 p-3 dark:bg-slate-900">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-main">Reviews in export</p>
                <p className="mt-1 text-sm font-semibold">{exportPayload?.user_reviews?.total ?? 0}</p>
              </div>
              <div className="rounded-lg bg-white/60 p-3 dark:bg-slate-900">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-main">Active alerts</p>
                <p className="mt-1 text-sm font-semibold">{exportPayload?.alerts?.length ?? 0}</p>
              </div>
              <div className="rounded-lg bg-white/60 p-3 dark:bg-slate-900">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-main">Top drivers</p>
                <p className="mt-1 text-sm font-semibold">{exportPayload?.weekly_summary?.top_drivers?.slice(0, 2).join(", ") || "-"}</p>
              </div>
            </div>
          </div>

          <div className={`rounded-xl border p-4 ${isDark ? "border-slate-800 bg-[#0f172a]" : "border-slate-200 bg-slate-50"}`}>
            <h3 className="text-lg font-semibold">Export Data Preview</h3>
            <p className="mt-1 text-sm text-muted-main">Latest review rows included in the JSON and PDF exports.</p>
            <div className="mt-4">
              <DataGridTable
                isDark={isDark}
                height={320}
                rows={previewRows}
                columns={[
                  { field: "created_at", headerName: "Created", flex: 0.9, minWidth: 140 },
                  { field: "username", headerName: "User", flex: 0.6, minWidth: 110 },
                  { field: "product_id", headerName: "Product", flex: 0.7, minWidth: 100 },
                  { field: "rating", headerName: "Rating", type: "number", flex: 0.4, minWidth: 80 },
                  { field: "review_title", headerName: "Title", flex: 0.9, minWidth: 140 },
                ]}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
