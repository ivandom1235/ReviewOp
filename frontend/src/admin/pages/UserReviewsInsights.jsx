import DataGridTable from "../components/DataGridTable";

function StatCard({ label, value, isDark }) {
  return (
    <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
      <p className="text-xs uppercase tracking-[0.1em] opacity-70">{label}</p>
      <p className="mt-1 text-2xl font-bold">{value}</p>
    </div>
  );
}

export default function UserReviewsInsights({ summary, list, isDark = false }) {
  const rows = (list?.rows || []).map((row) => ({
    id: row.review_id,
    ...row,
  }));

  return (
    <section className="space-y-4">
      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">User Review Insights</h3>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <StatCard label="Total Reviews" value={summary?.total_user_reviews ?? 0} isDark={isDark} />
          <StatCard label="Unique Reviewers" value={summary?.unique_reviewers ?? 0} isDark={isDark} />
          <StatCard label="Avg Rating" value={summary?.average_rating ?? 0} isDark={isDark} />
          <StatCard label="Recommendation %" value={summary?.recommendation_rate ?? 0} isDark={isDark} />
          <StatCard label="Last 7 Days" value={summary?.reviews_last_7_days ?? 0} isDark={isDark} />
        </div>
      </div>

      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">Top Products</h3>
        <DataGridTable
          isDark={isDark}
          height={250}
          rows={(summary?.top_products || []).map((p, idx) => ({ id: `${p.product_id}-${idx}`, ...p }))}
          columns={[
            { field: "product_id", headerName: "Product ID", flex: 0.8, minWidth: 120 },
            { field: "product_name", headerName: "Product Name", flex: 1.2, minWidth: 180 },
            { field: "review_count", headerName: "Reviews", type: "number", flex: 0.5, minWidth: 90 },
            { field: "average_rating", headerName: "Avg Rating", type: "number", flex: 0.5, minWidth: 100 },
          ]}
        />
      </div>

      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">User Reviews Drilldown</h3>
        <DataGridTable
          isDark={isDark}
          height={520}
          rows={rows}
          columns={[
            { field: "created_at", headerName: "Created At", flex: 0.8, minWidth: 150 },
            { field: "username", headerName: "User", flex: 0.6, minWidth: 110 },
            { field: "product_id", headerName: "Product ID", flex: 0.7, minWidth: 120 },
            { field: "product_name", headerName: "Product Name", flex: 1, minWidth: 140 },
            { field: "rating", headerName: "Rating", type: "number", flex: 0.4, minWidth: 80 },
            { field: "recommendation", headerName: "Recommends", flex: 0.5, minWidth: 100 },
            { field: "review_title", headerName: "Title", flex: 0.8, minWidth: 120 },
            { field: "review_text", headerName: "Review", flex: 1.8, minWidth: 250 },
          ]}
        />
      </div>
    </section>
  );
}
