import { useMemo, useState } from "react";

function StatCard({ label, value, isDark }) {
  return (
    <div className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
      <p className="text-xs uppercase tracking-[0.1em] opacity-70">{label}</p>
      <p className="mt-1 text-2xl font-bold">{value}</p>
    </div>
  );
}

export default function UserReviewsInsights({ summary, list, isDark = false, onAnalyzeReview }) {
  const [productFilter, setProductFilter] = useState("");
  const [usernameFilter, setUsernameFilter] = useState("");
  const [minRating, setMinRating] = useState("");
  const [maxRating, setMaxRating] = useState("");

  const rows = useMemo(
    () =>
      (list?.rows || []).map((row) => ({
        id: row.review_id,
        ...row,
      })),
    [list]
  );

  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      const matchesProduct = !productFilter || String(row.product_id || "").toLowerCase().includes(productFilter.toLowerCase()) || String(row.product_name || "").toLowerCase().includes(productFilter.toLowerCase());
      const matchesUser = !usernameFilter || String(row.username || "").toLowerCase().includes(usernameFilter.toLowerCase());
      const rating = Number(row.rating ?? 0);
      const matchesMin = minRating === "" || rating >= Number(minRating);
      const matchesMax = maxRating === "" || rating <= Number(maxRating);
      return matchesProduct && matchesUser && matchesMin && matchesMax;
    });
  }, [rows, productFilter, usernameFilter, minRating, maxRating]);

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
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {(summary?.top_products || []).length ? (
            summary.top_products.map((p, idx) => (
              <div key={`${p.product_id}-${idx}`} className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
                <p className="font-semibold">{p.product_name || p.product_id}</p>
                <p className="text-sm">Product: {p.product_id || "-"}</p>
                <p className="text-sm">Reviews: {p.review_count ?? 0}</p>
                <p className="text-sm">Avg rating: {p.average_rating ?? 0}</p>
                <p className="text-sm">Negative driver: {p.top_negative_aspect || "Unavailable"}</p>
              </div>
            ))
          ) : (
            <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No top products yet.</p>
          )}
        </div>
      </div>

      <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <h3 className="mb-3 text-lg font-semibold">User Reviews Drilldown</h3>
        <div className="mb-3 grid gap-2 md:grid-cols-2 xl:grid-cols-4">
          <input value={productFilter} onChange={(e) => setProductFilter(e.target.value)} placeholder="Filter by product" className={`rounded-xl border px-3 py-2 text-sm ${isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-slate-50 text-slate-800"}`} />
          <input value={usernameFilter} onChange={(e) => setUsernameFilter(e.target.value)} placeholder="Filter by username" className={`rounded-xl border px-3 py-2 text-sm ${isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-slate-50 text-slate-800"}`} />
          <input type="number" min="0" max="5" value={minRating} onChange={(e) => setMinRating(e.target.value)} placeholder="Min rating" className={`rounded-xl border px-3 py-2 text-sm ${isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-slate-50 text-slate-800"}`} />
          <input type="number" min="0" max="5" value={maxRating} onChange={(e) => setMaxRating(e.target.value)} placeholder="Max rating" className={`rounded-xl border px-3 py-2 text-sm ${isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-slate-50 text-slate-800"}`} />
        </div>
        <div className="space-y-3">
          {filteredRows.length ? (
            filteredRows.map((row) => (
              <div key={row.id} className={`rounded-xl p-3 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-semibold">{row.review_title || "Untitled review"}</p>
                  <span className={`rounded-full px-2 py-0.5 text-xs ${isDark ? "bg-slate-800 text-slate-300" : "bg-slate-200 text-slate-700"}`}>Rating: {row.rating ?? "-"}</span>
                </div>
                <p className="mt-1 text-sm">{row.review_text || "-"}</p>
                <p className={`mt-2 text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                  {row.username || "Anonymous"} | {row.product_name || row.product_id || "-"} | {row.recommendation === true ? "Recommends" : row.recommendation === false ? "Does not recommend" : "No recommendation"} | {row.created_at || "-"}
                </p>
                <button type="button" onClick={() => onAnalyzeReview?.(row.review_text || "")} className="mt-2 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white">Analyze this review</button>
              </div>
            ))
          ) : (
            <div className={`grid h-40 place-items-center rounded-xl border ${isDark ? "border-slate-800 bg-slate-950 text-slate-400" : "border-slate-200 bg-slate-50 text-slate-500"}`}>
              No reviews match the current filters.
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
