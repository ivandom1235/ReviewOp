import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { deleteMyReview, getMyReviews } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";

export default function MyReviewsPage() {
  const { token } = useAuth();
  const nav = useNavigate();
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    getMyReviews(token).then(setRows).catch((ex) => setError(ex.message || "Failed to load reviews"));
  }, [token]);

  async function handleDelete(reviewId) {
    try {
      await deleteMyReview(token, reviewId);
      setRows((prev) => prev.filter((r) => r.review_id !== reviewId));
    } catch (ex) {
      setError(ex.message || "Delete failed");
    }
  }

  function handleEditClick(review) {
    const params = new URLSearchParams({
      edit_review_id: String(review.review_id),
      product_id: review.product_id,
    });
    nav(`/create-review?${params.toString()}`, {
        state: {
          prefill: {
            review_id: review.review_id,
            product_id: review.product_id,
            reply_to_review_id: review.reply_to_review_id ?? null,
            rating: review.rating,
            review_title: review.review_title || "",
            review_text: review.review_text || "",
            recommendation: Boolean(review.recommendation),
          },
      },
    });
  }

  const visibleRows = rows.filter((row) => {
    const text = `${row.product_id} ${row.review_title || ""} ${row.review_text || ""}`.toLowerCase();
    const matchesQuery = !query || text.includes(query.toLowerCase());
    const status = row.is_reply ? "reply" : row.recommendation ? "recommend" : "review";
    const matchesStatus = statusFilter === "all" || statusFilter === status;
    return matchesQuery && matchesStatus;
  });

  return (
    <UserShell title="My Reviews">
      {error ? <div className="mb-4 rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
      <div className="mb-4 flex flex-wrap gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search your reviews" className="min-w-[220px] flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          <option value="all">All statuses</option>
          <option value="review">Reviews</option>
          <option value="reply">Replies</option>
          <option value="recommend">Recommended</option>
        </select>
      </div>
      <div className="space-y-3">
        {visibleRows.map((r) => (
          <article key={r.review_id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold text-slate-900 dark:text-slate-100">Product: {r.product_id}</p>
                {r.is_reply ? (
                  <p className="text-xs text-slate-500 dark:text-slate-300">Reply to {r.reply_to_review_title || `review #${r.reply_to_review_id}`}</p>
                ) : null}
                <p className="text-xs text-slate-500 dark:text-slate-300">{new Date(r.review_date).toLocaleString()}</p>
              </div>
              <div className="flex flex-col items-end gap-1">
                <div className="text-amber-600">{r.rating} {"\u2605"}</div>
                <div className="flex gap-2 text-[11px]">
                  <span className={`rounded-full px-2 py-1 ${r.is_reply ? "bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-200" : "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200"}`}>{r.is_reply ? "Reply" : "Review"}</span>
                  <span className={`rounded-full px-2 py-1 ${r.recommendation ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-200" : "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-200"}`}>{r.recommendation ? "Recommended" : "Not recommended"}</span>
                </div>
              </div>
            </div>
            {r.review_title ? <h3 className="mt-2 font-medium text-slate-900 dark:text-slate-100">{r.review_title}</h3> : null}
            <p className="mt-1 text-sm text-slate-700 dark:text-slate-200">{r.review_text}</p>
            <div className="mt-3 flex gap-2">
              <button type="button" onClick={() => handleEditClick(r)} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-emerald-600 dark:border-slate-700 dark:text-emerald-400 hover:bg-slate-50 dark:hover:bg-slate-800">
                Edit Review
              </button>
              <button type="button" onClick={() => handleDelete(r.review_id)} className="rounded-lg border border-red-300 px-3 py-1.5 text-sm text-red-700 dark:border-red-700 dark:text-red-300 hover:bg-red-50 dark:hover:bg-red-950">
                Delete
              </button>
            </div>
          </article>
        ))}
        {!visibleRows.length && !error ? <p className="text-sm text-slate-600 dark:text-slate-300">No reviews found.</p> : null}
      </div>
    </UserShell>
  );
}
