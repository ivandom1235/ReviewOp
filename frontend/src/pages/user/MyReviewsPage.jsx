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

  return (
    <UserShell title="My Reviews">
      {error ? <div className="mb-4 rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
      <div className="space-y-3">
        {rows.map((r) => (
          <article key={r.review_id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold text-slate-900 dark:text-slate-100">Product: {r.product_id}</p>
                {r.is_reply ? (
                  <p className="text-xs text-slate-500 dark:text-slate-300">Reply to {r.reply_to_review_title || `review #${r.reply_to_review_id}`}</p>
                ) : null}
                <p className="text-xs text-slate-500 dark:text-slate-300">{new Date(r.review_date).toLocaleString()}</p>
              </div>
              <div className="text-amber-600">{r.rating} {"\u2605"}</div>
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
        {!rows.length && !error ? <p className="text-sm text-slate-600 dark:text-slate-300">No reviews found.</p> : null}
      </div>
    </UserShell>
  );
}
