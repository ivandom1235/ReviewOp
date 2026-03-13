import { useEffect, useState } from "react";
import { deleteMyReview, getMyReviews, updateMyReview } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";

export default function MyReviewsPage() {
  const { token } = useAuth();
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editForm, setEditForm] = useState(null);

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
    setEditingId(review.review_id);
    setEditForm({
      product_id: review.product_id,
      rating: review.rating,
      review_title: review.review_title || "",
      review_text: review.review_text || "",
      pros: review.pros || "",
      cons: review.cons || "",
      recommendation: review.recommendation || false,
    });
  }

  async function handleUpdateSubmit(e) {
    e.preventDefault();
    if (!editForm.review_text.trim()) {
      setError("Review text is required.");
      return;
    }
    setError("");
    try {
      await updateMyReview(token, editingId, editForm);
      setRows((prev) =>
        prev.map((r) =>
          r.review_id === editingId
            ? { ...r, ...editForm, review_date: new Date().toISOString() }
            : r
        )
      );
      setEditingId(null);
    } catch (ex) {
      setError(ex.message || "Update failed");
    }
  }

  return (
    <UserShell title="My Reviews">
      {error ? <div className="mb-4 rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
      <div className="space-y-3">
        {rows.map((r) => (
          <article key={r.review_id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
            {editingId === r.review_id ? (
              <form onSubmit={handleUpdateSubmit} className="space-y-3">
                <div className="flex justify-between items-center text-sm font-semibold text-slate-700 dark:text-slate-200">
                  <span>Editing Review for Product: {r.product_id}</span>
                </div>
                <select value={editForm.rating} onChange={(e) => setEditForm({...editForm, rating: Number(e.target.value)})} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
                  <option value={5}>5 stars</option>
                  <option value={4}>4 stars</option>
                  <option value={3}>3 stars</option>
                  <option value={2}>2 stars</option>
                  <option value={1}>1 star</option>
                </select>
                <input value={editForm.review_title} onChange={(e) => setEditForm({...editForm, review_title: e.target.value})} placeholder="Title (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
                <textarea value={editForm.review_text} onChange={(e) => setEditForm({...editForm, review_text: e.target.value})} rows={4} placeholder="Review text" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
                <input value={editForm.pros} onChange={(e) => setEditForm({...editForm, pros: e.target.value})} placeholder="Pros (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
                <input value={editForm.cons} onChange={(e) => setEditForm({...editForm, cons: e.target.value})} placeholder="Cons (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
                  <input type="checkbox" checked={editForm.recommendation} onChange={(e) => setEditForm({...editForm, recommendation: e.target.checked})} />
                  I recommend this product
                </label>
                <div className="flex gap-2">
                  <button type="submit" className="rounded-lg bg-emerald-600 px-4 py-2 text-white text-sm">Save Changes</button>
                  <button type="button" onClick={() => setEditingId(null)} className="rounded-lg border border-slate-300 px-4 py-2 text-slate-700 dark:border-slate-600 dark:text-slate-300 text-sm">Cancel</button>
                </div>
              </form>
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-slate-900 dark:text-slate-100">Product: {r.product_id}</p>
                    <p className="text-xs text-slate-500 dark:text-slate-300">{new Date(r.review_date).toLocaleString()}</p>
                  </div>
                  <div className="text-amber-600">{r.rating} ★</div>
                </div>
                {r.review_title ? <h3 className="mt-2 font-medium text-slate-900 dark:text-slate-100">{r.review_title}</h3> : null}
                <p className="mt-1 text-sm text-slate-700 dark:text-slate-200">{r.review_text}</p>
                <div className="mt-3 flex gap-2">
                  <button type="button" onClick={() => handleEditClick(r)} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-emerald-600 dark:border-slate-700 dark:text-emerald-400 hover:bg-slate-50 dark:hover:bg-slate-800">
                    Edit
                  </button>
                  <button type="button" onClick={() => handleDelete(r.review_id)} className="rounded-lg border border-red-300 px-3 py-1.5 text-sm text-red-700 dark:border-red-700 dark:text-red-300 hover:bg-red-50 dark:hover:bg-red-950">
                    Delete
                  </button>
                </div>
              </>
            )}
          </article>
        ))}
        {!rows.length && !error ? <p className="text-sm text-slate-600 dark:text-slate-300">No reviews found.</p> : null}
      </div>
    </UserShell>
  );
}
