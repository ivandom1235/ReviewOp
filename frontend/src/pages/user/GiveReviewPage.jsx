import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { getProductDetail, getProductReviews, submitReview } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";

export default function GiveReviewPage() {
  const { token } = useAuth();
  const nav = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();

  const replyToReviewId = Number(params.get("reply_to_review_id") || location.state?.replyToReviewId || 0);
  const productId = params.get("product_id") || location.state?.productId || "";
  const [productName, setProductName] = useState(location.state?.productName || "");
  const [parentReview, setParentReview] = useState(location.state?.parentReview || null);
  const [rating, setRating] = useState(5);
  const [reviewTitle, setReviewTitle] = useState("");
  const [reviewText, setReviewText] = useState("");
  const [pros, setPros] = useState("");
  const [cons, setCons] = useState("");
  const [recommendation, setRecommendation] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const ready = useMemo(() => Boolean(replyToReviewId && productId), [replyToReviewId, productId]);

  useEffect(() => {
    if (!ready) return;
    if (!productName) {
      getProductDetail(token, productId)
        .then((detail) => setProductName(detail?.name || ""))
        .catch(() => {});
    }
    if (!parentReview) {
      getProductReviews(token, productId, { page: 1, page_size: 100, min_rating: 1, sort: "most_recent" })
        .then((rows) => {
          const match = (rows || []).find((row) => Number(row.review_id) === Number(replyToReviewId));
          if (match) setParentReview(match);
        })
        .catch((ex) => setError(ex.message || "Failed to load parent review"));
    }
  }, [ready, token, productId, productName, parentReview, replyToReviewId]);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    if (!ready) {
      setError("Missing parent review context.");
      return;
    }
    if (!reviewText.trim()) {
      setError("Reply text is required.");
      return;
    }

    setLoading(true);
    try {
      await submitReview(token, {
        product_id: productId,
        product_name: productName || null,
        reply_to_review_id: replyToReviewId,
        rating,
        review_title: reviewTitle || null,
        review_text: reviewText,
        pros: pros || null,
        cons: cons || null,
        recommendation,
      });
      nav(`/products/${encodeURIComponent(productId)}`);
    } catch (ex) {
      setError(ex.message || "Submit failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <UserShell title="Give Review">
      <form onSubmit={onSubmit} className="mx-auto w-full max-w-3xl space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        {error ? <div className="rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
          <p className="font-semibold">Replying to</p>
          <p>Product: {productName || productId || "Unknown"}</p>
          <p className="mt-1">Original review: {parentReview?.review_title || `#${replyToReviewId}`}</p>
          {parentReview?.review_text ? <p className="mt-1 line-clamp-3">{parentReview.review_text}</p> : null}
          <Link className="mt-2 inline-block text-sm font-semibold text-indigo-600 hover:text-indigo-500" to={`/products/${encodeURIComponent(productId)}`}>
            Back to product
          </Link>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <input value={productName} disabled className="w-full rounded-lg border border-slate-300 bg-slate-100 px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
          <input value={productId} disabled className="w-full rounded-lg border border-slate-300 bg-slate-100 px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <select value={rating} onChange={(e) => setRating(Number(e.target.value))} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          <option value={5}>5 stars</option>
          <option value={4}>4 stars</option>
          <option value={3}>3 stars</option>
          <option value={2}>2 stars</option>
          <option value={1}>1 star</option>
        </select>
        <input value={reviewTitle} onChange={(e) => setReviewTitle(e.target.value)} placeholder="Reply title (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <textarea value={reviewText} onChange={(e) => setReviewText(e.target.value)} rows={6} placeholder="Write your reply..." className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <input value={pros} onChange={(e) => setPros(e.target.value)} placeholder="Pros (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <input value={cons} onChange={(e) => setCons(e.target.value)} placeholder="Cons (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
          <input type="checkbox" checked={recommendation} onChange={(e) => setRecommendation(e.target.checked)} />
          I recommend this product
        </label>
        <button disabled={loading} className="rounded-lg bg-indigo-600 px-4 py-2 text-white disabled:opacity-60">
          {loading ? "Submitting..." : "Submit Reply"}
        </button>
      </form>
    </UserShell>
  );
}
