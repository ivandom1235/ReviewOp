import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { getMyReviewById, getProductDetail, submitReview, updateMyReview } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";

export default function SubmitReviewPage() {
  const { productId } = useParams();
  const [params] = useSearchParams();
  const location = useLocation();
  const { token } = useAuth();
  const nav = useNavigate();

  const editReviewId = params.get("edit_review_id");
  const prefill = location.state?.prefill || null;
  const initialProductId = productId || params.get("product_id") || prefill?.product_id || "";
  const initialProductName = params.get("product_name") || location.state?.productName || "";

  const [productRef, setProductRef] = useState(initialProductId);
  const [productName, setProductName] = useState(initialProductName);
  const [rating, setRating] = useState(prefill?.rating ?? 5);
  const [reviewText, setReviewText] = useState(prefill?.review_text ?? "");
  const [reviewTitle, setReviewTitle] = useState(prefill?.review_title ?? "");
  const [pros, setPros] = useState(prefill?.pros ?? "");
  const [cons, setCons] = useState(prefill?.cons ?? "");
  const [recommendation, setRecommendation] = useState(prefill?.recommendation ?? false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const isEditMode = useMemo(() => Boolean(editReviewId), [editReviewId]);

  useEffect(() => {
    document.title = isEditMode ? "Edit Review — ReviewOp" : "Write Review — ReviewOp";
  }, [isEditMode]);

  useEffect(() => {
    if (productId) setProductRef(productId);
  }, [productId]);

  useEffect(() => {
    if (!editReviewId || prefill) return;
    getMyReviewById(token, Number(editReviewId))
      .then((row) => {
        if (!row) {
          setError("Review not found.");
          return;
        }
        setProductRef(row.product_id || "");
        setRating(row.rating ?? 5);
        setReviewTitle(row.review_title || "");
        setReviewText(row.review_text || "");
        setRecommendation(Boolean(row.recommendation));
      })
      .catch((ex) => setError(ex.message || "Failed to load review for editing"));
  }, [token, editReviewId, prefill]);

  useEffect(() => {
    const cleanProductId = (productRef || "").trim();
    if (!cleanProductId || (productName || "").trim()) return;
    const controller = new AbortController();
    getProductDetail(token, cleanProductId, { signal: controller.signal })
      .then((detail) => {
        if (detail?.name) setProductName(detail.name);
      })
      .catch(() => {});
    return () => controller.abort();
  }, [token, productRef, productName]);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    const cleanProductId = (productRef || "").trim();
    if (!cleanProductId) {
      setError("Product reference is required.");
      return;
    }
    if (!reviewText.trim()) {
      setError("Review text is required.");
      return;
    }

    const payload = {
      product_id: cleanProductId,
      product_name: productName.trim() || null,
      rating,
      review_text: reviewText,
      review_title: reviewTitle || null,
      pros: pros || null,
      cons: cons || null,
      recommendation,
    };

    setLoading(true);
    try {
      if (isEditMode) {
        await updateMyReview(token, Number(editReviewId), payload);
        nav("/my-reviews");
      } else {
        await submitReview(token, payload);
        nav(`/products/${encodeURIComponent(cleanProductId)}`);
      }
    } catch (ex) {
      setError(ex.message || (isEditMode ? "Update failed" : "Submit failed"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <UserShell title={isEditMode ? "Edit Review" : "Write Review"}>
      <form onSubmit={onSubmit} className="mx-auto w-full max-w-3xl space-y-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900" id="review-form">
        {error ? <div className="rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200" role="alert">{error}</div> : null}
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <label htmlFor="review-product-name" className="sr-only">Product name</label>
            <input id="review-product-name" value={productName} onChange={(e) => setProductName(e.target.value)} placeholder="Product name (e.g. iPhone 14)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
          </div>
          <div>
            <label htmlFor="review-product-id" className="sr-only">Product ID</label>
            <input id="review-product-id" value={productRef} onChange={(e) => setProductRef(e.target.value)} placeholder="Product ID (e.g. SKU-1001)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
          </div>
        </div>
        <div>
          <label htmlFor="review-rating" className="sr-only">Rating</label>
          <select id="review-rating" value={rating} onChange={(e) => setRating(Number(e.target.value))} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
            <option value={5}>5 stars</option>
            <option value={4}>4 stars</option>
            <option value={3}>3 stars</option>
            <option value={2}>2 stars</option>
            <option value={1}>1 star</option>
          </select>
        </div>
        <div>
          <label htmlFor="review-title" className="sr-only">Review title</label>
          <input id="review-title" value={reviewTitle} onChange={(e) => setReviewTitle(e.target.value)} placeholder="Review title (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <div>
          <label htmlFor="review-text" className="sr-only">Review text</label>
          <textarea id="review-text" value={reviewText} onChange={(e) => setReviewText(e.target.value)} rows={6} placeholder="Share your experience..." className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <div>
          <label htmlFor="review-pros" className="sr-only">Pros</label>
          <input id="review-pros" value={pros} onChange={(e) => setPros(e.target.value)} placeholder="Pros (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <div>
          <label htmlFor="review-cons" className="sr-only">Cons</label>
          <input id="review-cons" value={cons} onChange={(e) => setCons(e.target.value)} placeholder="Cons (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200" htmlFor="review-recommendation">
          <input id="review-recommendation" type="checkbox" checked={recommendation} onChange={(e) => setRecommendation(e.target.checked)} />
          I recommend this product
        </label>
        <button id="btn-submit-review" disabled={loading} className="rounded-lg bg-emerald-600 px-4 py-2 text-white transition-colors hover:bg-emerald-700 disabled:opacity-60">
          {loading ? (isEditMode ? "Updating..." : "Submitting...") : isEditMode ? "Update Review" : "Submit Review"}
        </button>
      </form>
    </UserShell>
  );
}
