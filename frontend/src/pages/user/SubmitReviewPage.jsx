import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { getMyReviewById, getMyReviewJob, getProductDetail, submitReview, updateMyReview } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";
import { getComposerMode, isComposerLocked } from "./reviewComposerState";
import { pollReviewJobUntilTerminal } from "./reviewJobStatus";

const DRAFT_KEY = "reviewop-review-draft";

function loadDraft() {
  try {
    return JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
  } catch {
    return null;
  }
}

export default function SubmitReviewPage() {
  const { productId } = useParams();
  const [params] = useSearchParams();
  const location = useLocation();
  const { token } = useAuth();
  const nav = useNavigate();

  const editReviewId = params.get("edit_review_id");
  const prefill = location.state?.prefill || null;
  const hasPrefill = Boolean(prefill);
  const initialProductId = productId || params.get("product_id") || prefill?.product_id || "";
  const initialProductName = params.get("product_name") || location.state?.productName || "";
  const draft = useMemo(() => {
    // Drafts are for "create" only; never blend them into edit/prefill flows.
    if (editReviewId || hasPrefill) return null;
    return loadDraft();
  }, [editReviewId, hasPrefill]);

  const [productRef, setProductRef] = useState(prefill?.product_id || draft?.productRef || initialProductId);
  const [productName, setProductName] = useState(prefill?.product_name || draft?.productName || initialProductName);
  const [replyToReviewId, setReplyToReviewId] = useState(prefill?.reply_to_review_id ?? draft?.replyToReviewId ?? null);
  const [rating, setRating] = useState(prefill?.rating ?? draft?.rating ?? 5);
  const [reviewText, setReviewText] = useState(prefill?.review_text ?? draft?.reviewText ?? "");
  const [reviewTitle, setReviewTitle] = useState(prefill?.review_title ?? draft?.reviewTitle ?? "");
  const [pros, setPros] = useState(prefill?.pros ?? draft?.pros ?? "");
  const [cons, setCons] = useState(prefill?.cons ?? draft?.cons ?? "");
  const [recommendation, setRecommendation] = useState(prefill?.recommendation ?? draft?.recommendation ?? false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState(null);

  const isEditMode = useMemo(() => Boolean(editReviewId), [editReviewId]);
  const composerMode = getComposerMode({ isEditMode, replyToReviewId });
  const productLocked = isComposerLocked({ isEditMode, replyToReviewId });

  useEffect(() => {
    if (productId) setProductRef(productId);
  }, [productId]);

  useEffect(() => {
    if (prefill || isEditMode) return;
    localStorage.setItem(
      DRAFT_KEY,
      JSON.stringify({
        productRef,
        productName,
        replyToReviewId,
        rating,
        reviewText,
        reviewTitle,
        pros,
        cons,
        recommendation,
      }),
    );
  }, [productRef, productName, replyToReviewId, rating, reviewText, reviewTitle, pros, cons, recommendation, prefill, isEditMode]);

  useEffect(() => {
    if (!editReviewId || prefill) return;
    getMyReviewById(token, Number(editReviewId))
      .then((row) => {
        if (!row) {
          setError("Review not found.");
          return;
        }
        setProductRef(row.product_id || "");
        setReplyToReviewId(row.reply_to_review_id ?? null);
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
    getProductDetail(token, cleanProductId)
      .then((detail) => {
        if (detail?.name) setProductName(detail.name);
      })
      .catch(() => {
        // Best effort prefill.
      });
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
      reply_to_review_id: replyToReviewId,
    };

    setLoading(true);
    try {
      if (isEditMode) {
        await updateMyReview(token, Number(editReviewId), payload);
        nav("/my-reviews");
      } else {
        const result = await submitReview(token, payload);
        localStorage.removeItem(DRAFT_KEY);
        if (result?.job_id) {
          setAnalysisStatus({ status: result.analysis_status || "queued", jobId: result.job_id });
          const finalStatus = await pollReviewJobUntilTerminal({
            jobId: result.job_id,
            fetchStatus: (jobId) => getMyReviewJob(token, jobId),
            onStatus: (status, payload) => setAnalysisStatus({ status, jobId: result.job_id, error: payload?.error }),
          });
          if (String(finalStatus?.status || "").toLowerCase() === "failed") {
            setError(finalStatus?.error || "Review submitted, but analysis failed.");
            return;
          }
        }
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
      <form onSubmit={onSubmit} className="mx-auto w-full max-w-3xl space-y-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        {error ? <div className="rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
        {analysisStatus ? (
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-100">
            Analysis {String(analysisStatus.status || "queued").replace("_", " ")}.
          </div>
        ) : null}
        {productLocked ? (
          <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm dark:border-slate-700 dark:bg-slate-800">
            <div className="grid gap-1 md:grid-cols-2">
              <div>
                <p className="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Product name</p>
                <p className="mt-1 font-medium text-slate-900 dark:text-slate-100">{productName || "Unknown product"}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Product ID</p>
                <p className="mt-1 font-medium text-slate-900 dark:text-slate-100">{productRef || "Unknown"}</p>
              </div>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {composerMode === "reply"
                ? "This reply is locked to the selected review and product."
                : "This review is locked to your original product choice."}
            </p>
          </div>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            <input value={productName} onChange={(e) => setProductName(e.target.value)} placeholder="Product name (e.g. iPhone 14)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
            <input value={productRef} onChange={(e) => setProductRef(e.target.value)} placeholder="Product ID (e.g. SKU-1001)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
          </div>
        )}
        <select value={rating} onChange={(e) => setRating(Number(e.target.value))} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          <option value={5}>5 stars</option>
          <option value={4}>4 stars</option>
          <option value={3}>3 stars</option>
          <option value={2}>2 stars</option>
          <option value={1}>1 star</option>
        </select>
        <input value={reviewTitle} onChange={(e) => setReviewTitle(e.target.value)} placeholder="Review title (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <textarea value={reviewText} onChange={(e) => setReviewText(e.target.value)} rows={6} placeholder="Share your experience..." className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <input value={pros} onChange={(e) => setPros(e.target.value)} placeholder="Pros (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <input value={cons} onChange={(e) => setCons(e.target.value)} placeholder="Cons (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
          <input type="checkbox" checked={recommendation} onChange={(e) => setRecommendation(e.target.checked)} />
          I recommend this product
        </label>
        <button disabled={loading} className="rounded-lg bg-emerald-600 px-4 py-2 text-white disabled:opacity-60">
          {loading ? (isEditMode ? "Updating..." : analysisStatus ? "Analyzing..." : "Submitting...") : isEditMode ? "Update Review" : "Submit Review"}
        </button>
      </form>
    </UserShell>
  );
}
