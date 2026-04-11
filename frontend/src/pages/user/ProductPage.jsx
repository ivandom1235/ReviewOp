import { useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { getProductDetail, getProductReviews } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";
import { DEFAULT_SEARCH_SORT, updateSearchParams } from "./searchState";

export default function ProductPage() {
  const { productId } = useParams();
  const { token } = useAuth();
  const nav = useNavigate();
  const [params, setParams] = useSearchParams();
  const [detail, setDetail] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState("");
  const minRating = Number(params.get("min_rating") || "1");
  const selectedAspect = params.get("aspect") || "";
  const sort = params.get("sort") || DEFAULT_SEARCH_SORT;
  const pageSize = 5;
  const hasActiveFilters = minRating !== 1 || Boolean(selectedAspect) || sort !== "most_recent";

  function updateParams(next) {
    setParams(updateSearchParams(params, next), { replace: true });
  }

  function resetFilters() {
    setError("");
    setReviews([]);
    setHasMore(true);
    setParams(new URLSearchParams(), { replace: true });
  }

  useEffect(() => {
    if (!productId) return;
    setError("");
    setReviews([]);
    setHasMore(true);
    Promise.all([
      getProductDetail(token, productId),
      getProductReviews(token, productId, { page: 1, page_size: pageSize, min_rating: minRating, sort }),
    ])
      .then(([d, r]) => {
        setDetail(d);
        setReviews(r);
        setHasMore((r || []).length === pageSize);
      })
      .catch((ex) => setError(ex.message || "Failed to load product"));
  }, [token, productId, minRating, sort]);

  async function handleSeeMore() {
    if (!productId || loadingMore || !hasMore) return;
    setLoadingMore(true);
    setError("");
    try {
      const nextPage = Math.floor(reviews.length / pageSize) + 1;
      const nextBatch = await getProductReviews(token, productId, { page: nextPage, page_size: pageSize, min_rating: minRating, sort });
      setReviews((prev) => [...prev, ...(nextBatch || [])]);
      setHasMore((nextBatch || []).length === pageSize);
    } catch (ex) {
      setError(ex.message || "Failed to load more reviews");
    } finally {
      setLoadingMore(false);
    }
  }

  const filteredReviews = reviews.filter((review) => {
    const matchesAspect = !selectedAspect || (review.aspects || []).some((a) => String(a.aspect || "").toLowerCase() === selectedAspect.toLowerCase());
    return matchesAspect;
  });

  const aspectCounts = (reviews || []).reduce((acc, review) => {
    (review.aspects || []).forEach((aspect) => {
      const key = aspect.aspect || "";
      if (!key) return;
      acc[key] = acc[key] || { positive: 0, negative: 0, neutral: 0, total: 0 };
      const bucket = acc[key];
      bucket.total += 1;
      const sentiment = String(aspect.sentiment || "").toLowerCase();
      if (sentiment.includes("neg")) bucket.negative += 1;
      else if (sentiment.includes("pos")) bucket.positive += 1;
      else bucket.neutral += 1;
    });
    return acc;
  }, {});

  const topAspects = Object.entries(aspectCounts)
    .map(([aspect, counts]) => ({
      aspect,
      score: counts.negative * 2 + counts.total,
      total: counts.total,
      negative: counts.negative,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);

  const topPraise = [...reviews]
    .flatMap((review) => (review.aspects || []).filter((a) => String(a.sentiment || "").toLowerCase().includes("pos")).map((a) => a.aspect))
    .slice(0, 3);
  const topComplaints = [...reviews]
    .flatMap((review) => (review.aspects || []).filter((a) => String(a.sentiment || "").toLowerCase().includes("neg")).map((a) => a.aspect))
    .slice(0, 3);

  return (
    <UserShell title={detail?.name || "Product"}>
      {error ? <div className="rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
      {detail ? (
        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-300">ID: {detail.product_id}</p>
              <p className="text-sm text-slate-600 dark:text-slate-300">{detail.category || "General"}</p>
              {detail.summary ? <p className="mt-2 text-sm text-slate-700 dark:text-slate-200">{detail.summary}</p> : null}
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-amber-600">{Number(detail.average_rating).toFixed(1)} {"\u2605"}</div>
              <div className="text-sm text-slate-500 dark:text-slate-300">{detail.review_count} reviews</div>
            </div>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="rounded-lg bg-slate-50 p-3 text-sm dark:bg-slate-800">
              <p className="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Top praise</p>
              <p className="mt-1 font-semibold">{topPraise.length ? topPraise.join(", ") : "No strong positive aspects yet"}</p>
            </div>
            <div className="rounded-lg bg-slate-50 p-3 text-sm dark:bg-slate-800">
              <p className="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Top complaints</p>
              <p className="mt-1 font-semibold">{topComplaints.length ? topComplaints.join(", ") : "No strong negative aspects yet"}</p>
            </div>
            <div className="rounded-lg bg-slate-50 p-3 text-sm dark:bg-slate-800">
              <p className="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Active filter</p>
              <p className="mt-1 font-semibold">{selectedAspect || `${minRating}+ stars`}</p>
            </div>
          </div>

          <div className="mt-4 space-y-2">
            {(detail.star_distribution || []).map((row) => {
              const pct = detail.review_count ? (row.count / detail.review_count) * 100 : 0;
              return (
                <div key={row.stars} className="flex items-center gap-3 text-sm">
                  <span className="w-12 text-slate-600 dark:text-slate-300">{row.stars} stars</span>
                  <button
                    type="button"
                    onClick={() => updateParams({ min_rating: String(row.stars), sort, aspect: selectedAspect })}
                    className="h-2 flex-1 rounded bg-slate-200 text-left dark:bg-slate-700"
                    title={`Filter to ${row.stars}+ stars`}
                  >
                    <div className="h-2 rounded bg-amber-500" style={{ width: `${pct}%` }} />
                  </button>
                  <span className="w-8 text-right text-slate-600 dark:text-slate-300">{row.count}</span>
                </div>
              );
            })}
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={resetFilters}
              className={`rounded-full px-3 py-1 text-xs font-semibold ${!hasActiveFilters ? "bg-emerald-600 text-white" : "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200"}`}
            >
              Reset filters
            </button>
          </div>
          <div className="mt-4">
            <Link
              to={`/products/${encodeURIComponent(detail.product_id)}/review?product_name=${encodeURIComponent(detail.name || "")}`}
              state={{ productName: detail.name || "" }}
              className="rounded-lg bg-emerald-600 px-4 py-2 text-white"
            >
              Write Review
            </Link>
          </div>
        </section>
      ) : null}

      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Refine reviews</h2>
          <button
            type="button"
            onClick={resetFilters}
            className={`rounded-full px-3 py-1 text-xs font-semibold ${!hasActiveFilters ? "bg-emerald-600 text-white" : "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200"}`}
          >
            Reset filters
          </button>
        </div>
        <div className="mb-4 flex flex-wrap gap-3">
          <select value={minRating} onChange={(e) => updateParams({ min_rating: e.target.value, sort })} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
            <option value={4}>4 stars and above</option>
            <option value={3}>3 stars and above</option>
            <option value={2}>2 stars and above</option>
            <option value={1}>1 star and above</option>
          </select>
          <select value={selectedAspect} onChange={(e) => updateParams({ min_rating: String(minRating), sort, aspect: e.target.value })} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
            <option value="">All aspects</option>
            {topAspects.map((item) => (
              <option key={item.aspect} value={item.aspect}>
                {item.aspect}
              </option>
            ))}
          </select>
          <select value={sort} onChange={(e) => updateParams({ min_rating: String(minRating), sort: e.target.value })} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
            <option value="most_recent">Most Recent</option>
            <option value="most_helpful">Most Helpful</option>
            <option value="highest_rated">Highest Rated</option>
            <option value="lowest_rated">Lowest Rated</option>
          </select>
        </div>
        <div className="space-y-3">
          {filteredReviews.map((r) => (
            <article key={r.review_id} className="rounded-lg border border-slate-200 p-3 dark:border-slate-700">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-900 dark:text-slate-100">{r.reviewer_name}</div>
                <div className="text-sm text-amber-600">{r.rating} {"\u2605"}</div>
              </div>
              {r.is_reply ? (
                <div className="mt-1 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  Replying to {r.reply_to_review_title || `review #${r.reply_to_review_id}`}
                </div>
              ) : null}
              {r.review_title ? <h3 className="mt-1 font-medium text-slate-900 dark:text-slate-100">{r.review_title}</h3> : null}
              <p className="mt-1 text-sm text-slate-700 dark:text-slate-200">{r.review_text}</p>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-300">{new Date(r.review_date).toLocaleString()}</p>
              {r.aspects?.length ? (
                <div className="mt-2 flex flex-wrap gap-2">
                  {r.aspects.map((a, idx) => (
                    <button
                      key={`${a.aspect}-${idx}`}
                      type="button"
                      onClick={() => updateParams({ min_rating: String(minRating), sort, aspect: a.aspect })}
                      className={`rounded-full px-2 py-1 text-xs ${selectedAspect === a.aspect ? "bg-emerald-600 text-white" : "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200"}`}
                    >
                      {a.aspect} {a.sentiment}
                    </button>
                  ))}
                </div>
              ) : null}
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() =>
                    nav(
                      `/give-review?reply_to_review_id=${encodeURIComponent(r.review_id)}&product_id=${encodeURIComponent(r.product_id)}`,
                      {
                        state: {
                          replyToReviewId: r.review_id,
                          productId: r.product_id,
                          productName: detail?.name || "",
                          parentReview: r,
                        },
                      },
                    )
                  }
                  className="rounded-lg border border-indigo-300 px-3 py-1.5 text-sm text-indigo-700 hover:bg-indigo-50 dark:border-indigo-700 dark:text-indigo-300 dark:hover:bg-indigo-950"
                >
                  Reply to Review
                </button>
              </div>
            </article>
          ))}
        </div>
        {hasMore ? (
          <div className="mt-4">
            <button onClick={handleSeeMore} disabled={loadingMore} className="rounded-lg border border-slate-300 px-4 py-2 text-slate-700 dark:border-slate-600 dark:text-slate-200 disabled:opacity-60">
              {loadingMore ? "Loading..." : "See more"}
            </button>
          </div>
        ) : null}
      </section>
    </UserShell>
  );
}
