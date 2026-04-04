import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { getProductDetail, getProductReviews } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";

export default function ProductPage() {
  const { productId } = useParams();
  const { token } = useAuth();
  const [params, setParams] = useSearchParams();
  const [detail, setDetail] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState("");
  const minRating = Number(params.get("min_rating") || "1");
  const sort = params.get("sort") || "most_recent";
  const pageSize = 5;

  useEffect(() => {
    if (!productId) return;
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

  return (
    <UserShell title={detail?.name || "Product"}>
      <div className="page-fade-in space-y-6">
        {error ? <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">{error}</div> : null}
        {detail ? (
          <section className="glass-card rounded-xl p-4">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="text-sm text-muted-main">ID: {detail.product_id}</p>
                <p className="text-sm text-muted-main">{detail.category || "General"}</p>
                {detail.summary ? <p className="mt-2 text-sm text-[hsl(var(--text-main))]">{detail.summary}</p> : null}
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-amber-500">{Number(detail.average_rating).toFixed(1)} {"\u2605"}</div>
                <div className="text-sm text-muted-main">{detail.review_count} reviews</div>
              </div>
            </div>

            <div className="mt-4 space-y-2">
              {detail.star_distribution.map((row) => {
                const pct = detail.review_count ? (row.count / detail.review_count) * 100 : 0;
                return (
                  <div key={row.stars} className="flex items-center gap-3 text-sm">
                    <span className="w-12 text-muted-main">{row.stars} stars</span>
                    <div className="h-2 flex-1 rounded bg-[hsl(var(--border-subtle))]">
                      <div className="h-2 rounded bg-amber-500" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="w-8 text-right text-muted-main">{row.count}</span>
                  </div>
                );
              })}
            </div>
            <div className="mt-4">
              <Link
                to={`/products/${encodeURIComponent(detail.product_id)}/review?product_name=${encodeURIComponent(detail.name || "")}`}
                state={{ productName: detail.name || "" }}
                className="rounded-lg premium-gradient px-4 py-2 text-white transition-opacity hover:opacity-90"
              >
                Write Review
              </Link>
            </div>
          </section>
        ) : null}

        <section className="glass-card rounded-xl p-4">
          <div className="mb-4 flex flex-wrap gap-3">
            <select value={minRating} onChange={(e) => setParams({ min_rating: e.target.value, sort })} className="rounded-lg border border-border-subtle bg-[hsla(var(--bg-surface)/0.85)] px-3 py-2 text-[hsl(var(--text-main))]">
              <option value={4}>4 stars and above</option>
              <option value={3}>3 stars and above</option>
              <option value={2}>2 stars and above</option>
              <option value={1}>1 star and above</option>
            </select>
            <select value={sort} onChange={(e) => setParams({ min_rating: String(minRating), sort: e.target.value })} className="rounded-lg border border-border-subtle bg-[hsla(var(--bg-surface)/0.85)] px-3 py-2 text-[hsl(var(--text-main))]">
              <option value="most_recent">Most Recent</option>
              <option value="most_helpful">Most Helpful</option>
              <option value="highest_rated">Highest Rated</option>
              <option value="lowest_rated">Lowest Rated</option>
            </select>
          </div>
          <div className="space-y-3">
            {reviews.map((r) => (
              <article key={r.review_id} className="card-hover-lift rounded-lg border border-border-subtle bg-app/30 p-3">
                <div className="flex items-center justify-between">
                  <div className="font-semibold text-[hsl(var(--text-main))]">{r.reviewer_name}</div>
                  <div className="text-sm text-amber-500">{r.rating} {"\u2605"}</div>
                </div>
                {r.review_title ? <h3 className="mt-1 font-medium text-[hsl(var(--text-main))]">{r.review_title}</h3> : null}
                <p className="mt-1 text-sm text-[hsl(var(--text-main))]">{r.review_text}</p>
                <p className="mt-1 text-xs text-muted-main">{new Date(r.review_date).toLocaleString()}</p>
                {r.aspects?.length ? (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {r.aspects.map((a, idx) => (
                      <span key={`${a.aspect}-${idx}`} className="rounded-full border border-border-subtle bg-app/30 px-2 py-1 text-xs text-muted-main">
                        {a.aspect} {a.sentiment}
                      </span>
                    ))}
                  </div>
                ) : null}
              </article>
            ))}
          </div>
          {hasMore ? (
            <div className="mt-4">
              <button onClick={handleSeeMore} disabled={loadingMore} className="card-hover-lift subtle-pulse rounded-lg border border-border-subtle bg-app/30 px-4 py-2 text-brand-primary disabled:opacity-60">
                {loadingMore ? "Loading..." : "See more"}
              </button>
            </div>
          ) : null}
        </section>
      </div>
    </UserShell>
  );
}
