import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { searchProducts } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import ProductCard from "../../components/user/ProductCard";
import UserShell from "../../components/user/UserShell";

const SORT_OPTIONS = [
  { value: "most_recent", label: "Most Recent" },
  { value: "most_helpful", label: "Most Helpful" },
  { value: "highest_rated", label: "Highest Rated" },
  { value: "lowest_rated", label: "Lowest Rated" },
];

export default function SearchResultsPage() {
  const { token } = useAuth();
  const [params, setParams] = useSearchParams();
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  const q = params.get("q") || "";
  const minRating = Number(params.get("min_rating") ?? "0");
  const sort = params.get("sort") || "most_recent";
  const limit = 10;

  // Local state for the search input to allow debouncing
  const [searchInput, setSearchInput] = useState(q);

  useEffect(() => {
    // Reset when keywords or filters change
    setError("");
    setLoading(true);
    searchProducts(token, { q, min_rating: minRating, sort, offset: 0 })
      .then((data) => {
        setRows(data);
        setHasMore(data.length === limit);
      })
      .catch((ex) => setError(ex.message || "Search failed"))
      .finally(() => setLoading(false));
  }, [token, q, minRating, sort]);

  useEffect(() => {
    setSearchInput(q);
  }, [q]);

  // Debounce the search input update to the URL params
  useEffect(() => {
    const handler = setTimeout(() => {
      if (searchInput !== q) {
        setParams({ q: searchInput, min_rating: String(minRating), sort });
      }
    }, 400);
    return () => clearTimeout(handler);
  }, [searchInput, q, minRating, sort, setParams]);

  async function handleLoadMore() {
    if (loading || !hasMore) return;
    setLoading(true);
    try {
      const more = await searchProducts(token, { q, min_rating: minRating, sort, offset: rows.length });
      setRows((prev) => [...prev, ...more]);
      setHasMore(more.length === limit);
    } catch (ex) {
      setError(ex.message || "Failed to load more");
    } finally {
      setLoading(false);
    }
  }

  return (
    <UserShell title="Search Results">
      <div className="flex flex-wrap gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="relative flex-1 min-w-[220px]">
          <input
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            placeholder="Search products"
          />
          {searchInput !== q && (
            <div className="absolute right-3 top-2.5 h-4 w-4 animate-spin rounded-full border-2 border-emerald-600 border-t-transparent"></div>
          )}
        </div>
        <select value={minRating} onChange={(e) => setParams({ q, min_rating: e.target.value, sort })} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          <option value={4}>4 stars and above</option>
          <option value={3}>3 stars and above</option>
          <option value={2}>2 stars and above</option>
          <option value={1}>1 star and above</option>
        </select>
        <select value={sort} onChange={(e) => setParams({ q, min_rating: String(minRating), sort: e.target.value })} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      {loading && !rows.length ? (
        <div className="mt-8 flex justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-emerald-600 border-t-transparent"></div>
        </div>
      ) : (
        <>
          {error ? <div className="mt-4 rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
          
          <div className="mt-6 grid gap-3 md:grid-cols-2">
            {rows.map((p) => (
              <ProductCard key={p.product_id} product={p} />
            ))}
          </div>

          {!rows.length && !error && !loading ? (
            <p className="mt-6 text-sm text-slate-600 dark:text-slate-300">No products found.</p>
          ) : null}

          {hasMore && rows.length > 0 && (
            <div className="mt-8 flex justify-center">
              <button
                onClick={handleLoadMore}
                disabled={loading}
                className="rounded-lg bg-slate-100 px-6 py-2 text-sm font-medium text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700 disabled:opacity-50"
              >
                {loading ? "Loading..." : "Load More Products"}
              </button>
            </div>
          )}
        </>
      )}
    </UserShell>
  );
}
