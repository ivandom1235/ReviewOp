import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getProductSuggestions } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";
import ProductCard from "../../components/user/ProductCard";

function SkeletonCard() {
  return (
    <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-700">
      <div className="skeleton mb-3 h-5 w-3/4" />
      <div className="skeleton mb-2 h-4 w-1/2" />
      <div className="skeleton h-4 w-1/3" />
    </div>
  );
}

export default function UserHomePage() {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState({ recently_reviewed: [], similar_products: [] });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const { token } = useAuth();
  const nav = useNavigate();

  useEffect(() => {
    document.title = "Home — ReviewOp";
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    getProductSuggestions(token, { signal: controller.signal })
      .then(setSuggestions)
      .catch((ex) => {
        if (controller.signal.aborted) return;
        setError(ex.message || "Failed to load suggestions");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [token]);

  function onSearch(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    nav(`/search?q=${encodeURIComponent(q)}&min_rating=0&sort=most_recent`);
  }

  return (
    <UserShell title="Discover Products">
      {error ? <div className="mb-4 rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200" role="alert">{error}</div> : null}
      
      <form onSubmit={onSearch} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900" id="search-form">
        <label htmlFor="search-query" className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">Search by product name or product ID</label>
        <div className="flex gap-2">
          <input
            id="search-query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            placeholder="e.g. iPhone 14, SKU-1001"
          />
          <button id="btn-search" className="rounded-lg bg-emerald-600 px-4 py-2 text-white transition-colors hover:bg-emerald-700">Search</button>
          <button id="btn-create-review" type="button" onClick={() => nav("/create-review")} className="rounded-lg border border-emerald-600 px-4 py-2 text-emerald-700 dark:text-emerald-300 transition-colors hover:bg-emerald-50 dark:hover:bg-emerald-900/30">
            Create Review
          </button>
        </div>
      </form>

      {loading ? (
        <div className="mt-8 space-y-6 animate-fade-in">
          <section>
            <div className="skeleton mb-3 h-6 w-48" />
            <div className="grid gap-3 md:grid-cols-2">
              <SkeletonCard />
              <SkeletonCard />
            </div>
          </section>
          <section>
            <div className="skeleton mb-3 h-6 w-48" />
            <div className="grid gap-3 md:grid-cols-2">
              <SkeletonCard />
              <SkeletonCard />
            </div>
          </section>
        </div>
      ) : (
        <>
          <section className="mt-8">
            <h2 className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">Products You Reviewed</h2>
            <div className="grid gap-3 md:grid-cols-2">
              {suggestions.recently_reviewed.length ? (
                suggestions.recently_reviewed.map((p) => <ProductCard key={`recent-${p.product_id}`} product={p} />)
              ) : (
                <div className="col-span-full rounded-xl border border-dashed border-slate-300 p-6 text-center dark:border-slate-600">
                  <p className="text-sm font-medium text-slate-500 dark:text-slate-400">No reviewed products yet</p>
                  <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">Submit your first review to see it here.</p>
                </div>
              )}
            </div>
          </section>

          <section className="mt-8">
            <h2 className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">Suggested Products</h2>
            <div className="grid gap-3 md:grid-cols-2">
              {suggestions.similar_products.length ? (
                suggestions.similar_products.map((p) => <ProductCard key={`sim-${p.product_id}`} product={p} />)
              ) : (
                <div className="col-span-full rounded-xl border border-dashed border-slate-300 p-6 text-center dark:border-slate-600">
                  <p className="text-sm font-medium text-slate-500 dark:text-slate-400">No suggestions available</p>
                  <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">Suggestions appear as you review more products.</p>
                </div>
              )}
            </div>
          </section>
        </>
      )}
    </UserShell>
  );
}
