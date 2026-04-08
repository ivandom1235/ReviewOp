import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getProductSuggestions } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";
import ProductCard from "../../components/user/ProductCard";

export default function UserHomePage() {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState({ recently_reviewed: [], similar_products: [] });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const { token } = useAuth();
  const nav = useNavigate();

  useEffect(() => {
    setLoading(true);
    getProductSuggestions(token)
      .then(setSuggestions)
      .catch((ex) => setError(ex.message || "Failed to load suggestions"))
      .finally(() => setLoading(false));
  }, [token]);

  function onSearch(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    nav(`/search?q=${encodeURIComponent(q)}&min_rating=0&sort=most_recent`);
  }

  return (
    <UserShell title="Discover Products">
      {error ? <div className="mb-4 rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
      
      <form onSubmit={onSearch} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">Search by product name or product ID</label>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            placeholder="e.g. iPhone 14, SKU-1001"
          />
          <button className="rounded-lg bg-emerald-600 px-4 py-2 text-white hover:bg-emerald-700 transition-colors">Search</button>
          <button type="button" onClick={() => nav("/create-review")} className="rounded-lg border border-emerald-600 px-4 py-2 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-50 dark:hover:bg-emerald-900/30 transition-colors">
            Create Review
          </button>
        </div>
      </form>

      {loading ? (
        <div className="mt-8 flex justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-emerald-600 border-t-transparent"></div>
        </div>
      ) : (
        <>
          <section className="mt-8">
            <h2 className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">Products You Reviewed</h2>
            <div className="grid gap-3 md:grid-cols-2">
              {suggestions.recently_reviewed.length ? (
                suggestions.recently_reviewed.map((p) => <ProductCard key={`recent-${p.product_id}`} product={p} />)
              ) : (
                <p className="text-sm text-slate-600 dark:text-slate-300">No reviewed products yet.</p>
              )}
            </div>
          </section>

          <section className="mt-8">
            <h2 className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">Suggested Products</h2>
            <div className="grid gap-3 md:grid-cols-2">
              {suggestions.similar_products.length ? (
                suggestions.similar_products.map((p) => <ProductCard key={`sim-${p.product_id}`} product={p} />)
              ) : (
                <p className="text-sm text-slate-600 dark:text-slate-300">No suggestions available.</p>
              )}
            </div>
          </section>
        </>
      )}
    </UserShell>
  );
}
