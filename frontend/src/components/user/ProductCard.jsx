import { Link } from "react-router-dom";

export default function ProductCard({ product }) {
  return (
    <article className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">{product.name}</h3>
          <p className="text-sm text-slate-500 dark:text-slate-300">ID: {product.product_id}</p>
          {product.category ? <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{product.category}</p> : null}
        </div>
        <div className="text-right text-sm">
          <div className="font-semibold text-amber-600">{Number(product.average_rating || 0).toFixed(1)} ★</div>
          <div className="text-slate-500 dark:text-slate-300">{product.review_count || 0} reviews</div>
        </div>
      </div>
      {product.summary ? <p className="mt-3 text-sm text-slate-700 dark:text-slate-200">{product.summary}</p> : null}
      <div className="mt-4">
        <Link to={`/products/${encodeURIComponent(product.product_id)}`} className="rounded-lg bg-emerald-600 px-3 py-2 text-sm font-medium text-white hover:bg-emerald-700">
          View Reviews
        </Link>
      </div>
    </article>
  );
}
