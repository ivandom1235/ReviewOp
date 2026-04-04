import { Link } from "react-router-dom";

export default function ProductCard({ product }) {
  return (
    <article className="glass-card card-hover-lift rounded-xl p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-[hsl(var(--text-main))]">{product.name}</h3>
          <p className="text-sm text-muted-main">ID: {product.product_id}</p>
          {product.category ? <p className="mt-1 text-sm text-muted-main">{product.category}</p> : null}
        </div>
        <div className="text-right text-sm">
          <div className="font-semibold text-amber-500">{Number(product.average_rating || 0).toFixed(1)} {"\u2605"}</div>
          <div className="text-muted-main">{product.review_count || 0} reviews</div>
        </div>
      </div>
      {product.summary ? <p className="mt-3 text-sm text-[hsl(var(--text-main))]">{product.summary}</p> : null}
      <div className="mt-4">
        <Link to={`/products/${encodeURIComponent(product.product_id)}`} className="rounded-lg premium-gradient px-3 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90">
          View Reviews
        </Link>
      </div>
    </article>
  );
}
