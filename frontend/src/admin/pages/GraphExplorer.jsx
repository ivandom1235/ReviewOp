import AspectGraphView from "../components/graph/AspectGraphView";

export default function GraphExplorer({ graph, graphFilters, setGraphFilters, onApplyFilters, graphLoading, isDark = false }) {
  return (
    <section className="space-y-4">
      <form onSubmit={onApplyFilters} className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="flex flex-wrap gap-3">
          <input value={graphFilters.domain} onChange={(e) => setGraphFilters((p) => ({ ...p, domain: e.target.value }))} placeholder="Domain" className={`rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-slate-900" : "border-slate-200 bg-white"}`} />
          <input value={graphFilters.product_id} onChange={(e) => setGraphFilters((p) => ({ ...p, product_id: e.target.value }))} placeholder="Product ID" className={`rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-slate-900" : "border-slate-200 bg-white"}`} />
          <input type="number" min="1" value={graphFilters.min_edge_weight} onChange={(e) => setGraphFilters((p) => ({ ...p, min_edge_weight: Number(e.target.value || 1) }))} className={`w-32 rounded-xl border px-3 py-2 ${isDark ? "border-slate-700 bg-slate-900" : "border-slate-200 bg-white"}`} />
          <button type="submit" disabled={graphLoading} className="rounded-xl bg-emerald-500 px-4 py-2 font-semibold text-slate-950">{graphLoading ? "Loading..." : "Apply"}</button>
        </div>
      </form>
      <AspectGraphView graph={graph} scope="batch" isDark={isDark} emptyMessage="Upload batch data to populate the corpus graph." />
    </section>
  );
}
