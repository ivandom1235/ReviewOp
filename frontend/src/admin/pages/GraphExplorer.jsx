import AspectGraphView from "../components/graph/AspectGraphView";

export default function GraphExplorer({ graph, graphFilters, setGraphFilters, onApplyFilters, graphLoading, isDark = false }) {
  return (
    <section className="space-y-4">
      <form onSubmit={onApplyFilters} className={`app-card rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="flex flex-wrap gap-3">
          <input value={graphFilters.domain} onChange={(e) => setGraphFilters((p) => ({ ...p, domain: e.target.value.trim() }))} placeholder="Domain" className="app-input w-44" />
          <input value={graphFilters.product_id} onChange={(e) => setGraphFilters((p) => ({ ...p, product_id: e.target.value.trim() }))} placeholder="Product ID" className="app-input w-44" />
          <select
            value={graphFilters.graph_mode || "accepted"}
            onChange={(e) => setGraphFilters((p) => ({ ...p, graph_mode: e.target.value }))}
            className="app-input w-44"
          >
            <option value="accepted">Accepted Graph</option>
            <option value="novel_side">Novel Side Channel</option>
          </select>
          <input type="number" min="1" value={graphFilters.min_edge_weight} onChange={(e) => setGraphFilters((p) => ({ ...p, min_edge_weight: Math.max(1, Number(e.target.value || 1)) }))} className="app-input w-28" />
          <button type="submit" disabled={graphLoading} className="app-btn app-btn-primary">{graphLoading ? "Loading..." : "Apply"}</button>
        </div>
      </form>
      <AspectGraphView graph={graph} scope="batch" isDark={isDark} emptyMessage="Upload batch data to populate the corpus graph." />
    </section>
  );
}
