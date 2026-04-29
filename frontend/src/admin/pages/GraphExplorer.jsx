import { useMemo, useState } from "react";

import AspectGraphView from "../components/graph/AspectGraphView";
import { filterGraphSuggestions } from "./graphFilterUtils";

function SearchableFilterInput({ label, value, onChange, options = [], placeholder, isDark, helperText }) {
  const [open, setOpen] = useState(false);
  const filteredOptions = useMemo(() => filterGraphSuggestions(options, value, 8), [options, value]);

  return (
    <div className="relative min-w-[220px] flex-1 space-y-2 text-sm">
      <span className="font-semibold">{label}</span>
      <input
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          window.setTimeout(() => setOpen(false), 120);
        }}
        placeholder={placeholder}
        autoComplete="off"
        className={`w-full rounded-2xl border px-3 py-2.5 shadow-sm transition focus:outline-none focus:ring-2 focus:ring-emerald-400/30 ${
          isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-white text-slate-900"
        }`}
      />
      {helperText ? <p className={`text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>{helperText}</p> : null}
      {open ? (
        <div
          className={`absolute left-0 right-0 top-[72px] z-20 max-h-56 overflow-auto rounded-2xl border shadow-xl ${
            isDark ? "border-slate-700 bg-slate-950" : "border-slate-200 bg-white"
          }`}
        >
          {filteredOptions.length ? (
            filteredOptions.map((option) => (
              <button
                key={option}
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onChange(option);
                  setOpen(false);
                }}
                className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm transition ${
                  isDark ? "text-slate-100 hover:bg-slate-800" : "text-slate-800 hover:bg-slate-100"
                }`}
              >
                <span>{option}</span>
                <span className={`text-[11px] uppercase tracking-[0.16em] ${isDark ? "text-slate-500" : "text-slate-400"}`}>select</span>
              </button>
            ))
          ) : (
            <div className={`px-3 py-3 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>No matches</div>
          )}
        </div>
      ) : null}
    </div>
  );
}

export default function GraphExplorer({
  graph,
  graphFilters,
  setGraphFilters,
  onApplyFilters,
  onResetFilters,
  graphLoading,
  filterOptions = { domains: [], product_ids: [] },
  isDark = false,
}) {
  const hasActiveFilters = Boolean(
    String(graphFilters.domain || "").trim() ||
      String(graphFilters.product_id || "").trim() ||
      Number(graphFilters.min_edge_weight || 1) > 1
  );

  return (
    <section className="space-y-4">
      <div className={`rounded-[30px] border p-5 shadow-[0_10px_30px_rgba(15,23,42,0.04)] ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-500">Graph Intelligence</p>
            <h2 className={`mt-1 text-lg font-semibold ${isDark ? "text-slate-100" : "text-slate-900"}`}>Knowledge Graph and Corpus Graph</h2>
            <p className={`mt-1 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
              Explore relationships, co-occurrence, and sentiment signals with focused graph actions.
            </p>
          </div>
          <div className={`rounded-2xl border px-3 py-2 text-xs uppercase tracking-[0.18em] ${isDark ? "border-slate-700 text-slate-400" : "border-slate-200 text-slate-500"}`}>{hasActiveFilters ? "Filtered view" : "Default view"}</div>
        </div>

          <form onSubmit={onApplyFilters} className="space-y-4">
            <div className="flex flex-wrap gap-4">
              <SearchableFilterInput
                label="Domain"
                value={graphFilters.domain}
                onChange={(nextValue) => setGraphFilters((p) => ({ ...p, domain: nextValue }))}
                options={filterOptions.domains}
                placeholder="Start typing a domain"
                isDark={isDark}
                helperText="Suggestions match the corpus exactly once selected."
              />
              <SearchableFilterInput
                label="Product ID"
                value={graphFilters.product_id}
                onChange={(nextValue) => setGraphFilters((p) => ({ ...p, product_id: nextValue }))}
                options={filterOptions.product_ids}
                placeholder="Start typing a product ID"
                isDark={isDark}
                helperText="Pick a product ID from the corpus list."
              />
              <label className="min-w-[220px] space-y-2 text-sm">
                <span className="font-semibold">Minimum edge weight</span>
                <input
                  type="number"
                  min="1"
                  value={graphFilters.min_edge_weight}
                  onChange={(e) => setGraphFilters((p) => ({ ...p, min_edge_weight: Number(e.target.value || 1) }))}
                  className={`w-full rounded-2xl border px-3 py-2.5 shadow-sm transition focus:outline-none focus:ring-2 focus:ring-emerald-400/30 ${
                    isDark ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-white text-slate-900"
                  }`}
                />
                <p className={`text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>Higher values hide weaker co-occurrence edges.</p>
              </label>
              <div className="flex items-end gap-3">
                <button type="submit" disabled={graphLoading} className="rounded-2xl bg-emerald-500 px-5 py-2.5 font-semibold text-slate-950 shadow-sm transition hover:bg-emerald-400 disabled:opacity-50">
                  {graphLoading ? "Loading..." : "Apply"}
                </button>
                <button
                  type="button"
                  onClick={onResetFilters}
                  disabled={graphLoading}
                  className={`rounded-2xl border px-5 py-2.5 font-semibold transition disabled:opacity-50 ${
                    isDark ? "border-slate-700 bg-slate-900 text-slate-200 hover:bg-slate-800" : "border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  Reset
                </button>
              </div>
            </div>
            <p className={`text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>
              If a domain or product ID is not suggested, it may not exist in the current corpus slice.
            </p>
          </form>
      </div>
      <AspectGraphView
        graph={graph}
        scope="batch"
        isDark={isDark}
        emptyMessage={
          hasActiveFilters
            ? "No corpus graph data matched the current filters. Clear domain/product ID or lower the minimum edge weight."
            : "Upload batch data to populate the corpus graph."
        }
      />
    </section>
  );
}
