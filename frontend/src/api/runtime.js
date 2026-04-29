import { request } from "./request.js";

export async function inferSingleReview(reviewText, domain = null, productId = null, persist = true) {
  return request("/infer/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: reviewText, domain, product_id: productId, persist }) });
}
export async function inferBatchCsv(file) { const form = new FormData(); form.append("file", file); return request("/infer/csv", { method: "POST", body: form }); }
export async function getJob(jobId) { return request(`/jobs/${jobId}`); }
export async function getReviewGraph(reviewId) { return request(`/graph/review/${reviewId}`); }
export async function getGraphFilterOptions() { return request("/graph/filter-options"); }
export async function getBatchAspectGraph(filters = {}) { const p = new URLSearchParams(); if (filters.domain) p.set("domain", filters.domain); if (filters.product_id) p.set("product_id", filters.product_id); if (filters.from) p.set("from", filters.from); if (filters.to) p.set("to", filters.to); p.set("min_edge_weight", String(filters.min_edge_weight || 1)); if (filters.graph_mode) p.set("graph_mode", filters.graph_mode); return request(`/graph/aspects${p.toString() ? `?${p.toString()}` : ""}`); }
export async function rebuildKg(domain = "") { const q = domain ? `?domain=${encodeURIComponent(domain)}` : ""; return request(`/jobs/kg_rebuild${q}`, { method: "POST" }); }
