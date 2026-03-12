const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error("Backend is unreachable. Start backend API and check VITE_PROXY_TARGET/VITE_API_BASE_URL.");
  }

  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text || null;
  }

  if (!response.ok) {
    throw new Error(payload?.detail || payload?.error || payload || `Request failed: ${response.status}`);
  }

  return payload;
}

export async function inferSingleReview(reviewText, domain = null, productId = null) {
  return request("/infer/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: reviewText,
      domain,
      product_id: productId,
    }),
  });
}

export async function inferBatchCsv(file) {
  const form = new FormData();
  form.append("file", file);

  return request("/infer/csv", {
    method: "POST",
    body: form,
  });
}

export async function getJob(jobId) {
  return request(`/jobs/${jobId}`);
}

export async function getOverview() {
  return request("/analytics/overview");
}

export async function getTopAspects(limit = 12) {
  return request(`/analytics/top_aspects?limit=${limit}`);
}

export async function getAspectSentimentDistribution(limit = 12) {
  return request(`/analytics/aspect_sentiment_distribution?limit=${limit}`);
}

export async function getTrends(interval = "day") {
  return request(`/analytics/trends?interval=${interval}`);
}

export async function getKgCentrality(limit = 20, domain = "") {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (domain) params.set("domain", domain);
  return request(`/analytics/kg/centrality?${params.toString()}`);
}

export async function rebuildKg(domain = "") {
  const query = domain ? `?domain=${encodeURIComponent(domain)}` : "";
  return request(`/jobs/kg_rebuild${query}`, { method: "POST" });
}

export async function getReviewGraph(reviewId) {
  return request(`/graph/review/${reviewId}`);
}

export async function getBatchAspectGraph(filters = {}) {
  const params = new URLSearchParams();

  if (filters.domain) params.set("domain", filters.domain);
  if (filters.product_id) params.set("product_id", filters.product_id);
  if (filters.from) params.set("from", filters.from);
  if (filters.to) params.set("to", filters.to);
  params.set("min_edge_weight", String(filters.min_edge_weight || 1));

  const query = params.toString();
  return request(`/graph/aspects${query ? `?${query}` : ""}`);
}

export async function getDashboardKpis(filters = {}) {
  const params = new URLSearchParams();
  if (filters.from) params.set("from", filters.from);
  if (filters.to) params.set("to", filters.to);
  if (filters.domain) params.set("domain", filters.domain);
  const query = params.toString();
  return request(`/analytics/dashboard_kpis${query ? `?${query}` : ""}`);
}

export async function getAspectLeaderboard(limit = 25, domain = "") {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (domain) params.set("domain", domain);
  return request(`/analytics/aspect_leaderboard?${params.toString()}`);
}

export async function getAspectTrends(interval = "day", domain = "", limit = 500) {
  const params = new URLSearchParams();
  params.set("interval", interval);
  params.set("limit", String(limit));
  if (domain) params.set("domain", domain);
  return request(`/analytics/aspect_trends?${params.toString()}`);
}

export async function getEmergingAspects(interval = "day", lookbackBuckets = 7, domain = "") {
  const params = new URLSearchParams();
  params.set("interval", interval);
  params.set("lookback_buckets", String(lookbackBuckets));
  if (domain) params.set("domain", domain);
  return request(`/analytics/emerging_aspects?${params.toString()}`);
}

export async function getEvidence(aspect = "", sentiment = "", limit = 50, domain = "") {
  const params = new URLSearchParams();
  if (aspect) params.set("aspect", aspect);
  if (sentiment) params.set("sentiment", sentiment);
  params.set("limit", String(limit));
  if (domain) params.set("domain", domain);
  return request(`/analytics/evidence?${params.toString()}`);
}

export async function getAspectDetail(aspect, interval = "day", domain = "") {
  const params = new URLSearchParams();
  params.set("interval", interval);
  if (domain) params.set("domain", domain);
  return request(`/analytics/aspect_detail/${encodeURIComponent(aspect)}?${params.toString()}`);
}

export async function getAlerts(domain = "") {
  const query = domain ? `?domain=${encodeURIComponent(domain)}` : "";
  return request(`/analytics/alerts${query}`);
}

export async function getImpactMatrix(limit = 20, domain = "") {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (domain) params.set("domain", domain);
  return request(`/analytics/impact_matrix?${params.toString()}`);
}

export async function getSegments(limit = 20, domain = "") {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (domain) params.set("domain", domain);
  return request(`/analytics/segments?${params.toString()}`);
}

export async function getWeeklySummary(domain = "") {
  const query = domain ? `?domain=${encodeURIComponent(domain)}` : "";
  return request(`/analytics/weekly_summary${query}`);
}
