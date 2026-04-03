const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const DEFAULT_TIMEOUT_MS = 15000;

function withTimeout(options = {}) {
  const timeoutMs = Number(options.timeoutMs || DEFAULT_TIMEOUT_MS);
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  return {
    fetchOptions: {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
      signal: controller.signal,
    },
    timeoutId,
  };
}

async function request(path, options = {}) {
  const { fetchOptions, timeoutId } = withTimeout(options);
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, fetchOptions);
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Request timed out. Please retry.");
    }
    throw new Error("Backend is unreachable. Start backend API and check VITE_PROXY_TARGET/VITE_API_BASE_URL.");
  } finally {
    window.clearTimeout(timeoutId);
  }

  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text || null;
  }

  if (!response.ok) {
    const detail = payload?.detail;
    let message = payload?.error || payload;
    if (Array.isArray(detail)) {
      message = detail.map((d) => d?.msg || JSON.stringify(d)).join("; ");
    } else if (typeof detail === "string") {
      message = detail;
    } else if (detail && typeof detail === "object") {
      message = detail.msg || JSON.stringify(detail);
    }
    if (typeof message !== "string") {
      message = `Request failed: ${response.status}`;
    }
    throw new Error(message);
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

export async function clearAlert(alertId) {
  return request(`/analytics/alerts/${alertId}`, {
    method: "DELETE",
  });
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

export async function getUserReviewsSummary(domain = "") {
  const query = domain ? `?domain=${encodeURIComponent(domain)}` : "";
  return request(`/analytics/user_reviews/summary${query}`);
}

export async function getUserReviewsList({
  domain = "",
  product_id = "",
  username = "",
  min_rating = "",
  max_rating = "",
  limit = 50,
  offset = 0,
} = {}) {
  const params = new URLSearchParams();
  if (domain) params.set("domain", domain);
  if (product_id) params.set("product_id", product_id);
  if (username) params.set("username", username);
  if (min_rating !== "") params.set("min_rating", String(min_rating));
  if (max_rating !== "") params.set("max_rating", String(max_rating));
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  return request(`/analytics/user_reviews/list?${params.toString()}`);
}

async function downloadFile(path, filename) {
  const { fetchOptions, timeoutId } = withTimeout();
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, fetchOptions);
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Download timed out. Please retry.");
    }
    throw new Error("Download failed because backend is unreachable.");
  } finally {
    window.clearTimeout(timeoutId);
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Download failed: ${response.status}`);
  }
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function exportAdminJson() {
  await downloadFile("/analytics/export/json", "reviewop-admin-export.json");
}

export async function exportAdminPdf() {
  await downloadFile("/analytics/export/pdf", "reviewop-admin-export.pdf");
}

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function registerUser(username, password) {
  return request("/user/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
}

export async function loginUser(username, password) {
  return request("/user/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
}

export async function getMe(token) {
  return request("/user/auth/me", {
    headers: { ...authHeaders(token) },
  });
}

export async function getProductSuggestions(token) {
  return request("/user/products/suggestions", {
    headers: { ...authHeaders(token) },
  });
}

export async function searchProducts(token, { q = "", min_rating = 1, sort = "most_recent", offset = 0 } = {}) {
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  params.set("min_rating", String(min_rating));
  params.set("sort", sort);
  params.set("offset", String(offset));
  return request(`/user/products/search?${params.toString()}`, {
    headers: { ...authHeaders(token) },
  });
}

export async function getProductDetail(token, productId) {
  return request(`/user/products/${encodeURIComponent(productId)}`, {
    headers: { ...authHeaders(token) },
  });
}

export async function getProductReviews(token, productId, options = {}) {
  const params = new URLSearchParams();
  if (options.page) params.set("page", String(options.page));
  if (options.page_size) params.set("page_size", String(options.page_size));
  if (options.sort) params.set("sort", options.sort);
  if (options.min_rating) params.set("min_rating", String(options.min_rating));
  return request(`/user/products/${encodeURIComponent(productId)}/reviews?${params.toString()}`, {
    headers: { ...authHeaders(token) },
  });
}

export async function submitReview(token, payload) {
  return request("/user/reviews", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(payload),
  });
}

export async function getMyReviews(token) {
  return request("/user/reviews/me", {
    headers: { ...authHeaders(token) },
  });
}

export async function getMyReviewById(token, reviewId) {
  const rows = await getMyReviews(token);
  return (rows || []).find((row) => Number(row.review_id) === Number(reviewId)) || null;
}

export async function updateMyReview(token, reviewId, payload) {
  return request(`/user/reviews/${reviewId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(payload),
  });
}

export async function deleteMyReview(token, reviewId) {
  return request(`/user/reviews/${reviewId}`, {
    method: "DELETE",
    headers: { ...authHeaders(token) },
  });
}
