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

export async function getKgCentrality(limit = 20) {
  return request(`/analytics/kg/centrality?limit=${limit}`);
}
