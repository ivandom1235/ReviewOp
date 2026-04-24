import { authHeaders, request } from "./request.js";

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

export async function searchProducts(
  token,
  { q = "", min_rating = 1, sort = "most_recent", limit = 10, offset = 0 } = {}
) {
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  params.set("min_rating", String(min_rating));
  params.set("sort", sort);
  if (limit != null) params.set("limit", String(limit));
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
  if (options.aspect) params.set("aspect", options.aspect);
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

export async function getMyReviewJob(token, jobId) {
  return request(`/user/jobs/${encodeURIComponent(jobId)}`, {
    headers: { ...authHeaders(token) },
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

export async function getReviewById(token, reviewId) {
  return request(`/user/reviews/${reviewId}`, {
    headers: { ...authHeaders(token) },
  });
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
