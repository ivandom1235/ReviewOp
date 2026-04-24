const TERMINAL_STATUSES = new Set(["done", "completed", "failed"]);

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function pollReviewJobUntilTerminal({
  jobId,
  fetchStatus,
  onStatus = () => {},
  delay = sleep,
  intervalMs = 1500,
  maxAttempts = 80,
}) {
  if (!jobId) {
    throw new Error("jobId is required");
  }
  let lastStatus = null;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const status = await fetchStatus(jobId);
    lastStatus = status;
    onStatus(status.status, status);
    if (TERMINAL_STATUSES.has(String(status.status || "").toLowerCase())) {
      return status;
    }
    await delay(intervalMs);
  }
  return lastStatus || { status: "unknown" };
}
