import assert from "node:assert/strict";

async function testGetMyReviewJob() {
  const { getMyReviewJob } = await import("../src/api/user.js");
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return {
      ok: true,
      async text() {
        return JSON.stringify({ job_id: "job-1", status: "queued" });
      },
    };
  };
  globalThis.localStorage = { getItem: () => "" };

  const payload = await getMyReviewJob("token-1", "job-1");

  assert.equal(payload.status, "queued");
  assert.equal(calls[0].url, "/user/jobs/job-1");
  assert.equal(calls[0].options.headers.Authorization, "Bearer token-1");
}

async function testPollStopsOnTerminalStatus() {
  const { pollReviewJobUntilTerminal } = await import("../src/pages/user/reviewJobStatus.js");
  const seen = [];
  const statuses = ["queued", "running", "done"];

  const result = await pollReviewJobUntilTerminal({
    jobId: "job-1",
    fetchStatus: async () => ({ status: statuses.shift() }),
    delay: async () => undefined,
    onStatus: (status) => seen.push(status),
  });

  assert.equal(result.status, "done");
  assert.deepEqual(seen, ["queued", "running", "done"]);
}

async function testPollStopsOnFailedStatus() {
  const { pollReviewJobUntilTerminal } = await import("../src/pages/user/reviewJobStatus.js");
  const result = await pollReviewJobUntilTerminal({
    jobId: "job-1",
    fetchStatus: async () => ({ status: "failed", error: "model_unavailable" }),
    delay: async () => undefined,
  });

  assert.equal(result.status, "failed");
  assert.equal(result.error, "model_unavailable");
}

await testGetMyReviewJob();
await testPollStopsOnTerminalStatus();
await testPollStopsOnFailedStatus();
console.log("review job status tests passed");
