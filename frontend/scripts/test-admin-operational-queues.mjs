import assert from "node:assert/strict";

async function testAdminOperationalQueueHelpers() {
  const { getNeedsReview, getNovelCandidates } = await import("../src/api/client.js");
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return {
      ok: true,
      async text() {
        return JSON.stringify([]);
      },
    };
  };
  globalThis.localStorage = { getItem: () => "" };

  await getNeedsReview(25);
  await getNovelCandidates(15);

  assert.equal(calls[0].url, "/analytics/needs_review?limit=25");
  assert.equal(calls[1].url, "/analytics/novel_candidates?limit=15");
}

await testAdminOperationalQueueHelpers();
console.log("admin operational queue tests passed");
