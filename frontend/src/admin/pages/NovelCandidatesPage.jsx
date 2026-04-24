export default function NovelCandidatesPage({ rows = [], isDark }) {
  return (
    <section className="space-y-3">
      <h1 className="text-2xl font-bold">Novel Candidates</h1>
      <div className={`overflow-hidden rounded-xl border ${isDark ? "border-slate-800 bg-slate-900" : "border-slate-200 bg-white"}`}>
        <table className="w-full text-left text-sm">
          <thead className={isDark ? "bg-slate-800 text-slate-200" : "bg-slate-100 text-slate-700"}>
            <tr>
              <th className="px-3 py-2">Aspect</th>
              <th className="px-3 py-2">Novelty</th>
              <th className="px-3 py-2">Evidence</th>
              <th className="px-3 py-2">Review</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.id}-${row.review_id}`} className={isDark ? "border-t border-slate-800" : "border-t border-slate-200"}>
                <td className="px-3 py-2 font-semibold">{row.aspect}</td>
                <td className="px-3 py-2">{Number(row.novelty_score || 0).toFixed(2)}</td>
                <td className="px-3 py-2">{row.evidence || "No evidence recorded"}</td>
                <td className="px-3 py-2">{row.review_text}</td>
              </tr>
            ))}
            {!rows.length ? (
              <tr>
                <td className="px-3 py-6 text-center text-slate-500" colSpan={4}>No novel candidates.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
