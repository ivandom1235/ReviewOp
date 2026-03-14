export default function AlertDetailPage({ alert, isDark, onBack }) {
  if (!alert) return null;

  return (
    <section className="space-y-6">
      <button 
        onClick={onBack}
        className="flex items-center gap-2 text-sm font-semibold text-indigo-500 hover:text-indigo-400"
      >
        ← Back to Alerts
      </button>

      <div className={`rounded-2xl border p-8 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-bold">{alert.aspect}</h2>
          <span className={`px-4 py-2 rounded-xl text-sm font-bold ${
            alert.severity === "Critical" ? "bg-red-500/10 text-red-500 border border-red-500/20" :
            alert.severity === "Warning" ? "bg-amber-500/10 text-amber-500 border border-amber-500/20" :
            "bg-blue-500/10 text-blue-500 border border-blue-500/20"
          }`}>
            {alert.severity}
          </span>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="space-y-4">
            <div>
              <p className="text-xs uppercase tracking-[0.14em] opacity-60 mb-1">Message</p>
              <p className="text-lg leading-relaxed">{alert.message}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.14em] opacity-60 mb-1">Detected At</p>
              <p className="text-sm">{new Date().toLocaleString()} (Mocked current time)</p>
            </div>
          </div>

          <div className="space-y-4">
             <div className={`rounded-xl p-4 ${isDark ? "bg-slate-900" : "bg-slate-50"}`}>
                <p className="text-xs uppercase tracking-[0.14em] opacity-60 mb-2">Contextual Details</p>
                <ul className="text-sm space-y-2">
                    <li>• Aspect: <span className="font-semibold">{alert.aspect}</span></li>
                    <li>• Significance: High</li>
                    <li>• Impact Score: 8.4/10</li>
                </ul>
             </div>
             <p className="text-xs text-slate-500 italic">Historical data and full evidence for this alert can be found in the Aspect Analytics page.</p>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-slate-800/50">
            <h3 className="text-lg font-semibold mb-4">Recommended Next Steps</h3>
            <div className="grid gap-3 sm:grid-cols-2">
                <div className={`p-4 rounded-xl ${isDark ? "bg-emerald-500/5 border border-emerald-500/10" : "bg-emerald-50 border border-emerald-100"}`}>
                    <p className="font-bold text-emerald-500 mb-1">Investigation</p>
                    <p className="text-sm opacity-80">Review recent logs and evidence related to {alert.aspect} in the Evidence Panel.</p>
                </div>
                <div className={`p-4 rounded-xl ${isDark ? "bg-indigo-500/5 border border-indigo-500/10" : "bg-indigo-50 border border-indigo-100"}`}>
                    <p className="font-bold text-indigo-500 mb-1">Action</p>
                    <p className="text-sm opacity-80">Reach out to the product team to address reported performance issues.</p>
                </div>
            </div>
        </div>
      </div>
    </section>
  );
}
