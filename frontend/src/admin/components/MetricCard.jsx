export default function MetricCard({ label, value, tone = "default" }) {
  const toneClass =
    tone === "danger"
      ? "border-danger/40"
      : tone === "success"
      ? "border-success/40"
      : tone === "warning"
      ? "border-warning/40"
      : "border-accent/35";

  return (
    <div className={`rounded-xl border ${toneClass} bg-panel/75 p-4 shadow-glow`}>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-100">{value}</p>
    </div>
  );
}
