export default function GraphModeToggle({ value, onChange, isDark = false }) {
  const options = [
    { id: "single_review", label: "Review Explanation" },
    { id: "batch", label: "Corpus Analytics" },
  ];

  return (
    <div className={`inline-flex rounded-2xl p-1 ${isDark ? "bg-[#0f172a]" : "bg-slate-200"}`}>
      {options.map((option) => {
        const active = value === option.id;
        return (
          <button
            key={option.id}
            type="button"
            onClick={() => onChange(option.id)}
            className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
              active
                ? "bg-emerald-500 text-slate-950 shadow-lg"
                : isDark
                  ? "text-slate-300 hover:bg-slate-800"
                  : "text-slate-700 hover:bg-white"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
