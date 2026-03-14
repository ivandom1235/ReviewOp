const sentimentClasses = {
  positive: "bg-green-500/30 text-green-100 ring-1 ring-green-400/40",
  neutral: "bg-slate-500/30 text-slate-100 ring-1 ring-slate-400/30",
  negative: "bg-red-500/30 text-red-100 ring-1 ring-red-400/40",
};

function buildSegments(text, nodes) {
  const ranges = (nodes || [])
    .filter((node) => Number.isInteger(node.evidence_start) && Number.isInteger(node.evidence_end))
    .map((node) => ({
      start: node.evidence_start,
      end: node.evidence_end,
      label: node.label,
      sentiment: node.sentiment,
    }))
    .sort((a, b) => a.start - b.start);

  if (!text || !ranges.length) {
    return [{ text, highlighted: false }];
  }

  const segments = [];
  let cursor = 0;

  ranges.forEach((range) => {
    const start = Math.max(range.start, cursor);
    const end = Math.max(start, range.end);
    if (cursor < start) {
      segments.push({ text: text.slice(cursor, start), highlighted: false });
    }
    segments.push({
      text: text.slice(start, end),
      highlighted: true,
      label: range.label,
      sentiment: range.sentiment,
    });
    cursor = end;
  });

  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), highlighted: false });
  }

  return segments.filter((segment) => segment.text);
}

export default function ReviewEvidencePanel({ text = "", nodes = [], isDark = false }) {
  const segments = buildSegments(text, nodes);

  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#08101f]" : "border-slate-200 bg-slate-50"}`}>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-500">Review Evidence</p>
      <div className={`mt-3 whitespace-pre-wrap text-sm leading-7 ${isDark ? "text-slate-200" : "text-slate-700"}`}>
        {segments.map((segment, index) =>
          segment.highlighted ? (
            <mark
              key={`${segment.label}-${index}`}
              className={`mx-[1px] rounded px-1.5 py-0.5 ${sentimentClasses[segment.sentiment] || sentimentClasses.neutral}`}
              title={segment.label}
            >
              {segment.text}
            </mark>
          ) : (
            <span key={`plain-${index}`}>{segment.text}</span>
          )
        )}
      </div>
    </div>
  );
}
