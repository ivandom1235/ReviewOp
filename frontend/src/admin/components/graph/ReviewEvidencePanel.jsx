const sentimentClasses = {
  positive: {
    dark: "bg-green-500/30 text-green-100 ring-1 ring-green-400/40",
    light: "bg-green-200 text-black ring-1 ring-green-300",
  },
  neutral: {
    dark: "bg-slate-500/30 text-slate-100 ring-1 ring-slate-400/30",
    light: "bg-slate-200 text-black ring-1 ring-slate-300",
  },
  negative: {
    dark: "bg-red-500/30 text-red-100 ring-1 ring-red-400/40",
    light: "bg-red-200 text-black ring-1 ring-red-300",
  },
};

function normalizeSentiment(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) return "neutral";
  if (raw === "positive" || raw === "pos" || raw === "+") return "positive";
  if (raw === "negative" || raw === "neg" || raw === "-") return "negative";
  if (raw === "neutral" || raw === "neu" || raw === "mixed") return "neutral";
  return "neutral";
}

function pushSegment(segments, segment) {
  if (!segment?.text) return;
  const prev = segments[segments.length - 1];
  if (
    prev &&
    prev.highlighted === segment.highlighted &&
    (!segment.highlighted || (prev.sentiment === segment.sentiment && prev.label === segment.label))
  ) {
    prev.text += segment.text;
    return;
  }
  segments.push(segment);
}

function isWordChar(ch) {
  return /[A-Za-z0-9]/.test(ch || "");
}

function snapRangeToWordBoundaries(sourceText, range) {
  let start = range.start;
  let end = range.end;
  const max = sourceText.length;

  while (start > 0 && start < max && isWordChar(sourceText[start - 1]) && isWordChar(sourceText[start])) {
    start -= 1;
  }
  while (end > 0 && end < max && isWordChar(sourceText[end - 1]) && isWordChar(sourceText[end])) {
    end += 1;
  }

  return { ...range, start, end };
}

function buildEvidenceItems(text, nodes) {
  const sourceText = typeof text === "string" ? text : "";
  const textLength = sourceText.length;
  const ranges = (nodes || [])
    .filter((node) => Number.isInteger(node.evidence_start) && Number.isInteger(node.evidence_end))
    .map((node) => ({
      start: Math.max(0, Math.min(Number(node.evidence_start), textLength)),
      end: Math.max(0, Math.min(Number(node.evidence_end), textLength)),
      aspect: String(node?.aspect_raw || node?.label || "").trim(),
      label: node.label,
      sentiment: normalizeSentiment(node.sentiment),
    }))
    .filter((range) => range.end > range.start)
    .map((range) => snapRangeToWordBoundaries(sourceText, range))
    .sort((a, b) => a.start - b.start);

  if (!sourceText || !ranges.length) {
    return sourceText ? [{ text: sourceText, highlighted: false }] : [];
  }

  const segments = [];
  let cursor = 0;

  ranges.forEach((range) => {
    let start = Math.max(range.start, cursor);
    while (start < range.end && start > 0 && isWordChar(sourceText[start - 1]) && isWordChar(sourceText[start])) {
      start += 1;
    }
    const end = Math.max(start, Math.min(range.end, textLength));
    if (cursor < start) {
      pushSegment(segments, { text: sourceText.slice(cursor, start), highlighted: false });
    }
    const highlightedText = sourceText.slice(start, end);
    const leadingWhitespace = (highlightedText.match(/^\s+/) || [""])[0];
    const trailingWhitespace = (highlightedText.match(/\s+$/) || [""])[0];
    const coreStart = leadingWhitespace.length;
    const coreEnd = highlightedText.length - trailingWhitespace.length;
    const coreText = highlightedText.slice(coreStart, coreEnd);

    if (leadingWhitespace) {
      pushSegment(segments, { text: leadingWhitespace, highlighted: false });
    }
    if (coreText) {
      pushSegment(segments, {
        text: coreText,
        highlighted: true,
        aspect: range.aspect,
        label: range.label,
        sentiment: range.sentiment,
      });
    }
    if (trailingWhitespace) {
      pushSegment(segments, { text: trailingWhitespace, highlighted: false });
    }
    cursor = end;
  });

  if (cursor < textLength) {
    pushSegment(segments, { text: sourceText.slice(cursor), highlighted: false });
  }

  return segments;
}

export default function ReviewEvidencePanel({ text = "", nodes = [], isDark = false }) {
  const segments = buildEvidenceItems(text, nodes);
  const tone = isDark ? "dark" : "light";

  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#08101f]" : "border-slate-200 bg-slate-50"}`}>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-500">Review Evidence</p>
      <div className={`mt-3 whitespace-pre-wrap text-sm leading-7 ${isDark ? "text-slate-200" : "text-slate-700"}`}>
        {segments.map((segment, index) =>
          segment.highlighted ? (
            <mark
              key={`${segment.label}-${index}`}
              className={`rounded px-1.5 py-0.5 align-baseline ${(sentimentClasses[segment.sentiment] || sentimentClasses.neutral)[tone]}`}
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
