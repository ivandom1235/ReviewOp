from __future__ import annotations
from dataclasses import dataclass
from ..explicit.phrase_cleaning import is_noisy_label

ACTION_START_VERBS = {
    "recommend", "save", "lose", "have", "make", "use", "buy", "try", "go", "come", "get", "take", "give", "watch",
    "upload", "download", "open", "close"
}
PRONOUN_STARTS = {"this", "that", "these", "those", "my", "your", "our", "their", "his", "her", "its"}
CONTEXT_NOUNS = {"occasion", "beginning", "time", "day", "night", "moment", "thing", "stuff", "way", "part", "kind", "type", "lot"}
SENTIMENT_PREFIXES = {"great", "excellent", "amazing", "terrible", "awful", "bad", "good", "poor", "delicious", "tasty", "slow", "fast"}

@dataclass(frozen=True)
class CandidateDecision:
    bucket: str
    score: float
    reasons: tuple[str, ...]

def _looks_like_noise(candidate: str, evidence_text: str = "") -> bool:
    raw = str(candidate or "").strip().lower()
    if not raw or is_noisy_label(raw):
        return True
    parts = raw.split()
    if parts and parts[0] in ACTION_START_VERBS:
        return True
    if parts and parts[0] in PRONOUN_STARTS:
        return True
    if raw in CONTEXT_NOUNS:
        return True
    if raw.replace(".", "", 1).isdigit():
        return True
    if not str(evidence_text or "").strip():
        return True
    return False

def _score_unmapped_candidate(candidate: str, evidence_text: str, support_count: int = 1) -> tuple[float, tuple[str, ...]]:
    score = 0.0
    reasons: list[str] = []
    c = str(candidate or "").strip().lower()
    e = str(evidence_text or "").strip().lower()
    tokens = c.split()
    if tokens and tokens[0] not in ACTION_START_VERBS and tokens[0] not in PRONOUN_STARTS and all(t not in {"is", "was", "were", "are", "be", "being", "been"} for t in tokens) and 1 <= len(tokens) <= 3:
        score += 0.25; reasons.append("noun_like")
    if c and c in e and c != e:
        score += 0.15; reasons.append("evidence_context_match")
    opinion_cues = ("good", "bad", "great", "poor", "slow", "fast", "broken", "broke", "excellent", "awful", "amazing", "terrible", "soft", "hard")
    if any(tok in e for tok in opinion_cues):
        score += 0.25; reasons.append("opinion_linked")
    if not _looks_like_noise(candidate, evidence_text):
        score += 0.15; reasons.append("not_noise")
    if int(support_count) > 1:
        score += 0.1; reasons.append("repeated_support")
    return score, tuple(reasons)

def strip_sentiment_modifiers(candidate: str) -> str:
    tokens = str(candidate or "").lower().split()
    while tokens and tokens[0] in SENTIMENT_PREFIXES:
        tokens = tokens[1:]
    return " ".join(tokens).strip()

def keep_open_world_candidate(candidate: str, confidence: float) -> bool:
    raw = str(candidate or "").strip()
    if not raw:
        return False
    if is_noisy_label(raw):
        return False
    parts = raw.split()
    # Keep for open-world review only when moderately specific.
    return 1 <= len(parts) <= 4 and any(ch.isalpha() for ch in raw) and float(confidence) >= 0.0


def mark_provisional_canonical(candidate: str) -> str:
    raw = str(candidate or "").strip()
    if not raw or is_noisy_label(raw):
        return ""
    return raw.lower().replace(" ", "_")


def classify_unmapped_candidate(
    candidate: str,
    evidence_text: str = "",
    *,
    doc=None,
    support_count: int = 1,
    provisional_policy: str = "strict",
) -> CandidateDecision:
    if _looks_like_noise(candidate, evidence_text):
        return CandidateDecision("dropped_noise", 0.0, ("noise",))
    score, reasons = _score_unmapped_candidate(candidate, evidence_text, support_count=support_count)
    if provisional_policy == "memory_only":
        return CandidateDecision("open_world_candidate", score, reasons)
    if score >= 0.75:
        return CandidateDecision("open_world", score, reasons)
    if score >= 0.50:
        return CandidateDecision("provisional", score, reasons)
    if score >= 0.35:
        return CandidateDecision("open_world_candidate", score, reasons)
    return CandidateDecision("dropped_noise", score, reasons)
