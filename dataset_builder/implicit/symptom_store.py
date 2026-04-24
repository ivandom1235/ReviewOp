from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class SymptomPatternCandidate:
    pattern_id: str
    phrase: str
    aspect_canonical: str
    latent_family: str = "unknown"
    support_count: int = 0
    domains: tuple[str, ...] = ()
    evidence_valid_count: int = 0
    precision_estimate: float = 0.0
    evidence_valid_rate: float = 0.0
    domain_entropy: float = 0.0
    status: str = "candidate"
    domain_scope: str = "unknown"
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "phrase": self.phrase,
            "aspect_canonical": self.aspect_canonical,
            "latent_family": self.latent_family,
            "support_count": self.support_count,
            "domains": list(self.domains),
            "evidence_valid_count": self.evidence_valid_count,
            "precision_estimate": self.precision_estimate,
            "evidence_valid_rate": self.evidence_valid_rate,
            "domain_entropy": self.domain_entropy,
            "status": self.status,
            "domain_scope": self.domain_scope,
            "reason_codes": list(self.reason_codes),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SymptomPatternCandidate":
        return cls(
            pattern_id=str(payload.get("pattern_id") or ""),
            phrase=str(payload.get("phrase") or ""),
            aspect_canonical=str(payload.get("aspect_canonical") or ""),
            latent_family=str(payload.get("latent_family") or "unknown"),
            support_count=int(payload.get("support_count") or 0),
            domains=tuple(str(value) for value in payload.get("domains") or ()),
            evidence_valid_count=int(payload.get("evidence_valid_count") or 0),
            precision_estimate=float(payload.get("precision_estimate") or 0.0),
            evidence_valid_rate=float(payload.get("evidence_valid_rate") or 0.0),
            domain_entropy=float(payload.get("domain_entropy") or 0.0),
            status=str(payload.get("status") or "candidate"),
            domain_scope=str(payload.get("domain_scope") or "unknown"),
            reason_codes=tuple(str(value) for value in payload.get("reason_codes") or ()),
            confidence=float(payload.get("confidence", payload.get("precision_estimate", 0.0)) or 0.0),
        )


@dataclass(frozen=True)
class SymptomMatch:
    pattern_id: str
    matched_pattern: str
    matched_text: str
    start_char: int
    end_char: int
    aspect_canonical: str
    latent_family: str
    confidence: float
    match_type: str


class SymptomPatternStore:
    def __init__(self, patterns: list[SymptomPatternCandidate] | None = None):
        self.patterns = list(patterns or [])
        self._validate()

    def _validate(self) -> None:
        seen: set[str] = set()
        for pattern in self.patterns:
            if not str(pattern.pattern_id or "").strip():
                raise ValueError("symptom pattern requires pattern_id")
            if pattern.pattern_id in seen:
                raise ValueError(f"duplicate pattern_id: {pattern.pattern_id}")
            seen.add(pattern.pattern_id)

    @property
    def promoted(self) -> list[SymptomPatternCandidate]:
        return [pattern for pattern in self.patterns if pattern.status == "promoted"]

    def matching_canonicals(self, text: str, domain: str | None = None) -> list[str]:
        domain_norm = str(domain or "").strip().lower()
        out: list[str] = []
        for pattern in self.promoted:
            if pattern.domain_scope == "domain_scoped" and domain_norm not in {value.lower() for value in pattern.domains}:
                continue
            if _find_phrase_span(pattern.phrase, text) and pattern.aspect_canonical not in out:
                out.append(pattern.aspect_canonical)
        return out

    def match(self, text: str, domain: str | None = None) -> list[SymptomMatch]:
        domain_norm = str(domain or "").strip().lower()
        matches: list[SymptomMatch] = []
        for pattern in self.promoted:
            if pattern.domain_scope == "domain_scoped" and domain_norm not in {value.lower() for value in pattern.domains}:
                continue
            span = _find_phrase_span(pattern.phrase, text)
            if not span:
                continue
            start, end, matched_text, match_type = span
            matches.append(
                SymptomMatch(
                    pattern_id=pattern.pattern_id,
                    matched_pattern=pattern.phrase,
                    matched_text=matched_text,
                    start_char=start,
                    end_char=end,
                    aspect_canonical=pattern.aspect_canonical,
                    latent_family=pattern.latent_family,
                    confidence=max(pattern.confidence, pattern.precision_estimate),
                    match_type=match_type,
                )
            )
        matches.sort(key=lambda item: (-(item.confidence or 0.0), item.start_char, item.end_char))
        deduped: list[SymptomMatch] = []
        occupied: list[tuple[int, int]] = []
        for match in matches:
            if any(not (match.end_char <= start or match.start_char >= end) for start, end in occupied):
                continue
            occupied.append((match.start_char, match.end_char))
            deduped.append(match)
        return deduped

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([pattern.to_dict() for pattern in self.patterns], indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SymptomPatternStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("symptom pattern store must contain a JSON list")
        return cls.load_from_list(payload)

    @classmethod
    def load_from_list(cls, data: list[dict]) -> "SymptomPatternStore":
        return cls([SymptomPatternCandidate.from_dict(item) for item in data])


def _phrase_in_text(phrase: str, text: str) -> bool:
    return _find_phrase_span(phrase, text) is not None


def _find_phrase_span(phrase: str, text: str) -> tuple[int, int, str, str] | None:
    phrase = str(phrase or "").strip()
    text = str(text or "")
    if not phrase or not text:
        return None
    lowered = text.lower()
    phrase_lower = phrase.lower()
    start = lowered.find(phrase_lower)
    if start >= 0:
        end = start + len(phrase)
        return start, end, text[start:end], "exact"

    variants = _simple_variants(phrase_lower)
    for variant in variants:
        match = re.search(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", lowered)
        if match:
            return match.start(), match.end(), text[match.start():match.end()], "lemma"

    tokens = [token for token in re.findall(r"[a-z0-9']+", phrase_lower) if len(token) > 2]
    if not tokens:
        return None
    token_matches = [re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", lowered) for token in tokens]
    token_matches = [match for match in token_matches if match is not None]
    if len(token_matches) >= max(1, len(tokens) - 1):
        start = min(match.start() for match in token_matches)
        end = max(match.end() for match in token_matches)
        
        # Expand to nearest word boundaries within a 24-char limit
        window_start = max(0, start - 24)
        window_end = min(len(text), end + 24)
        
        # Refine boundaries to not cut mid-word
        # Look for the last space before start and first space after end
        expanded_start = text.rfind(" ", window_start, start)
        if expanded_start == -1: expanded_start = window_start
        else: expanded_start += 1 # Skip the space
        
        expanded_end = text.find(" ", end, window_end)
        if expanded_end == -1: expanded_end = window_end
        
        return expanded_start, expanded_end, text[expanded_start:expanded_end].strip(), "phrase_window"
    return None


def _simple_variants(phrase: str) -> set[str]:
    variants = {phrase}
    if phrase.startswith("keeps "):
        rest = phrase.removeprefix("keeps ")
        variants.add(f"keep {rest}")
        variants.add(f"kept {rest}")
    variants.add(phrase.replace(" does not ", " doesn't "))
    variants.add(phrase.replace(" doesn't ", " does not "))
    return variants
