from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from .symptom_store import SymptomPatternCandidate


def _canonicalize(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def _neutral_symptom_phrases(text: str) -> list[str]:
    lowered = str(text or "").lower()
    phrases: list[str] = []
    for match in re.finditer(r"\bkeeps?\s+([a-z]+(?:ing|ed|s)?)\b", lowered):
        phrases.append(f"keeps {match.group(1)}")
    for match in re.finditer(r"\bstopped\s+([a-z]+(?:ing|ed)?)\b", lowered):
        phrases.append(f"stopped {match.group(1)}")
    for match in re.finditer(r"\bdoes\s+not\s+([a-z]+)\b", lowered):
        phrases.append(f"does not {match.group(1)}")
    for match in re.finditer(r"\b([a-z]+s?)\s+under\s+([a-z]+)\b", lowered):
        phrases.append(f"{match.group(1)} under {match.group(2)}")
    return phrases


def phrase_in_text(phrase: str, text: str) -> bool:
    phrase = re.sub(r"\s+", " ", str(phrase or "").strip().lower())
    lowered = str(text or "").lower()
    if phrase in lowered:
        return True
    if phrase.startswith("keeps "):
        return ("keep " + phrase.removeprefix("keeps ")) in lowered
    return False


def mine_symptom_patterns(rows: list[dict[str, Any]], *, min_support: int = 2) -> list[SymptomPatternCandidate]:
    support: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for row in rows:
        text = str(row.get("text") or row.get("review_text") or "")
        domain = str(row.get("domain") or "unknown")
        phrases = [str(row["symptom_phrase"]).lower()] if row.get("symptom_phrase") else _neutral_symptom_phrases(text)
        for phrase in phrases:
            normalized = re.sub(r"\s+", " ", phrase.strip().lower())
            if not normalized:
                continue
            evidence_valid = phrase_in_text(normalized, text)
            support[normalized].append((domain, evidence_valid))

    candidates: list[SymptomPatternCandidate] = []
    for phrase, observations in sorted(support.items()):
        if len(observations) < min_support:
            continue
        domains = tuple(sorted({domain for domain, _valid in observations}))
        evidence_valid_count = sum(1 for _domain, valid in observations if valid)
        support_count = len(observations)
        candidates.append(
            SymptomPatternCandidate(
                pattern_id=f"mined_{_canonicalize(phrase)}",
                phrase=phrase,
                aspect_canonical=_canonicalize(phrase),
                support_count=support_count,
                domains=domains,
                evidence_valid_count=evidence_valid_count,
                precision_estimate=evidence_valid_count / support_count,
                evidence_valid_rate=evidence_valid_count / support_count,
                domain_entropy=1.0 if len(domains) > 1 else 0.0,
            )
        )
    return candidates
