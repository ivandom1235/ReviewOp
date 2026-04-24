from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SymptomPatternCandidate:
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SymptomPatternCandidate":
        return cls(
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
        )


class SymptomPatternStore:
    def __init__(self, patterns: list[SymptomPatternCandidate] | None = None):
        self.patterns = list(patterns or [])

    @property
    def promoted(self) -> list[SymptomPatternCandidate]:
        return [pattern for pattern in self.patterns if pattern.status == "promoted"]

    def matching_canonicals(self, text: str, domain: str | None = None) -> list[str]:
        domain_norm = str(domain or "").strip().lower()
        out: list[str] = []
        for pattern in self.promoted:
            if pattern.domain_scope == "domain_scoped" and domain_norm not in {value.lower() for value in pattern.domains}:
                continue
            if _phrase_in_text(pattern.phrase, text) and pattern.aspect_canonical not in out:
                out.append(pattern.aspect_canonical)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([pattern.to_dict() for pattern in self.patterns], indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SymptomPatternStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("symptom pattern store must contain a JSON list")
        return cls([SymptomPatternCandidate.from_dict(item) for item in payload])


def _phrase_in_text(phrase: str, text: str) -> bool:
    phrase = str(phrase or "").strip().lower()
    lowered = str(text or "").lower()
    if phrase in lowered:
        return True
    if phrase.startswith("keeps "):
        return ("keep " + phrase.removeprefix("keeps ")) in lowered
    return False
