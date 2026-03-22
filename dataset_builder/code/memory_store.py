from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils import ensure_parent, normalize_text, stable_hash, write_jsonl


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryResolveResult:
    canonical_aspect: str
    confidence: float
    source: str
    hit_count: int = 0


class AspectMemoryStore:
    def __init__(self, base_dir: Path, *, load_event_log: bool = True, read_only: bool = False):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = base_dir / "aspect_memory_events.jsonl"
        self.snapshot_path = base_dir / "aspect_memory_terms.jsonl"
        self.promotions_path = base_dir / "aspect_memory_promotions.jsonl"
        self.calibration_path = base_dir / "aspect_memory_calibration.json"
        self.load_event_log = load_event_log
        self.read_only = read_only
        self._terms: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._hits = Counter()
        self._evidence: List[Dict[str, Any]] = []
        self._promotions: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        paths = [self.snapshot_path]
        if self.load_event_log:
            paths.append(self.event_log)
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    kind = event.get("event")
                    if kind == "upsert_term":
                        key = (normalize_text(event.get("term", "")).lower(), normalize_text(event.get("domain", "")).lower())
                        self._terms[key] = dict(event)
                    elif kind == "increment_hit":
                        key = (normalize_text(event.get("term", "")).lower(), normalize_text(event.get("domain", "")).lower())
                        self._hits[key] += int(event.get("delta", 1))
                    elif kind == "evidence":
                        self._evidence.append(dict(event))
                    elif kind == "promotion":
                        self._promotions.append(dict(event))

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("AspectMemoryStore is read-only during evaluation.")

    def _append_event(self, payload: Dict[str, Any]) -> None:
        self._ensure_writable()
        ensure_parent(self.event_log)
        with self.event_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def upsert_term(self, term: str, canonical_aspect: str, domain: str, source: str = "builder", confidence: float = 0.5, status: str = "candidate") -> Dict[str, Any]:
        self._ensure_writable()
        key = (normalize_text(term).lower(), normalize_text(domain).lower())
        prev = self._terms.get(key, {})
        payload = {
            "event": "upsert_term",
            "id": prev.get("id", stable_hash(term, domain, canonical_aspect, source)),
            "term": normalize_text(term),
            "canonical_aspect": normalize_text(canonical_aspect),
            "domain": normalize_text(domain),
            "source": source,
            "confidence": float(confidence),
            "hit_count": int(prev.get("hit_count", 0)),
            "created_at": prev.get("created_at", _now()),
            "updated_at": _now(),
            "last_seen_at": _now(),
            "status": status,
        }
        self._terms[key] = payload
        self._append_event(payload)
        return payload

    def increment_hit(self, term: str, domain: str = "", delta: int = 1) -> int:
        self._ensure_writable()
        key = (normalize_text(term).lower(), normalize_text(domain).lower())
        self._hits[key] += delta
        payload = {"event": "increment_hit", "term": normalize_text(term), "domain": normalize_text(domain), "delta": delta, "created_at": _now()}
        self._append_event(payload)
        return self._hits[key]

    def resolve_term(self, term: str, domain: str = "") -> MemoryResolveResult:
        key = (normalize_text(term).lower(), normalize_text(domain).lower())
        row = self._terms.get(key)
        if row:
            return MemoryResolveResult(
                canonical_aspect=row.get("canonical_aspect", normalize_text(term)),
                confidence=float(row.get("confidence", 0.0)),
                source="memory",
                hit_count=int(self._hits.get(key, row.get("hit_count", 0))),
            )
        generic_key = (normalize_text(term).lower(), "")
        row = self._terms.get(generic_key)
        if row:
            return MemoryResolveResult(row.get("canonical_aspect", normalize_text(term)), float(row.get("confidence", 0.0)) * 0.9, "memory_generic", int(self._hits.get(generic_key, row.get("hit_count", 0))))
        return MemoryResolveResult(normalize_text(term), 0.0, "unresolved", 0)

    def record_evidence(self, term: str, review_id: str, text_span: str, sentiment: str, is_implicit: bool) -> None:
        self._ensure_writable()
        payload = {
            "event": "evidence",
            "id": stable_hash(term, review_id, text_span, sentiment),
            "term": normalize_text(term),
            "review_id": review_id,
            "text_span": text_span,
            "sentiment": sentiment,
            "is_implicit": bool(is_implicit),
            "created_at": _now(),
        }
        self._evidence.append(payload)
        self._append_event(payload)

    def list_candidates_for_promotion(self, min_hits: int, min_confidence: float) -> List[Dict[str, Any]]:
        rows = []
        for (term, domain), row in self._terms.items():
            hits = int(self._hits.get((term, domain), row.get("hit_count", 0)))
            if hits < min_hits or float(row.get("confidence", 0.0)) < min_confidence:
                continue
            rows.append(
                {
                    "term": row.get("term", term),
                    "canonical_aspect": row.get("canonical_aspect", ""),
                    "domain": row.get("domain", domain),
                    "confidence": float(row.get("confidence", 0.0)),
                    "hit_count": hits,
                    "status": row.get("status", "candidate"),
                }
            )
        return sorted(rows, key=lambda r: (r["hit_count"], r["confidence"]), reverse=True)

    def promote_term(self, term: str, new_canonical: str, reason: str, approved_by: str = "system", domain: str = "") -> Dict[str, Any]:
        self._ensure_writable()
        payload = {
            "event": "promotion",
            "id": stable_hash(term, new_canonical, reason, approved_by, domain),
            "term": normalize_text(term),
            "old_label": self.resolve_term(term, domain).canonical_aspect,
            "new_label": normalize_text(new_canonical),
            "reason": reason,
            "approved_by": approved_by,
            "domain": normalize_text(domain),
            "created_at": _now(),
        }
        self._promotions.append(payload)
        self._append_event(payload)
        return payload

    def write_promotions(self) -> None:
        self._ensure_writable()
        write_jsonl(self.promotions_path, self._promotions)

    def write_snapshot(self) -> None:
        self._ensure_writable()
        rows = []
        for (term, domain), row in sorted(self._terms.items()):
            key = (normalize_text(term).lower(), normalize_text(domain).lower())
            rows.append(
                {
                    "event": "upsert_term",
                    "id": row.get("id", stable_hash(term, domain, row.get("canonical_aspect", ""), row.get("source", "builder"))),
                    "term": row.get("term", term),
                    "canonical_aspect": row.get("canonical_aspect", ""),
                    "domain": row.get("domain", domain),
                    "source": row.get("source", "builder"),
                    "confidence": float(row.get("confidence", 0.0)),
                    "hit_count": int(self._hits.get(key, row.get("hit_count", 0))),
                    "created_at": row.get("created_at", _now()),
                    "updated_at": row.get("updated_at", _now()),
                    "last_seen_at": row.get("last_seen_at", _now()),
                    "status": row.get("status", "candidate"),
                }
            )
        write_jsonl(self.snapshot_path, rows)

    def write_calibration(self, bins: List[Dict[str, Any]]) -> None:
        self._ensure_writable()
        ensure_parent(self.calibration_path)
        self.calibration_path.write_text(json.dumps({"bins": bins}, indent=2, ensure_ascii=False), encoding="utf-8")

    def reset(self, *, clear_files: bool = True) -> None:
        self._terms.clear()
        self._hits.clear()
        self._evidence.clear()
        self._promotions.clear()
        if clear_files:
            for path in [self.event_log, self.snapshot_path, self.promotions_path, self.calibration_path]:
                if path.exists():
                    path.unlink()

    def stats(self) -> Dict[str, Any]:
        return {
            "term_count": len(self._terms),
            "evidence_count": len(self._evidence),
            "promotion_count": len(self._promotions),
        }
