from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
CODE_ROOT = ROOT / "code"
METADATA_ROOT = ROOT / "metadata"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from .code.runtime_infer import load_runtime, split_clauses
except ImportError:
    from code.runtime_infer import load_runtime, split_clauses

try:
    from backend.services.seq2seq_infer import Seq2SeqEngine
except Exception:  # pragma: no cover
    Seq2SeqEngine = None


WHITESPACE_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


@lru_cache(maxsize=1)
def _load_runtime() -> Dict[str, Any]:
    bundle_path = METADATA_ROOT / "model_bundle.pt"
    if not bundle_path.exists():
        raise RuntimeError(
            "No protonet runtime bundle could be loaded. "
            "Expected protonet/metadata/model_bundle.pt"
        )
    return {
        "runtime": load_runtime(bundle_path),
        "bundle_path": str(bundle_path),
        "seq2seq_engine": Seq2SeqEngine.load() if Seq2SeqEngine is not None else None,
    }


def _predict_with_runtime(
    runtime: Dict[str, Any],
    *,
    review_text: str,
    domain: Optional[str],
    top_k: int,
    sentiment_engine: Seq2SeqEngine,
) -> List[Dict[str, Any]]:
    protonet_runtime = runtime["runtime"]
    best_by_aspect: Dict[str, Dict[str, Any]] = {}
    abstained: List[Dict[str, Any]] = []
    novel_candidates: List[Dict[str, Any]] = []
    clauses = split_clauses(review_text)
    clauses = clauses or [{"snippet": review_text, "start_char": 0, "end_char": len(review_text)}]

    for clause in clauses:
        snippet = str(clause.get("snippet", "")).strip() or review_text
        selective = protonet_runtime.score_text_selective(review_text=review_text, evidence_text=snippet, domain=domain)
        if selective.get("decision") == "abstain":
            abstained.append(
                {
                    "abstain": True,
                    "decision": "abstain",
                    "decision_band": str(selective.get("decision_band") or "boundary"),
                    "reason": "low_selective_confidence",
                    "confidence": float(selective.get("confidence", 0.0)),
                    "ambiguity_score": float(selective.get("ambiguity_score", 0.0)),
                    "novelty_score": float(selective.get("novelty_score", 0.0)),
                    "evidence_spans": [dict(clause)],
                    "source": "implicit",
                }
            )
            continue
        for candidate in list(selective.get("accepted_predictions", []))[: max(top_k * 2, 8)]:
            aspect = str(candidate.get("aspect", "")).strip()
            if not aspect:
                continue
            sentiment = str(candidate.get("sentiment", "neutral")).strip().lower() or "neutral"
            sentiment_conf = float(candidate.get("confidence", 0.0))
            if sentiment == "neutral" and sentiment_engine is not None:
                sentiment, sentiment_conf = sentiment_engine.classify_sentiment_with_confidence(
                    evidence_text=snippet,
                    aspect=aspect,
                )
            row = {
                "aspect_raw": aspect,
                "aspect_cluster": aspect,
                "sentiment": sentiment,
                "confidence": round(float(candidate.get("confidence", 0.0)), 6),
                "implicit_confidence": round(float(candidate.get("confidence", 0.0)), 6),
                "sentiment_confidence": round(float(sentiment_conf), 6),
                "raw_score": round(float(candidate.get("raw_score", 0.0)), 6),
                "evidence_spans": [dict(clause)],
                "rationale": "Implicit aspect inferred by protonet prototype bank; sentiment taken from the joint label or refined with Seq2Seq on the best snippet.",
                "model_family": "protonet",
                "source": "implicit",
                "decision": str(selective.get("decision") or "single_label"),
                "decision_band": str(selective.get("decision_band") or "known"),
                "abstain": False,
                "ambiguity_score": float(selective.get("ambiguity_score", 0.0)),
                "novelty_score": float(selective.get("novelty_score", 0.0)),
                "routing": str(candidate.get("routing") or "known"),
                "novel_cluster_id": str(candidate.get("novel_cluster_id") or "").strip() or None,
                "novel_alias": str(candidate.get("novel_alias") or "").strip() or None,
                "novel_candidates": list(selective.get("novel_candidates", [])),
            }
            existing = best_by_aspect.get(aspect)
            if existing is None or float(row["confidence"]) > float(existing["confidence"]):
                best_by_aspect[aspect] = row
        novel_candidates.extend(list(selective.get("novel_candidates", [])))

    merged = sorted(
        best_by_aspect.values(),
        key=lambda item: (float(item.get("confidence", 0.0)), float(item.get("raw_score", 0.0))),
        reverse=True,
    )
    if abstained and not merged:
        return abstained[:top_k]
    for row in merged:
        row["abstained_predictions"] = abstained
        row["novel_candidates"] = novel_candidates
    return merged


def _merge_predictions(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("aspect_cluster") or row.get("aspect_raw") or "").strip().lower()
        if not key:
            continue
        existing = merged.get(key)
        if existing is None or float(row.get("raw_score", 0.0)) > float(existing.get("raw_score", 0.0)):
            merged[key] = dict(row)
    return sorted(
        merged.values(),
        key=lambda item: (float(item.get("raw_score", 0.0)), float(item.get("confidence", 0.0))),
        reverse=True,
    )[:top_k]


def predict_implicit_aspects(
    review_text: str,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    text = _normalize_ws(review_text)
    if not text:
        return []

    runtime = _load_runtime()
    sentiment_engine: Seq2SeqEngine = runtime["seq2seq_engine"]
    rows = _predict_with_runtime(
        runtime,
        review_text=text,
        domain=domain,
        top_k=top_k,
        sentiment_engine=sentiment_engine,
    )
    if rows and not any(
        str(row.get("aspect_cluster") or row.get("aspect_raw") or "").strip()
        for row in rows
    ):
        return rows[:top_k]
    return _merge_predictions(rows, top_k=top_k)
