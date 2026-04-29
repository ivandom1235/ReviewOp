from __future__ import annotations

import json
import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
CODE_ROOT = ROOT / "code"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from .runtime_infer import load_runtime, split_clauses
except ImportError:  # pragma: no cover
    from runtime_infer import load_runtime, split_clauses

try:
    from backend.services.seq2seq_infer import Seq2SeqEngine
except Exception:  # pragma: no cover
    Seq2SeqEngine = None


logger = logging.getLogger(__name__)

DEFAULT_BUNDLE_PATH = ROOT / "metadata" / "model_bundle.pt"
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def resolve_bundle_path(bundle_path: str | Path | None = None) -> Path:
    env_path = _env_value("REVIEWOP_PROTONET_BUNDLE_PATH", "PROTONET_BUNDLE_PATH")
    candidate = bundle_path or env_path or DEFAULT_BUNDLE_PATH
    path = Path(candidate)
    if path.is_absolute():
        return path
    repo_relative = PROJECT_ROOT / path
    if repo_relative.exists():
        return repo_relative
    return path


@lru_cache(maxsize=2)
def _load_runtime(bundle_path: str) -> Dict[str, Any]:
    path = Path(bundle_path)
    if not path.exists():
        raise RuntimeError(
            "No protonet runtime bundle could be loaded. "
            f"Expected {path}"
        )
    return {
        "runtime": load_runtime(path),
        "bundle_path": str(path),
        "seq2seq_engine": Seq2SeqEngine.load() if Seq2SeqEngine is not None else None,
    }


def load_service_runtime(bundle_path: str | Path | None = None) -> Dict[str, Any]:
    path = resolve_bundle_path(bundle_path)
    return _load_runtime(str(path))


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
            top_row = dict((selective.get("scored_rows") or [{}])[0] or {})
            aspect = str(top_row.get("aspect") or "").strip()
            abstained.append(
                {
                    "abstain": True,
                    "decision": "abstain",
                    "decision_band": str(selective.get("decision_band") or "boundary"),
                    "reason": "low_selective_confidence",
                    "aspect_raw": aspect,
                    "aspect_cluster": aspect,
                    "sentiment": str(top_row.get("sentiment") or "neutral").strip().lower() or "neutral",
                    "confidence": float(selective.get("confidence", 0.0)),
                    "ambiguity_score": float(selective.get("ambiguity_score", 0.0)),
                    "novelty_score": float(selective.get("novelty_score", 0.0)),
                    "routing": "boundary",
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


def extract_review_text(payload: Dict[str, Any]) -> str:
    for key in ("review_text", "text", "review", "content", "sentence", "comment"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_ws(value)
    raise ValueError("Missing review text in request payload")


def normalize_request_payload(event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        if isinstance(event.get("input"), dict):
            return dict(event["input"])
        if isinstance(event.get("body"), str):
            try:
                body = json.loads(event["body"])
            except json.JSONDecodeError as exc:
                raise ValueError("Request body is not valid JSON") from exc
            if isinstance(body, dict):
                if isinstance(body.get("input"), dict):
                    return dict(body["input"])
                return body
            raise ValueError("Request body must decode to an object")
        return dict(event)
    raise ValueError("Request payload must be a JSON object")


def predict_implicit_aspects(
    review_text: str,
    domain: Optional[str] = None,
    top_k: int = 5,
    bundle_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    text = _normalize_ws(review_text)
    if not text:
        return []

    runtime = load_service_runtime(bundle_path)
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


def infer_from_request(payload: Dict[str, Any], bundle_path: str | Path | None = None) -> Dict[str, Any]:
    review_text = extract_review_text(payload)
    domain = payload.get("domain")
    top_k = int(payload.get("top_k", 5) or 5)
    review_id = payload.get("review_id")
    product_id = payload.get("product_id")
    predictions = predict_implicit_aspects(
        review_text=review_text,
        domain=domain if isinstance(domain, str) else None,
        top_k=top_k,
        bundle_path=bundle_path,
    )
    response: Dict[str, Any] = {"predictions": predictions}
    if review_id is not None:
        response["review_id"] = review_id
    if product_id is not None:
        response["product_id"] = product_id
    if domain is not None:
        response["domain"] = domain
    return response


def handle_event(event: Any, bundle_path: str | Path | None = None) -> Dict[str, Any]:
    payload = normalize_request_payload(event)
    return {"output": infer_from_request(payload, bundle_path=bundle_path)}


def service_status(bundle_path: str | Path | None = None) -> Dict[str, Any]:
    path = resolve_bundle_path(bundle_path)
    return {
        "bundle_path": str(path),
        "bundle_exists": path.exists(),
        "runtime_loaded": path.exists(),
        "seq2seq_engine_available": Seq2SeqEngine is not None,
    }
