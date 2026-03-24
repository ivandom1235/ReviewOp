# proto/ProtoBackend/infer_api.py
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
OUTPUTS_DIR = ROOT / "outputs"

# allow importing ProtoBackend/implicit_proto and backend/services/seq2seq_infer.py
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from implicit_proto.inference import ImplicitAspectDetector
from backend.services.seq2seq_infer import Seq2SeqEngine


CLAUSE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+|[;,]\s+")
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def _score_to_confidence(score: float) -> float:
    conf = (float(score) + 1.0) / 2.0
    return max(0.0, min(1.0, conf))


def _resolve_family_paths(family: str) -> Tuple[Path, Path]:
    family_dir = OUTPUTS_DIR / family
    return family_dir / "prototypes.npz", family_dir / "best_config.json"


def _load_best_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _aspect_aliases(aspect: str) -> List[str]:
    base = (aspect or "").strip().lower()
    aliases = {
        "battery": ["battery", "battery life", "backup", "drain", "charge holding"],
        "charging": ["charging", "charge", "full charge", "recharge"],
        "network": ["network", "signal", "call", "calling", "connect", "connection"],
        "connectivity": ["connectivity", "internet", "wifi", "data", "pages load"],
        "display": ["display", "screen", "brightness"],
        "audio": ["audio", "speaker", "sound", "volume"],
        "service_speed": ["service", "waiting", "delay", "late", "slow"],
        "delivery": ["delivery", "parcel", "arrived", "shipment"],
        "packaging": ["box", "packaging", "package", "crushed"],
        "return_refund": ["refund", "return", "money back"],
        "room_comfort": ["room", "noise", "sleep", "walls"],
        "food_quality": ["food", "taste", "meal", "dish"],
        "freshness": ["fresh", "stale", "expiry"],
        "staff_behavior": ["staff", "polite", "rude", "attitude"],
        "performance": ["performance", "lag", "slow", "fast", "hang"],
        "heating": ["heat", "heating", "hot", "burning", "warm"],
    }
    return aliases.get(base, [base])


def _find_best_snippet(review_text: str, aspect: str) -> Dict[str, Any]:
    text = review_text or ""
    clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(text) if c and c.strip()]
    if not clauses:
        return {"start_char": 0, "end_char": len(text), "snippet": text}

    aliases = _aspect_aliases(aspect)
    best_clause = clauses[0]
    best_score = -1

    for clause in clauses:
        clause_lower = clause.lower()
        score = 0
        for alias in aliases:
            if alias and alias in clause_lower:
                score += 4

        # extra generic weak clues
        if any(tok in clause_lower for tok in ["but", "however", "although"]):
            score += 1
        if any(tok in clause_lower for tok in ["slow", "late", "great", "excellent", "poor", "bad", "good"]):
            score += 1

        if score > best_score:
            best_score = score
            best_clause = clause

    start = text.lower().find(best_clause.lower())
    if start < 0:
        start = 0
    end = start + len(best_clause)
    return {"start_char": start, "end_char": end, "snippet": best_clause}


@lru_cache(maxsize=4)
def _load_family_runtime(family: str) -> Dict[str, Any]:
    prototypes_path, best_config_path = _resolve_family_paths(family)

    if not prototypes_path.exists():
        raise FileNotFoundError(f"Missing prototypes file: {prototypes_path}")

    best_cfg = _load_best_config(best_config_path)

    detector = ImplicitAspectDetector.from_artifacts(
        prototypes_path=prototypes_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
    )

    return {
        "family": family,
        "detector": detector,
        "best_config": best_cfg,
        "prototypes_path": str(prototypes_path),
        "best_config_path": str(best_config_path),
    }


@lru_cache(maxsize=1)
def _load_runtime() -> Dict[str, Any]:
    runtimes: List[Dict[str, Any]] = []
    loaded_families: List[str] = []

    for family in ("reviewlevel", "episodic"):
        try:
            rt = _load_family_runtime(family)
            runtimes.append(rt)
            loaded_families.append(family)
        except Exception:
            continue

    if not runtimes:
        raise RuntimeError(
            "No implicit prototype runtime could be loaded. "
            "Expected outputs/reviewlevel/prototypes.npz and/or outputs/episodic/prototypes.npz"
        )

    # load shared seq2seq sentiment engine once
    seq2seq_engine = Seq2SeqEngine.load()

    return {
        "runtimes": runtimes,
        "loaded_families": loaded_families,
        "seq2seq_engine": seq2seq_engine,
    }


def _predict_from_family(
    runtime: Dict[str, Any],
    review_text: str,
    top_k: int,
    sentiment_engine: Seq2SeqEngine,
) -> List[Dict[str, Any]]:
    detector: ImplicitAspectDetector = runtime["detector"]
    best_cfg = runtime.get("best_config", {}) or {}

    family_top_k = int(best_cfg.get("top_k", 1))
    family_threshold = float(best_cfg.get("threshold", 0.35))
    label_thresholds = best_cfg.get("label_thresholds") or {}

    preds = detector.predict_aspects(
        sentence=review_text,
        top_k=max(top_k, family_top_k),
        threshold=family_threshold,
        return_top1_if_empty=False,
        label_thresholds=label_thresholds,
    )

    rows: List[Dict[str, Any]] = []
    for pred in preds:
        aspect = str(pred.aspect).strip()
        raw_score = float(pred.score)

        snippet_obj = _find_best_snippet(review_text, aspect)
        snippet = str(snippet_obj.get("snippet", "")).strip()

        sentiment, sentiment_conf = sentiment_engine.classify_sentiment_with_confidence(
            evidence_text=snippet or review_text,
            aspect=aspect,
        )

        rows.append(
            {
                "aspect_raw": aspect,
                "aspect_cluster": aspect,
                "sentiment": sentiment,
                "confidence": round(_score_to_confidence(raw_score), 6),
                "implicit_confidence": round(_score_to_confidence(raw_score), 6),
                "sentiment_confidence": round(float(sentiment_conf), 6),
                "raw_score": round(raw_score, 6),
                "evidence_spans": [snippet_obj],
                "rationale": f"Implicit aspect inferred by {runtime['family']} prototype model; sentiment assigned by Seq2SeqEngine on evidence snippet.",
                "model_family": runtime["family"],
                "source": "implicit",
            }
        )
    return rows


def _merge_family_predictions(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        key = str(row.get("aspect_cluster") or row.get("aspect_raw") or "").strip().lower()
        if not key:
            continue

        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(row)
            continue

        if float(row.get("raw_score", 0.0)) > float(existing.get("raw_score", 0.0)):
            merged[key] = dict(row)

    ranked = sorted(
        merged.values(),
        key=lambda x: (float(x.get("raw_score", 0.0)), float(x.get("confidence", 0.0))),
        reverse=True,
    )
    return ranked[:top_k]


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

    all_rows: List[Dict[str, Any]] = []
    for family_runtime in runtime["runtimes"]:
        family_rows = _predict_from_family(
            runtime=family_runtime,
            review_text=text,
            top_k=top_k,
            sentiment_engine=sentiment_engine,
        )
        all_rows.extend(family_rows)

    final_rows = _merge_family_predictions(all_rows, top_k=top_k)
    return final_rows
