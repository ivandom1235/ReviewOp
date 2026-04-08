from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional

import httpx

from core.config import settings

ImplicitPrediction = Dict[str, Any]
logger = logging.getLogger(__name__)

class ImplicitClient:
    def __init__(self) -> None:
        self.mode = settings.protonet_mode
        self._predict_fn = None
        if self.mode == "import":
            self._setup_import_mode()
        elif self.mode == "http":
            self._setup_http_mode()

    def _setup_http_mode(self) -> None:
        """Verify http configuration and prepare local fallback."""
        if not settings.protonet_url:
            raise RuntimeError("protonet_url is not configured for http mode")
        try:
            self._setup_import_mode()
        except Exception:
            self._predict_fn = None

    def _setup_import_mode(self) -> None:
        try:
            module = importlib.import_module("protonet.infer_api")
            self._predict_fn = getattr(module, "predict_implicit_aspects")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import protonet.infer_api.predict_implicit_aspects. "
                "Make sure protonet/metadata/model_bundle.pt exists and protonet inference dependencies are installed."
            ) from exc

    def _predict_http(self, review_text: str, domain: Optional[str], top_k: int) -> List[ImplicitPrediction]:
        payload = {"text": review_text, "domain": domain, "top_k": top_k}
        try:
            with httpx.Client(timeout=settings.protonet_request_timeout_seconds) as client:
                response = client.post(f"{settings.protonet_url}/infer/implicit", json=payload)
                response.raise_for_status()
                data = response.json()
                return list(data.get("predictions", []))
        except Exception as exc:
            if self._predict_fn:
                logger.warning("ProtoNet HTTP request failed, falling back to local: %s", exc)
                return self._predict_fn(
                    review_text=review_text,
                    domain=domain,
                    top_k=top_k,
                    bundle_path=settings.protonet_bundle_path,
                )
            raise RuntimeError(f"ProtoNet HTTP request failed: {exc}") from exc

    def predict(
        self,
        review_text: str,
        domain: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[ImplicitPrediction]:
        if not review_text.strip():
            return []

        top_k = top_k or settings.max_implicit_candidates

        if self.mode == "import":
            results = self._predict_fn(
                review_text=review_text,
                domain=domain,
                top_k=top_k,
                bundle_path=settings.protonet_bundle_path,
            )
        elif self.mode == "http":
            results = self._predict_http(review_text=review_text, domain=domain, top_k=top_k)
        else:
            raise RuntimeError(f"Unsupported protonet_mode: {self.mode}")

        cleaned: List[ImplicitPrediction] = []
        for row in results or []:
            conf = float(row.get("confidence", 0.0))
            if conf < settings.protonet_implicit_min_confidence:
                continue
            cleaned.append(
                {
                    "aspect_raw": str(row.get("aspect_raw") or row.get("aspect") or "").strip(),
                    "aspect_cluster": str(
                        row.get("aspect_cluster")
                        or row.get("canonical_aspect")
                        or row.get("aspect_raw")
                        or row.get("aspect")
                        or ""
                    ).strip(),
                    "sentiment": str(row.get("sentiment") or "neutral").strip().lower(),
                    "confidence": conf,
                    "evidence_spans": row.get("evidence_spans") or [],
                    "rationale": row.get("rationale") or "",
                    "source": "implicit",
                    "decision": row.get("decision") or "single_label",
                    "abstain": bool(row.get("abstain", False)),
                    "ambiguity_score": float(row.get("ambiguity_score", 0.0)),
                    "novelty_score": float(row.get("novelty_score", 0.0)),
                    "routing": row.get("routing") or "known",
                    "decision_band": row.get("decision_band") or "known",
                    "novel_cluster_id": row.get("novel_cluster_id"),
                    "novel_alias": row.get("novel_alias"),
                    "novel_candidates": row.get("novel_candidates") or [],
                    "abstained_predictions": row.get("abstained_predictions") or [],
                }
            )

        return [
            x
            for x in cleaned
            if x["aspect_raw"]
            or bool(x.get("abstain"))
            or str(x.get("decision") or "").lower() == "abstain"
            or bool(x.get("abstained_predictions"))
        ]
