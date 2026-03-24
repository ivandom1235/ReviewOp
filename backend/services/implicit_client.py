# proto/backend/services/implicit_client.py
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from core.config import settings


ImplicitPrediction = Dict[str, Any]


class ImplicitClient:
    def __init__(self) -> None:
        self.mode = settings.protonet_mode
        self.base_url = settings.protonet_url.rstrip("/")
        self._predict_fn = None

        if self.mode == "import":
            self._setup_import_mode()

    def _setup_import_mode(self) -> None:
        project_root = Path(__file__).resolve().parents[2]  # proto/
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            module = importlib.import_module("protonet.infer_api")
            self._predict_fn = getattr(module, "predict_implicit_aspects")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import protonet.infer_api.predict_implicit_aspects. "
                "Make sure protonet/metadata/model_bundle.pt exists and protonet inference dependencies are installed."
            ) from exc

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
            results = self._predict_fn(review_text=review_text, domain=domain, top_k=top_k)
        else:
            response = requests.post(
                f"{self.base_url}/infer/implicit",
                json={"text": review_text, "domain": domain, "top_k": top_k},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("predictions", [])

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
                }
            )

        return [x for x in cleaned if x["aspect_raw"]]
