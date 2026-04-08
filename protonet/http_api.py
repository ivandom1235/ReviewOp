from __future__ import annotations

import logging
import os
from typing import Any, Dict

from protonet.code.inference_service import infer_from_request, service_status

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("fastapi is required to run the Protonet HTTP service") from exc

    app = FastAPI(title="Protonet Inference Service", version="6.0.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, **service_status()}

    @app.post("/infer/implicit")
    def infer_implicit(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return infer_from_request(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            logger.exception("Protonet inference failed", exc_info=exc)
            if os.environ.get("PROTONET_DEBUG_ERRORS", "false").lower() in {"true", "1", "yes"}:
                raise HTTPException(status_code=500, detail=f"Protonet inference failed: {exc}") from exc
            raise HTTPException(status_code=500, detail="Protonet inference failed") from exc

    return app


app = create_app()
