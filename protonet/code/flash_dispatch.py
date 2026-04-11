from __future__ import annotations

from typing import Any, Dict


def _serialize_namespace(namespace: Any) -> Dict[str, Any]:
    if hasattr(namespace, "__dict__"):
        return dict(namespace.__dict__)
    return dict(namespace)


def run_train_remote(args: Any) -> Dict[str, Any]:
    raise RuntimeError("Remote Protonet execution has been removed. Run train locally.")


def run_eval_remote(args: Any) -> Dict[str, Any]:
    raise RuntimeError("Remote Protonet execution has been removed. Run eval locally.")


def status() -> Dict[str, Any]:
    return {"endpoint_enabled": False, "endpoint_ready": False, "endpoint_id": None}
