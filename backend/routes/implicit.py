from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/implicit", tags=["implicit"])


@router.get("/health")
def implicit_health() -> dict[str, bool]:
    return {"ok": True}
