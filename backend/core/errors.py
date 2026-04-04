from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AppError(Exception):
    code: str
    message: str
    status_code: int = 500
    details: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class OperationalFailure(AppError):
    def __init__(self, message: str = "Operation failed", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code="operational_failure", message=message, status_code=500, details=details)


class DatabaseFailure(AppError):
    def __init__(self, message: str = "Database unavailable", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code="database_unavailable", message=message, status_code=503, details=details)


class AnalyticsFailure(AppError):
    def __init__(self, message: str = "Analytics operation failed", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code="analytics_failure", message=message, status_code=500, details=details)
