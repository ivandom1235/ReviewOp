from __future__ import annotations

from typing import Any


class QualityGateError(ValueError):
    """Raised when a quality gate failure occurs during the pipeline."""

    def __init__(self, gate_results: dict[str, Any], message: str = "Quality gate failed"):
        super().__init__(message)
        self.gate_results = gate_results
