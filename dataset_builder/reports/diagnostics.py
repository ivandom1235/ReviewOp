from __future__ import annotations

from collections import Counter

from ..schemas.reports import DiagnosticsReport


def build_diagnostics(reason_rows: list[object]) -> DiagnosticsReport:
    counter: Counter[str] = Counter()
    for row in reason_rows:
        for reason in getattr(row, "reason_codes", []):
            counter[str(reason)] += 1
    return DiagnosticsReport(gate_failures=dict(counter))
