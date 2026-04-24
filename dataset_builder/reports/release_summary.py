from __future__ import annotations


def build_release_summary(counts: dict[str, int], release_ready: bool) -> dict[str, object]:
    return {"release_ready": release_ready, "export_counts": dict(counts)}
