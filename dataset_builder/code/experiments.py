from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from contracts import BuilderConfig
from research_stack import benchmark_registry_payload, build_experiment_plan, model_registry_payload
from utils import stable_id, write_json, utc_now_iso


def run_experiments(base_cfg: BuilderConfig, overrides: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    run_id = stable_id(
        base_cfg.input_dir,
        base_cfg.output_dir,
        base_cfg.model_family,
        base_cfg.benchmark_key or "auto",
        base_cfg.implicit_mode,
        base_cfg.multilingual_mode,
        base_cfg.use_coref,
    )
    plan = build_experiment_plan()
    results: list[dict[str, Any]] = []
    for index, override in enumerate(overrides, start=1):
        cfg_data = asdict(base_cfg)
        cfg_data.update(override)
        result = {
            "experiment_id": f"exp_{index}",
            "generated_at": utc_now_iso(),
            "config": cfg_data,
            "status": "configured",
            "run_id": run_id,
        }
        results.append(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_manifest.json", results)
    write_json(output_dir / "benchmark_registry.json", benchmark_registry_payload())
    write_json(output_dir / "model_registry.json", model_registry_payload())
    write_json(output_dir / "experiment_plan.json", [asdict(item) for item in plan])
    return results
