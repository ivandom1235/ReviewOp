from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from .config import ProtonetConfig
    from .dataset_reader import write_json
except ImportError:
    from config import ProtonetConfig
    from dataset_reader import write_json


def export_model_bundle(
    *,
    cfg: ProtonetConfig,
    model,
    prototype_bank,
    checkpoint_path: Path,
    metrics: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Path:
    bundle_path = cfg.metadata_dir / "model_bundle.pt"
    payload = {
        "bundle_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "encoder": model.encoder.export_info(),
        "encoder_state": model.encoder.export_state(),
        "projection_state_dict": model.projection.state_dict(),
        "temperature": float(model.temperature.detach().cpu().item()),
        "prototype_bank": prototype_bank.to_serializable(),
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "history": history,
    }
    torch.save(payload, bundle_path)
    return bundle_path


def export_report(
    *,
    cfg: ProtonetConfig,
    input_summary: Dict[str, Any],
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    bundle_path: Path,
    checkpoint_path: Path,
    history: List[Dict[str, Any]],
) -> Path:
    report_path = cfg.metadata_dir / "report.json"
    payload = {
        "config": cfg.to_dict(),
        "input_summary": input_summary,
        "runtime": {
            "production_require_transformer": cfg.production_require_transformer,
            "low_confidence_threshold": cfg.low_confidence_threshold,
        },
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "bundle_path": str(bundle_path),
            "history_path": str(cfg.output_dir / "training_history.json"),
            "episode_cache_dir": str(cfg.episode_cache_dir),
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "history": history,
    }
    write_json(report_path, payload)
    return report_path
