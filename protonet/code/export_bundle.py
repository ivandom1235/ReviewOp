from __future__ import annotations

import json
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
    novelty_calibration: Dict[str, Any] | None = None,
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
        "novelty_calibration": novelty_calibration or {},
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
    novelty_calibration: Dict[str, Any] | None = None,
) -> Path:
    report_path = cfg.metadata_dir / "report.json"
    previous_report: Dict[str, Any] | None = None
    if report_path.exists():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous_report = loaded
        except Exception:
            previous_report = None

    def _metric_delta(key: str) -> Dict[str, float] | None:
        if not previous_report:
            return None
        previous_metrics = previous_report.get("metrics", {}) if isinstance(previous_report.get("metrics", {}), dict) else {}
        previous_test = previous_metrics.get("test", {}) if isinstance(previous_metrics.get("test", {}), dict) else {}
        current_test = test_metrics if isinstance(test_metrics, dict) else {}
        if key not in previous_test or key not in current_test:
            return None
        try:
            return {
                "previous": float(previous_test.get(key, 0.0)),
                "current": float(current_test.get(key, 0.0)),
                "delta": float(current_test.get(key, 0.0)) - float(previous_test.get(key, 0.0)),
            }
        except Exception:
            return None

    payload = {
        "config": cfg.to_dict(),
        "input_summary": input_summary,
        "runtime": {
            "production_require_transformer": cfg.production_require_transformer,
            "low_confidence_threshold": cfg.low_confidence_threshold,
            "novelty_known_threshold": cfg.novelty_known_threshold,
            "novelty_novel_threshold": cfg.novelty_novel_threshold,
        },
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "bundle_path": str(bundle_path),
            "history_path": str(cfg.output_dir / "training_history.json"),
            "episode_cache_dir": str(cfg.episode_cache_dir),
            "novelty_calibration_path": str(cfg.novelty_calibration_path),
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "comparison": {
            "source_report": str(report_path) if previous_report else None,
            "test_accuracy": _metric_delta("accuracy"),
            "test_macro_f1": _metric_delta("macro_f1"),
            "test_aspect_only_accuracy": _metric_delta("aspect_only_accuracy"),
            "test_coverage": _metric_delta("coverage"),
            "test_low_confidence_rate": _metric_delta("low_confidence_rate"),
        },
        "history": history,
        "novelty_calibration": novelty_calibration or {},
    }
    write_json(report_path, payload)
    return report_path
