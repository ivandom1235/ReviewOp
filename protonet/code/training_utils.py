from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List
import json

import torch

try:
    from .config import ProtonetConfig
    from .model import ProtoNetModel
    from .progress import task_bar
    from .quality_signals import example_quality_weight
    from .prototype_bank import PrototypeBank
except ImportError:
    from config import ProtonetConfig
    from model import ProtoNetModel
    from progress import task_bar
    from quality_signals import example_quality_weight
    from prototype_bank import PrototypeBank


def composite_selection_score(metrics: Dict[str, Any]) -> float:
    return float(0.30 * float(metrics.get("accuracy", 0.0)) + 0.20 * float(metrics.get("macro_f1", 0.0)) + 0.12 * float(metrics.get("aspect_only_accuracy", metrics.get("accuracy", 0.0))) + 0.13 * float(metrics.get("protocol_breakdown", {}).get("grouped", {}).get("accuracy", metrics.get("accuracy", 0.0))) + 0.10 * float(metrics.get("protocol_breakdown", {}).get("grouped", {}).get("macro_f1", metrics.get("macro_f1", 0.0))) + 0.05 * float(metrics.get("known_vs_novel_f1_macro", metrics.get("abstention_f1", 0.0))) + 0.05 * float(metrics.get("abstention_f1", 0.0)) + 0.03 * float(1.0 - metrics.get("calibration_ece", 1.0)) + 0.02 * float(metrics.get("coverage", 0.0)))


def joint_label_from_item(item: Dict[str, Any], separator: str) -> str:
    label = item.get("joint_label")
    if label:
        return str(label)
    aspect = str(item.get("aspect") or item.get("implicit_aspect") or "unknown").strip()
    sentiment = str(item.get("sentiment") or "neutral").strip().lower() or "neutral"
    return f"{aspect}{separator}{sentiment}"


def collect_unique_items(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    items: List[Dict[str, Any]] = []
    for episode in episodes:
        for item in list(episode.get("support_set", [])) + list(episode.get("query_set", [])):
            key = str(item.get("example_id") or item.get("parent_review_id"))
            if key in seen:
                continue
            seen.add(key)
            items.append(dict(item))
    return items


def build_offline_embedding_cache(model: ProtoNetModel, episodes_by_split: Dict[str, List[Dict[str, Any]]], cfg: ProtonetConfig) -> Dict[str, Any]:
    if model.encoder.trainable:
        return {"enabled": False, "reason": "encoder_trainable"}
    all_items: List[Dict[str, Any]] = []
    for split, episodes in episodes_by_split.items():
        all_items.extend(collect_unique_items(episodes))
    if not all_items:
        return {"enabled": False, "reason": "no_items"}
    model.eval()
    batch_size = 128 if cfg.device.type == "cuda" else 48
    with torch.no_grad():
        for idx in range(0, len(all_items), batch_size):
            _ = model.encode_items(all_items[idx : idx + batch_size])
    cache_dir = cfg.output_dir / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {"enabled": True, "items": len(model.precomputed_embeddings), "splits": {split: len(episodes) for split, episodes in episodes_by_split.items()}}
    (cache_dir / "summary.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def clear_embedding_cache_if_trainable(model: ProtoNetModel) -> bool:
    if not bool(getattr(getattr(model, "encoder", None), "trainable", False)):
        return False
    cache = getattr(model, "precomputed_embeddings", None)
    if isinstance(cache, dict):
        cache.clear()
        return True
    return False


def warmup_label_cap(model: ProtoNetModel) -> int:
    if model.encoder.backend == "transformer":
        return 4 if model.cfg.device.type == "cpu" else 8
    return 12


def warmup_batch_size(model: ProtoNetModel) -> int:
    if model.encoder.backend == "transformer":
        return 16 if model.cfg.device.type == "cpu" else 48
    return 128

