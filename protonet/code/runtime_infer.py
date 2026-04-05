from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, List

import torch
from torch.nn import functional as F

try:
    from .config import ProtonetConfig
    from .encoder import HybridTextEncoder, format_input_text
    from .projection_head import ProjectionHead
except ImportError:
    from config import ProtonetConfig
    from encoder import HybridTextEncoder, format_input_text
    from projection_head import ProjectionHead


CLAUSE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+|[;,]\s+")
WHITESPACE_RE = re.compile(r"\s+")

PATH_FIELDS = {
    field.name
    for field in fields(ProtonetConfig)
    if field.name.endswith("_dir") or field.name == "input_dir" or field.name == "output_dir" or field.name == "metadata_dir"
}


def _normalize_ws(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def _build_config(payload: Dict[str, Any]) -> ProtonetConfig:
    data = dict(payload)
    data.pop("device", None)
    for key in PATH_FIELDS:
        if key in data and isinstance(data[key], str):
            data[key] = Path(data[key])
    return ProtonetConfig(**data)


class ProtonetRuntime:
    def __init__(self, *, cfg: ProtonetConfig, encoder: HybridTextEncoder, projection: ProjectionHead, prototype_bank: Dict[str, Any], temperature: float) -> None:
        self.cfg = cfg
        self.encoder = encoder
        self.projection = projection
        self.labels = list(prototype_bank["labels"])
        self.prototypes = prototype_bank["prototypes"].to(cfg.device)
        self.temperature = max(0.05, float(temperature))

    @classmethod
    def load(cls, bundle_path: str | Path) -> "ProtonetRuntime":
        payload = torch.load(Path(bundle_path), map_location="cpu")
        cfg = _build_config(payload["config"])
        encoder = HybridTextEncoder(cfg)
        projection = ProjectionHead(encoder.hidden_size, int(cfg.projection_dim), float(cfg.dropout))
        projection.load_state_dict(payload["projection_state_dict"])
        if "encoder_state" in payload:
            encoder_state = payload["encoder_state"]
            if "state_dict" in encoder_state and encoder.model is not None:
                encoder.model.load_state_dict(encoder_state["state_dict"])
        encoder.eval()
        projection.eval()
        prototype_bank = payload["prototype_bank"]
        return cls(
            cfg=cfg,
            encoder=encoder,
            projection=projection,
            prototype_bank=prototype_bank,
            temperature=float(payload.get("temperature", 1.0)),
        )

    def _embed(self, review_text: str, evidence_text: str, domain: str | None) -> torch.Tensor:
        text = format_input_text(review_text, evidence_text, domain or "unknown")
        with torch.no_grad():
            encoded = self.encoder([text]).to(self.cfg.device)
            projected = self.projection(encoded)
        return projected

    def score_text(self, review_text: str, evidence_text: str, domain: str | None = None) -> List[Dict[str, Any]]:
        embedding = self._embed(review_text, evidence_text, domain)
        with torch.no_grad():
            logits = -torch.cdist(embedding, self.prototypes, p=2).pow(2) / self.temperature
            probabilities = torch.softmax(logits, dim=-1)[0]
        ranked_indices = torch.argsort(probabilities, descending=True).tolist()
        rows: List[Dict[str, Any]] = []
        for index in ranked_indices:
            label = self.labels[index]
            aspect, _, sentiment = label.partition(self.cfg.joint_label_separator)
            rows.append(
                {
                    "label": label,
                    "aspect": aspect,
                    "sentiment": sentiment or "neutral",
                    "confidence": float(probabilities[index].item()),
                    "raw_score": float(logits[0, index].item()),
                }
            )
        return rows


def split_clauses(review_text: str) -> List[Dict[str, Any]]:
    text = _normalize_ws(review_text)
    if not text:
        return []
    clauses = [part.strip() for part in CLAUSE_SPLIT_RE.split(text) if part and part.strip()]
    if not clauses:
        return [{"snippet": text, "start_char": 0, "end_char": len(text)}]
    spans: List[Dict[str, Any]] = []
    cursor = 0
    lower_text = text.lower()
    for clause in clauses:
        start = lower_text.find(clause.lower(), cursor)
        if start < 0:
            start = lower_text.find(clause.lower())
        if start < 0:
            start = 0
        end = start + len(clause)
        cursor = end
        spans.append({"snippet": clause, "start_char": start, "end_char": end})
    return spans


@lru_cache(maxsize=2)
def load_runtime(bundle_path: str | Path) -> ProtonetRuntime:
    return ProtonetRuntime.load(bundle_path)
