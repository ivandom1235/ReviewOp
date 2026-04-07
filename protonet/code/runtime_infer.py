from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
import hashlib
import json
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
        self.novelty_calibration = self._load_novelty_calibration()

    def _load_novelty_calibration(self, bundled: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if isinstance(bundled, dict) and bundled:
            t_known = float(bundled.get("thresholds", {}).get("T_known", self.cfg.novelty_known_threshold))
            t_novel = float(bundled.get("thresholds", {}).get("T_novel", self.cfg.novelty_novel_threshold))
            if t_known > t_novel:
                t_known, t_novel = t_novel, t_known
            return {
                "scorer": str(bundled.get("scorer", "distance_energy")),
                "T_known": float(max(0.0, min(1.0, t_known))),
                "T_novel": float(max(0.0, min(1.0, t_novel))),
                "version": str(bundled.get("version", "v2")),
                "snapshot_hash": bundled.get("validation_snapshot_hash"),
            }
        path = self.cfg.novelty_calibration_path
        if not isinstance(path, Path):
            return {
                "scorer": "distance_energy",
                "T_known": float(self.cfg.novelty_known_threshold),
                "T_novel": float(self.cfg.novelty_novel_threshold),
                "version": "default",
            }
        if not path.exists():
            return {
                "scorer": "distance_energy",
                "T_known": float(self.cfg.novelty_known_threshold),
                "T_novel": float(self.cfg.novelty_novel_threshold),
                "version": "default",
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "scorer": "distance_energy",
                "T_known": float(self.cfg.novelty_known_threshold),
                "T_novel": float(self.cfg.novelty_novel_threshold),
                "version": "default",
            }
        t_known = float(payload.get("thresholds", {}).get("T_known", self.cfg.novelty_known_threshold))
        t_novel = float(payload.get("thresholds", {}).get("T_novel", self.cfg.novelty_novel_threshold))
        if t_known > t_novel:
            t_known, t_novel = t_novel, t_known
        return {
            "scorer": str(payload.get("scorer", "distance_energy")),
            "T_known": float(max(0.0, min(1.0, t_known))),
            "T_novel": float(max(0.0, min(1.0, t_novel))),
            "version": str(payload.get("version", "v2")),
            "snapshot_hash": payload.get("validation_snapshot_hash"),
        }

    @staticmethod
    def _novel_cluster_id(domain: str, hint: str) -> str:
        basis = f"v2-novel-cluster|{domain.strip().lower()}|{_normalize_ws(hint).lower()}"
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
        return f"novel_{digest}"

    @staticmethod
    def _novel_alias(aspect: str, evidence_text: str) -> str:
        text = _normalize_ws(evidence_text)
        tokens = [tok for tok in text.split(" ") if tok]
        if not tokens:
            return aspect.strip().lower() or "novel_candidate"
        return " ".join(tokens[:4]).lower()

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
        runtime = cls(
            cfg=cfg,
            encoder=encoder,
            projection=projection,
            prototype_bank=prototype_bank,
            temperature=float(payload.get("temperature", 1.0)),
        )
        runtime.novelty_calibration = runtime._load_novelty_calibration(payload.get("novelty_calibration"))
        return runtime

    def _embed(self, review_text: str, evidence_text: str, domain: str | None) -> torch.Tensor:
        text = format_input_text(review_text, evidence_text, domain or "unknown")
        with torch.no_grad():
            encoded = self.encoder([text]).to(self.cfg.device)
            projected = self.projection(encoded)
        return projected

    def score_text(self, review_text: str, evidence_text: str, domain: str | None = None) -> List[Dict[str, Any]]:
        embedding = self._embed(review_text, evidence_text, domain)
        with torch.no_grad():
            dists = torch.cdist(embedding, self.prototypes, p=2)
            dist2 = dists.pow(2)
            logits = -dist2 / self.temperature
            probabilities = torch.softmax(logits, dim=-1)[0]
            min_dist2 = float(dist2.min().item())
            energy = float((-self.temperature * torch.logsumexp(logits, dim=-1))[0].item())
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
                    "distance_sq": float(dist2[0, index].item()),
                    "min_distance_sq": float(min_dist2),
                    "energy": float(energy),
                }
            )
        return rows

    def score_text_selective(self, review_text: str, evidence_text: str, domain: str | None = None) -> Dict[str, Any]:
        rows = self.score_text(review_text=review_text, evidence_text=evidence_text, domain=domain)
        if not rows:
            return {
                "decision": "abstain",
                "abstain": True,
                "confidence": 0.0,
                "ambiguity_score": 1.0,
                "accepted_predictions": [],
                "abstained_predictions": [],
                "novel_candidates": [],
                "scored_rows": [],
            }
        p1 = float(rows[0].get("confidence", 0.0))
        p2 = float(rows[1].get("confidence", 0.0)) if len(rows) > 1 else 0.0
        ambiguity = max(0.0, min(1.0, 1.0 - (p1 - p2)))
        distance_novelty = p1
        if rows:
            distance_sq = float(rows[0].get("min_distance_sq", 0.0))
            distance_novelty = max(0.0, min(1.0, distance_sq / (distance_sq + 1.0)))
        novelty = max(0.0, min(1.0, distance_novelty))
        evidence_quality = 1.0 if evidence_text.strip() else 0.0
        selective_conf = (
            self.cfg.selective_alpha * p1
            + self.cfg.selective_beta * evidence_quality
            - self.cfg.selective_gamma * ambiguity
            - self.cfg.selective_delta * novelty
        )
        accepted: List[Dict[str, Any]] = []
        abstained_predictions: List[Dict[str, Any]] = []
        novel_candidates: List[Dict[str, Any]] = []
        decision = "single_label"
        t_known = float(self.novelty_calibration.get("T_known", self.cfg.novelty_known_threshold))
        t_novel = float(self.novelty_calibration.get("T_novel", self.cfg.novelty_novel_threshold))
        decision_band = "known"
        if novelty >= t_novel:
            decision_band = "novel"
        elif novelty > t_known:
            decision_band = "boundary"

        if decision_band == "boundary":
            decision = "abstain"
        elif decision_band == "novel":
            decision = "novel"
            accepted.append(dict(rows[0]))
        elif selective_conf < float(self.cfg.abstain_threshold):
            decision = "abstain"
        else:
            margin = float(self.cfg.multi_label_margin)
            top = rows[0]
            accepted.append(dict(top))
            for row in rows[1:]:
                if (float(top.get("confidence", 0.0)) - float(row.get("confidence", 0.0))) <= margin:
                    accepted.append(dict(row))
            if len(accepted) > 1:
                decision = "multi_label"
            else:
                decision = "single_label"

        if decision == "abstain":
            abstained_predictions.append(
                {
                    "reason": "boundary" if decision_band == "boundary" else "low_selective_confidence",
                    "confidence": float(max(0.0, min(1.0, selective_conf))),
                    "ambiguity_score": float(ambiguity),
                }
            )

        if decision_band == "novel":
            top_aspect = str(rows[0].get("aspect") or "novel_candidate")
            cluster_hint = evidence_text or review_text
            cluster_id = self._novel_cluster_id(str(domain or "unknown"), cluster_hint)
            alias = self._novel_alias(top_aspect, evidence_text)
            novel_candidates.append(
                {
                    "aspect": top_aspect,
                    "novelty_score": novelty,
                    "confidence": float(rows[0].get("confidence", 0.0)),
                    "novel_cluster_id": cluster_id,
                    "novel_alias": alias,
                    "evidence_text": evidence_text,
                }
            )
            for row in accepted:
                row["routing"] = "novel"
                row["novel_cluster_id"] = cluster_id
                row["novel_alias"] = alias
        else:
            for row in accepted:
                row["routing"] = "known"

        return {
            "decision": decision,
            "abstain": decision == "abstain",
            "decision_band": decision_band,
            "confidence": float(max(0.0, min(1.0, selective_conf))),
            "ambiguity_score": float(ambiguity),
            "novelty_score": float(novelty),
            "novelty_thresholds": {"T_known": t_known, "T_novel": t_novel},
            "accepted_predictions": accepted,
            "abstained_predictions": abstained_predictions,
            "novel_candidates": novel_candidates,
            "scored_rows": rows,
        }


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
