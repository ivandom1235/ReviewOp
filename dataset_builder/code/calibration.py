from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from policy import reliability_bins


def _proxy_correct(aspect: Dict[str, Any]) -> bool:
    confidence = float(aspect.get("confidence", 0.0))
    evidence_quality = float(aspect.get("evidence_quality", 0.0))
    aspect_type = str(aspect.get("aspect_type", aspect.get("type", "explicit"))).lower()
    if aspect_type == "implicit":
        return confidence >= 0.8 and evidence_quality >= 0.7 and not bool(aspect.get("is_sentence_fallback", False))
    return confidence >= 0.7 and evidence_quality >= 0.6 and not bool(aspect.get("is_sentence_fallback", False))


def build_calibration_summary(records: Iterable[Dict[str, Any]], n_bins: int = 10, confidence_key: str = "confidence") -> Dict[str, Any]:
    pairs: List[Tuple[float, bool]] = []
    for row in records:
        aspects = row.get("aspects", []) or row.get("labels", [])
        for aspect in aspects:
            confidence = float(aspect.get(confidence_key, aspect.get("confidence", 0.0)))
            correct = _proxy_correct(aspect)
            pairs.append((confidence, correct))
    bins = reliability_bins(pairs, n_bins=n_bins)
    return {
        "n_bins": n_bins,
        "total_points": len(pairs),
        "bins": bins,
    }


def _bin_for_confidence(confidence: float, bins: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not bins:
        return {"accuracy": confidence, "count": 0, "lower": 0.0, "upper": 1.0}
    value = max(0.0, min(0.999999, float(confidence)))
    for bucket in bins:
        lo = float(bucket.get("lower", 0.0))
        hi = float(bucket.get("upper", 1.0))
        if value >= lo and (value < hi or hi >= 1.0):
            return bucket
    return bins[-1]


@dataclass
class ConfidenceCalibrator:
    n_bins: int = 10
    threshold: float = 0.75
    blend: float = 0.55
    explicit_bins: List[Dict[str, Any]] = field(default_factory=list)
    implicit_bins: List[Dict[str, Any]] = field(default_factory=list)
    overall_bins: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def fit(cls, records: Iterable[Dict[str, Any]], n_bins: int = 10, threshold: float = 0.75, blend: float = 0.55) -> "ConfidenceCalibrator":
        rows = list(records)
        overall = build_calibration_summary(rows, n_bins=n_bins)["bins"]
        explicit_rows: List[Dict[str, Any]] = []
        implicit_rows: List[Dict[str, Any]] = []
        for row in rows:
            aspects = row.get("aspects", []) or row.get("labels", [])
            for aspect in aspects:
                if str(aspect.get("aspect_type", aspect.get("type", "explicit"))).lower() == "implicit":
                    implicit_rows.append(aspect)
                else:
                    explicit_rows.append(aspect)
        explicit_bins = reliability_bins([(float(a.get("confidence", 0.0)), _proxy_correct(a)) for a in explicit_rows], n_bins=n_bins)
        implicit_bins = reliability_bins([(float(a.get("confidence", 0.0)), _proxy_correct(a)) for a in implicit_rows], n_bins=n_bins)
        return cls(
            n_bins=n_bins,
            threshold=threshold,
            blend=blend,
            explicit_bins=explicit_bins,
            implicit_bins=implicit_bins,
            overall_bins=overall,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "threshold": self.threshold,
            "blend": self.blend,
            "explicit_bins": self.explicit_bins,
            "implicit_bins": self.implicit_bins,
            "overall_bins": self.overall_bins,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfidenceCalibrator":
        return cls(
            n_bins=int(data.get("n_bins", 10)),
            threshold=float(data.get("threshold", 0.75)),
            blend=float(data.get("blend", 0.55)),
            explicit_bins=list(data.get("explicit_bins", [])),
            implicit_bins=list(data.get("implicit_bins", [])),
            overall_bins=list(data.get("overall_bins", [])),
        )

    def _lookup_bin(self, confidence: float, aspect_type: str) -> Dict[str, Any]:
        bins = self.implicit_bins if aspect_type == "implicit" and self.implicit_bins else self.explicit_bins if self.explicit_bins else self.overall_bins
        if not bins:
            return {"accuracy": confidence, "count": 0, "lower": 0.0, "upper": 1.0}
        return _bin_for_confidence(confidence, bins)

    def calibrate_confidence(self, confidence: float, *, aspect_type: str = "explicit", evidence_quality: float = 0.0, is_sentence_fallback: bool = False, sentiment_ambiguous: bool = False, sentiment_unresolved: bool = False) -> Tuple[float, bool, Dict[str, Any]]:
        aspect_type = str(aspect_type or "explicit").lower()
        conf = max(0.0, min(0.999999, float(confidence)))
        if aspect_type == "implicit":
            conf *= 0.88
        else:
            conf *= 1.02
        conf *= 0.8 + 0.2 * max(0.0, min(1.0, float(evidence_quality)))
        if is_sentence_fallback:
            conf *= 0.85
        if sentiment_ambiguous or sentiment_unresolved:
            conf *= 0.9
        bucket = self._lookup_bin(conf, aspect_type)
        bucket_acc = float(bucket.get("accuracy", conf))
        calibrated = max(0.0, min(0.99, (1.0 - self.blend) * conf + self.blend * bucket_acc))
        return calibrated, calibrated < self.threshold, bucket

    def apply(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in records:
            row2 = dict(row)
            calibrated_labels = []
            for aspect in row2.get("aspects", []) or row2.get("labels", []):
                aspect2 = dict(aspect)
                raw_conf = float(aspect2.get("confidence", 0.0))
                aspect_type = str(aspect2.get("aspect_type", aspect2.get("type", "explicit"))).lower()
                calibrated, uncertain, bucket = self.calibrate_confidence(
                    raw_conf,
                    aspect_type=aspect_type,
                    evidence_quality=float(aspect2.get("evidence_quality", 0.0)),
                    is_sentence_fallback=bool(aspect2.get("is_sentence_fallback", False)),
                    sentiment_ambiguous=bool(aspect2.get("sentiment_ambiguous", False)),
                    sentiment_unresolved=bool(aspect2.get("sentiment_unresolved", False)),
                )
                aspect2["raw_confidence"] = raw_conf
                aspect2["calibrated_confidence"] = calibrated
                aspect2["confidence"] = calibrated
                aspect2["uncertain"] = uncertain
                aspect2["confidence_bucket"] = {
                    "lower": bucket.get("lower"),
                    "upper": bucket.get("upper"),
                    "accuracy": bucket.get("accuracy"),
                    "count": bucket.get("count"),
                }
                calibrated_labels.append(aspect2)
            if "aspects" in row2:
                row2["aspects"] = calibrated_labels
            else:
                row2["labels"] = calibrated_labels
            out.append(row2)
        return out
