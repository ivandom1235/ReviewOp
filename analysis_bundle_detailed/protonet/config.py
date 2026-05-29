from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ECProtoNetV2Config:
    """Runtime configuration for EC-ProtoNet V2.

    Defaults are conservative. They are not meant to be final paper thresholds;
    use calibration.py on a validation split before reporting results.
    """

    # Artifact governance
    require_artifact_pass: bool = True
    require_active_contract: bool = True
    active_contract_path: str = "CURRENT_ACTIVE_ARTIFACT.json"
    metrics_schema_version: str = "ec_v2_eval_v1"

    # Encoder
    encoder: str = "sentence-transformers"  # hashing | sentence-transformers
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    embedding_dim: int = 384
    hashing_dim: int = 2048
    batch_size: int = 64

    # Prototype construction
    use_evidence_text_for_prototypes: bool = True
    include_aspect_name_in_support_text: bool = True
    min_support_per_aspect: int = 3
    denoise_prototypes: bool = True
    denoise_min_support: int = 4
    denoise_keep_quantile: float = 0.85
    evidence_scope_weighting: bool = True
    source_type_weighting: bool = True
    description_weight: float = 0.25
    train_evidence_weight: float = 0.75

    # Scoring weights
    w_proto: float = 1.00
    w_evidence: float = 0.20
    w_memory: float = 0.15
    w_margin: float = 0.10
    w_scope: float = 0.05
    w_description: float = 0.15
    w_prior: float = 0.05
    aspect_priors: dict[str, float] = field(default_factory=dict)


    # Router thresholds; calibrate on val
    accept_threshold: float = 0.32
    abstain_threshold: float = 0.15
    novel_threshold: float = 0.72
    boundary_margin_threshold: float = 0.035
    evidence_abstain_threshold: float = 0.24
    memory_accept_boost_threshold: float = 0.70
    known_label_evidence_floor: float = 0.45
    implicit_known_label_evidence_floor: float | None = None
    explicit_known_label_evidence_floor: float | None = None
    implicit_accept_threshold: float | None = None
    explicit_accept_threshold: float | None = None
    open_world_evidence_floor: float = 0.35
    known_label_proto_floor: float = 0.25
    open_world_margin_ceiling: float = 0.15
    open_world_known_confidence_ceiling: float = 0.35
    open_world_top1_proto_ceiling: float = 0.30
    open_world_evidence_quality_floor: float = 0.45
    open_world_unknown_residual_weight: float = 0.45
    open_world_unknown_energy_weight: float = 0.35
    open_world_unknown_confidence_weight: float = 0.20
    open_world_unknown_threshold: float = 0.60
    emit_open_world_candidate: bool = True
    class_thresholds: dict[str, float] = field(default_factory=dict)

    # Inventory cleanup
    allow_singleton_equivalence_prototypes: bool = False
    exclude_open_world_mapping_from_known_prototypes: bool = True
    export_prototype_inventory: bool = True

    top_k: int = 5

    # Evaluation
    accepted_decisions: tuple[str, ...] = ("accept_known",)
    open_world_accepted_decisions: tuple[str, ...] = ("accept_known", "open_world_candidate")
    review_accepted_decisions: tuple[str, ...] = ("accept_known", "needs_review", "open_world_candidate")
    positive_open_world_decisions: tuple[str, ...] = ("open_world_candidate",)
    abstain_decisions: tuple[str, ...] = ("abstain",)

    # Memory
    use_memory: bool = True
    memory_min_status: str = "promoted"  # promoted only by default
    memory_source_mode: str = "promoted"  # promoted | review_queue | candidates | oracle_all
    memory_similarity_threshold: float = 0.65

    # Equivalence mapping
    use_equivalence: bool = True

    # Output
    export_predictions: bool = True

    evidence_scope_weights: dict[str, float] = field(
        default_factory=lambda: {
            "exact_phrase": 1.00,
            "token_span": 0.95,
            "phrase_window": 0.85,
            "clause": 0.80,
            "sentence": 0.65,
            "full_review": 0.25,
            "unknown": 0.25,
            "": 0.25,
        }
    )

    source_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "implicit": 1.00,
            "explicit": 0.90,
            "counterfactual": 0.85,
            "synthetic": 0.80,
            "silver": 0.70,
            "unknown": 0.75,
            "": 0.75,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ECProtoNetV2Config":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ECProtoNetV2Config":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
