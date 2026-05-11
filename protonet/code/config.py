from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ECConfig:
    # Scoring weights
    w_proto: float = 1.50
    w_evidence: float = 0.40
    w_memory: float = 0.25
    w_verifier: float = 0.20
    w_ambiguity: float = 0.25
    w_novelty: float = 0.15
    w_contradiction: float = 0.20
    
    # Decision thresholds
    accept_threshold: float = 0.35
    abstain_threshold: float = 0.20
    novel_threshold: float = 0.80
    margin_threshold: float = 0.03
    multi_gold_delta: float = 0.06
    memory_sim_threshold: float = 0.65
    
    # Module toggles
    use_evidence: bool = True
    use_memory: bool = True
    use_novelty: bool = True
    use_contradiction: bool = True
    use_router: bool = True
    
    # Evidence scope weights
    evidence_scope_weights: dict[str, float] = field(default_factory=lambda: {
        "exact_phrase": 1.00,
        "clause": 0.85,
        "phrase_window": 0.75,
        "sentence": 0.60,
        "full_review": 0.25,
        "unknown": 0.25,
    })

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ECConfig:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
