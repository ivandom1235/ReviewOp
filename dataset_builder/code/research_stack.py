from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from utils import stable_id, to_jsonable, utc_now_iso


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    name: str
    family: str
    task_types: tuple[str, ...]
    domains: tuple[str, ...] = ()
    languages: tuple[str, ...] = ("en",)
    primary_metric: str = "f1"
    auxiliary: bool = False
    source: str = "internal"
    source_url: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class ModelFamilySpec:
    key: str
    name: str
    kind: str
    supported_tasks: tuple[str, ...]
    prompt_mode: str = "constrained"
    augmentation_mode: str = "none"
    requires_training: bool = False
    supports_multilingual: bool = False
    supports_implicit: bool = True
    supports_span_extraction: bool = True
    source: str = "internal"
    notes: str = ""


@dataclass(frozen=True)
class ExperimentRunSpec:
    run_id: str
    benchmark_key: str
    benchmark_family: str
    model_family_key: str
    model_kind: str
    prompt_mode: str
    augmentation_mode: str
    status: str = "planned"
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def default_benchmark_registry() -> dict[str, BenchmarkSpec]:
    return {
        "semeval_english_core": BenchmarkSpec(
            key="semeval_english_core",
            name="SemEval English Core",
            family="english_core",
            task_types=("explicit_aspect", "polarity", "implicit_aspect"),
            domains=("laptop", "restaurant"),
            primary_metric="macro_f1",
            source="benchmark",
            source_url="https://aclanthology.org/",
            notes="Primary English ABSA baseline for laptop and restaurant domains.",
        ),
        "shoes_acosi": BenchmarkSpec(
            key="shoes_acosi",
            name="Shoes-ACOSI",
            family="implicit_heavy",
            task_types=("implicit_opinion_span", "implicit_aspect", "quad"),
            domains=("shoes",),
            primary_metric="span_f1",
            source="benchmark",
            source_url="https://aclanthology.org/2024.findings-emnlp.907/",
            notes="Implicit-heavy benchmark for opinion span extraction.",
        ),
        "m_absa": BenchmarkSpec(
            key="m_absa",
            name="M-ABSA",
            family="multilingual",
            task_types=("triplet", "quad", "implicit_aspect"),
            domains=("electronics", "restaurant", "hotel", "service", "sports", "travel", "beauty"),
            languages=("ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "nl", "pl", "pt", "ru", "th", "tr", "vi", "zh"),
            primary_metric="macro_f1",
            source="benchmark",
            source_url="https://aclanthology.org/2025.emnlp-main.128/",
            notes="Multilingual parallel ABSA benchmark with 21 languages.",
        ),
        "auxiliary_synthetic": BenchmarkSpec(
            key="auxiliary_synthetic",
            name="Auxiliary Synthetic / Internal",
            family="auxiliary",
            task_types=("explicit_aspect", "implicit_aspect", "sentiment"),
            auxiliary=True,
            source="internal",
            notes="Fallback bucket for synthetic or project-specific datasets.",
        ),
    }


def default_model_registry() -> dict[str, ModelFamilySpec]:
    return {
        "heuristic_latent": ModelFamilySpec(
            key="heuristic_latent",
            name="Heuristic Latent Facet Baseline",
            kind="baseline",
            supported_tasks=("implicit_aspect", "span_grounding"),
            notes="Current clean-room latent-facet pipeline used as a diagnostic baseline.",
        ),
        "zeroshot_latent": ModelFamilySpec(
            key="zeroshot_latent",
            name="Zero-Shot Latent Facet Pipeline",
            kind="baseline",
            supported_tasks=("implicit_aspect", "span_grounding"),
            supports_multilingual=True,
            notes="Research-ready zero-shot lane with multilingual routing and heuristic discovery.",
        ),
        "supervised_ate": ModelFamilySpec(
            key="supervised_ate",
            name="Supervised Aspect Term Extraction",
            kind="token_classification",
            supported_tasks=("implicit_aspect", "span_extraction", "quad"),
            requires_training=True,
            supports_multilingual=True,
            notes="Supervised or pseudo-labelled ATE lane for benchmark comparisons.",
        ),
        "hybrid_reasoner": ModelFamilySpec(
            key="hybrid_reasoner",
            name="Hybrid Implicit Reasoner",
            kind="instruction_tuned",
            supported_tasks=("implicit_aspect", "quad", "span_grounding"),
            requires_training=True,
            prompt_mode="reasoning_chain",
            supports_multilingual=True,
            notes="Hybrid zero-shot plus supervised fallback lane.",
        ),
        "encoder_absa": ModelFamilySpec(
            key="encoder_absa",
            name="Encoder ABSA",
            kind="encoder",
            supported_tasks=("explicit_aspect", "polarity", "triplet"),
            requires_training=True,
            supports_implicit=False,
            source="pyabsa",
            notes="DeBERTa-style encoder baseline for standard ABSA tasks.",
        ),
        "end_to_end_absa": ModelFamilySpec(
            key="end_to_end_absa",
            name="End-to-End ABSA",
            kind="token_classification",
            supported_tasks=("span_extraction", "triplet", "quad"),
            requires_training=True,
            source="pyabsa",
            notes="Token-classification extraction path for joint aspect and sentiment outputs.",
        ),
        "implicit_reasoner": ModelFamilySpec(
            key="implicit_reasoner",
            name="Implicit Reasoner",
            kind="instruction_tuned",
            supported_tasks=("implicit_aspect", "implicit_opinion_span", "quad"),
            requires_training=True,
            prompt_mode="reasoning_chain",
            source="iacos/itscl",
            notes="Instruction-tuned research lane for implicit aspects and opinion reasoning.",
        ),
        "llm_prompted": ModelFamilySpec(
            key="llm_prompted",
            name="LLM Prompted ABSA",
            kind="llm",
            supported_tasks=("implicit_aspect", "triplet", "quad", "quintuple"),
            requires_training=False,
            prompt_mode="constrained",
            source="chatabsa/syn-chain",
            notes="Constrained prompting and reasoning-chain evaluation lane.",
        ),
        "augmentation": ModelFamilySpec(
            key="augmentation",
            name="Augmented ABSA",
            kind="augmentation",
            supported_tasks=("explicit_aspect", "implicit_aspect", "triplet", "quad"),
            augmentation_mode="ds2-absa/laca",
            requires_training=True,
            supports_multilingual=True,
            source="ds2-absa/laca",
            notes="Synthetic and cross-lingual augmentation lane for low-resource experiments.",
        ),
    }


def resolve_benchmark(
    *,
    benchmark_key: str | None = None,
    domains: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    source_files: Iterable[str] | None = None,
) -> BenchmarkSpec:
    registry = default_benchmark_registry()
    if benchmark_key and benchmark_key in registry:
        return registry[benchmark_key]

    canonical_domains = {str(domain).strip().lower() for domain in (domains or []) if str(domain).strip()}
    langs = {str(language).strip().lower() for language in (languages or []) if str(language).strip()}
    sources = " ".join(str(source).lower() for source in (source_files or []))

    if canonical_domains and canonical_domains.issubset({"laptop", "restaurant"}):
        return registry["semeval_english_core"]
    if "shoes" in sources or "acosi" in sources:
        return registry["shoes_acosi"]
    if langs and langs.difference({"en"}):
        return registry["m_absa"]
    return registry["auxiliary_synthetic"]


def resolve_model_family(model_family_key: str | None = None) -> ModelFamilySpec:
    registry = default_model_registry()
    if model_family_key and model_family_key in registry:
        return registry[model_family_key]
    return registry["heuristic_latent"]


def benchmark_registry_payload() -> dict[str, Any]:
    return {key: asdict(spec) for key, spec in default_benchmark_registry().items()}


def model_registry_payload() -> dict[str, Any]:
    return {key: asdict(spec) for key, spec in default_model_registry().items()}


def build_experiment_plan(
    *,
    benchmark_keys: Iterable[str] | None = None,
    model_family_keys: Iterable[str] | None = None,
) -> list[ExperimentRunSpec]:
    benchmark_registry = default_benchmark_registry()
    model_registry = default_model_registry()

    selected_benchmarks = [benchmark_registry[key] for key in (benchmark_keys or benchmark_registry.keys()) if key in benchmark_registry]
    selected_models = [model_registry[key] for key in (model_family_keys or model_registry.keys()) if key in model_registry]

    plan: list[ExperimentRunSpec] = []
    for benchmark in selected_benchmarks:
        for model_family in selected_models:
            if benchmark.family == "implicit_heavy" and model_family.key not in {"heuristic_latent", "zeroshot_latent", "supervised_ate", "hybrid_reasoner", "end_to_end_absa", "implicit_reasoner", "llm_prompted"}:
                continue
            if benchmark.family == "multilingual" and model_family.key not in {"heuristic_latent", "zeroshot_latent", "supervised_ate", "hybrid_reasoner", "encoder_absa", "llm_prompted", "augmentation"}:
                continue
            if benchmark.family == "english_core" and model_family.key not in {"heuristic_latent", "zeroshot_latent", "supervised_ate", "hybrid_reasoner", "encoder_absa", "end_to_end_absa", "implicit_reasoner", "llm_prompted", "augmentation"}:
                continue
            run_id = stable_id(benchmark.key, model_family.key)
            plan.append(
                ExperimentRunSpec(
                    run_id=run_id,
                    benchmark_key=benchmark.key,
                    benchmark_family=benchmark.family,
                    model_family_key=model_family.key,
                    model_kind=model_family.kind,
                    prompt_mode=model_family.prompt_mode,
                    augmentation_mode=model_family.augmentation_mode,
                    metrics={},
                )
            )
    return plan


def build_research_manifest(
    *,
    dataset: dict[str, Any],
    benchmark: BenchmarkSpec,
    model_family: ModelFamilySpec,
    metrics: dict[str, Any] | None = None,
    prompt_mode: str | None = None,
    augmentation_mode: str | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "dataset": to_jsonable(dataset),
        "benchmark": asdict(benchmark),
        "model_family": asdict(model_family),
        "prompt_mode": prompt_mode or model_family.prompt_mode,
        "augmentation_mode": augmentation_mode or model_family.augmentation_mode,
        "metrics": to_jsonable(metrics or {}),
    }
