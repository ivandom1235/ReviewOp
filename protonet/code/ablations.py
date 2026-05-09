from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ECConfig
from .dataset_loader import ECDatasetLoader
from .encoder import ECEncoder
from .evaluator import ECEvaluator
from .export import export_metrics, export_predictions
from .memory_support import MemorySupportModule
from .prototype_store import PrototypeStore
from .scorer import ECProtoNetScorer


def _build_store_with_descriptions_and_memory(
    loader: ECDatasetLoader,
    encoder: ECEncoder,
    artifact_dir: Path,
) -> tuple[PrototypeStore, MemorySupportModule]:
    """
    Build the full prototype store:
      1. Train-evidence prototypes
      2. Generic description prototypes (fills unseen aspects)
      3. AspectMemory prototypes (learned clusters)
    Returns the store and a loaded MemorySupportModule.
    """
    store = PrototypeStore().build_from_examples(loader.splits["train"], encoder)
    store.build_from_descriptions(encoder)

    memory_mod = MemorySupportModule.from_artifact(ECConfig(), artifact_dir, encoder=encoder)
    store.build_from_memory(encoder, memory_mod.memory_summary)

    return store, memory_mod


def run_ablation_study(
    artifact_dir: str | Path,
    output_dir: str | Path,
    export_preds: bool = False,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ECDatasetLoader(artifact_dir).load_all()
    encoder = ECEncoder()

    train_only_store = PrototypeStore().build_from_examples(loader.splits["train"], encoder)
    full_store, memory_mod = _build_store_with_descriptions_and_memory(
        loader, encoder, artifact_dir
    )

    experiments: dict[str, tuple[ECConfig, PrototypeStore, MemorySupportModule | None]] = {
        "plain_protonet": (
            ECConfig(use_evidence=False, use_memory=False, use_novelty=False, use_contradiction=False, use_router=False),
            train_only_store,
            None,
        ),
        "protonet_plus_description_prototypes": (
            ECConfig(use_evidence=False, use_memory=False, use_novelty=False, use_contradiction=False, use_router=False),
            full_store,
            None,
        ),
        "protonet_plus_evidence": (
            ECConfig(use_evidence=True, use_memory=False, use_novelty=False, use_contradiction=False, use_router=False),
            full_store,
            None,
        ),
        "protonet_plus_memory": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=False, use_contradiction=False, use_router=False),
            full_store,
            memory_mod,
        ),
        "protonet_plus_selective": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=False, use_router=True),
            full_store,
            memory_mod,
        ),
        "full_ec_without_contradiction": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=False, use_router=True),
            full_store,
            memory_mod,
        ),
        "full_ec_protonet": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=True, use_router=True),
            full_store,
            memory_mod,
        ),
    }

    all_results: dict[str, Any] = {}
    test_examples = loader.splits["test"]

    for name, (config, store, mem) in experiments.items():
        print(f"Running ablation: {name}...")
        scorer = ECProtoNetScorer(encoder, store, config, memory_mod=mem)
        evaluator = ECEvaluator()

        predictions = [scorer.predict(ex) for ex in test_examples]
        metrics = evaluator.evaluate(test_examples, predictions)
        metrics["experiment"] = name

        if export_preds:
            pred_path = output_dir / f"predictions_{name}.jsonl"
            export_predictions(predictions, test_examples, pred_path)

        export_metrics(metrics, output_dir / f"metrics_{name}.json")
        all_results[name] = metrics

    with open(output_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    proto_summary = full_store.summary()
    with open(output_dir / "prototype_summary.json", "w", encoding="utf-8") as f:
        json.dump(proto_summary, f, indent=2)

    return all_results
