from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .dataset import DatasetDiagnostics, SentenceDataset, diagnostics_to_dict, load_dataset_bundle
from .encoder import PrototypeEncoder
from .prototype_builder import PrototypeBuilder


def train_prototypes(
    output_dir: str | Path,
    train_csv_path: str | Path | None = None,
    backend_root: str | Path | None = None,
    dataset_family: str = "reviewlevel",
    input_dir: str | Path | None = None,
    data_source: str = "backend_raw",
    batch_size: int = 32,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: Optional[str] = None,
    dedupe_sentences: bool = True,
    shrinkage_alpha: float = 4.0,
    min_examples_per_centroid: int = 6,
    max_centroids_per_label: int = 2,
    low_support_single_centroid_threshold: int = 10,
    low_support_shrinkage_boost: float = 2.0,
    fail_on_degenerate_val: bool = False,
    label_merge_enabled: bool = True,
    label_merge_map: Dict[str, str] | None = None,
    label_merge_config: str | Path | None = None,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics: DatasetDiagnostics | None = None
    if train_csv_path:
        train_set = SentenceDataset.from_csv(train_csv_path, split="train")
    else:
        bundle = load_dataset_bundle(
            dataset_family=dataset_family,
            data_source=data_source,
            input_dir=input_dir,
            root_dir=backend_root,
            label_merge_enabled=label_merge_enabled,
            label_merge_map=label_merge_map,
            label_merge_config=label_merge_config,
        )
        train_set = bundle.splits["train"]
        diagnostics = bundle.diagnostics
        if fail_on_degenerate_val and diagnostics.is_degenerate_validation:
            raise ValueError("Validation split is degenerate; aborting because fail_on_degenerate_val=True")

    encoder = PrototypeEncoder(model_name=model_name, device=device)
    builder = PrototypeBuilder(
        encoder=encoder,
        dedupe_sentences=dedupe_sentences,
        shrinkage_alpha=shrinkage_alpha,
        min_examples_per_centroid=min_examples_per_centroid,
        max_centroids_per_label=max_centroids_per_label,
        low_support_single_centroid_threshold=low_support_single_centroid_threshold,
        low_support_shrinkage_boost=low_support_shrinkage_boost,
    )
    artifacts = builder.build(dataset=train_set, batch_size=batch_size)
    paths = builder.save(artifacts=artifacts, output_dir=out_dir, model_name=model_name)
    encoder_dir = encoder.save(out_dir / "encoder_model")

    summary = {
        "num_train_rows": len(train_set.rows),
        "num_train_aspects": len(train_set.unique_aspects()),
        "aspect_distribution": train_set.summary(),
        "builder_config": artifacts.config,
        "label_counts": artifacts.label_counts,
        "num_centroids": len(artifacts.centroid_labels),
    }
    if diagnostics is not None:
        summary["dataset_diagnostics"] = diagnostics_to_dict(diagnostics)
    (out_dir / "train_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    payload: Dict[str, object] = {k: str(v) for k, v in paths.items()}
    payload["encoder_model_dir"] = str(encoder_dir)
    payload["data_source"] = data_source
    if diagnostics is not None:
        payload["dataset_diagnostics"] = diagnostics_to_dict(diagnostics)
    return payload
