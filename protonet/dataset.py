from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .io_utils import read_json, read_jsonl
from .label_normalizer import LabelNormalizer
from .schema import ReviewExample, example_from_raw


@dataclass
class DatasetBundle:
    artifact_dir: Path
    normalizer: LabelNormalizer
    splits: dict[str, list[ReviewExample]] = field(default_factory=lambda: {"train": [], "val": [], "test": []})
    domain_holdout: dict[str, list[ReviewExample]] = field(default_factory=lambda: {"train": [], "val": [], "test": []})
    grouped: dict[str, list[ReviewExample]] = field(default_factory=lambda: {"train": [], "val": [], "test": []})
    aspect_memory_summary: dict = field(default_factory=dict)
    aspect_memory_candidates: list[dict] = field(default_factory=list)
    aspect_memory_review_queue: list[dict] = field(default_factory=list)
    aspect_memory_promoted: list[dict] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    metrics_summary: dict = field(default_factory=dict)
    label_equivalence: dict[str, list[str]] = field(default_factory=dict)

    @property
    def train(self) -> list[ReviewExample]:
        return self.splits["train"]

    @property
    def val(self) -> list[ReviewExample]:
        return self.splits["val"]

    @property
    def test(self) -> list[ReviewExample]:
        return self.splits["test"]

    def summary(self) -> dict:
        def count_gold(rows: list[ReviewExample]) -> int:
            return sum(len(r.gold_aspects) for r in rows)

        rows = self.train + self.val + self.test
        return {
            "artifact_dir": str(self.artifact_dir),
            "train_rows": len(self.train),
            "val_rows": len(self.val),
            "test_rows": len(self.test),
            "total_rows": len(rows),
            "train_gold": count_gold(self.train),
            "val_gold": count_gold(self.val),
            "test_gold": count_gold(self.test),
            "abstain_rows": sum(1 for r in rows if r.abstain_acceptable),
            "novel_rows": sum(1 for r in rows if r.novelty_status == "novel" or r.gold_novel_labels),
            "boundary_rows": sum(1 for r in rows if r.novelty_status == "boundary" or r.gold_boundary_labels),
            "domain_holdout_rows": sum(len(v) for v in self.domain_holdout.values()),
            "grouped_rows": sum(len(v) for v in self.grouped.values()),
            "memory_candidates": len(self.aspect_memory_candidates),
            "memory_review_queue": len(self.aspect_memory_review_queue),
            "memory_promoted": len(self.aspect_memory_promoted),
        }


def _load_split_dir(base: Path, normalizer: LabelNormalizer) -> dict[str, list[ReviewExample]]:
    out = {"train": [], "val": [], "test": []}
    for split in out:
        path = base / f"{split}.jsonl"
        for idx, row in enumerate(read_jsonl(path)):
            ex = example_from_raw(row, split=split)
            # fallback if row_id is missing or unknown
            if not ex.row_id or ex.row_id == "unknown":
                from dataclasses import replace
                ex = replace(ex, row_id=f"{split}_{idx:06d}")

            # normalize labels once at load time
            norm_gold = []
            for g in ex.gold_aspects:
                norm_gold.append(g.__class__(**{**g.to_dict(), "aspect": normalizer.normalize(g.aspect)}))
            out[split].append(ex.__class__(**{**ex.to_dict(), "gold_aspects": norm_gold}))
    return out


def load_dataset_bundle(artifact_dir: str | Path) -> DatasetBundle:
    artifact_dir = Path(artifact_dir)
    normalizer = LabelNormalizer.from_artifact(artifact_dir)
    bundle = DatasetBundle(artifact_dir=artifact_dir, normalizer=normalizer)
    bundle.splits = _load_split_dir(artifact_dir, normalizer)

    if (artifact_dir / "domain_holdout").exists():
        bundle.domain_holdout = _load_split_dir(artifact_dir / "domain_holdout", normalizer)
    if (artifact_dir / "grouped").exists():
        bundle.grouped = _load_split_dir(artifact_dir / "grouped", normalizer)

    bundle.manifest = read_json(artifact_dir / "manifest.json", default={}) or {}
    bundle.metrics_summary = read_json(artifact_dir / "metrics_summary.json", default={}) or {}
    bundle.aspect_memory_summary = read_json(artifact_dir / "aspect_memory_summary.json", default={}) or {}

    cands = read_json(artifact_dir / "aspect_memory_candidates.json", default=[])
    if isinstance(cands, dict):
        cands = cands.get("items") or cands.get("candidates") or cands.get("top_clusters") or []
    bundle.aspect_memory_candidates = list(cands or [])

    queue = read_json(artifact_dir / "aspect_memory_review_queue.json", default=[])
    if isinstance(queue, dict):
        queue = queue.get("items") or queue.get("review_queue") or queue.get("top_clusters") or []
    bundle.aspect_memory_review_queue = list(queue or [])

    promoted = read_json(artifact_dir / "aspect_memory_promoted.json", default=[])
    if isinstance(promoted, dict):
        promoted = promoted.get("items") or promoted.get("promoted") or promoted.get("top_clusters") or []
    bundle.aspect_memory_promoted = list(promoted or [])

    bundle.label_equivalence = read_json(artifact_dir / "label_equivalence.json", default={}) or {}

    # Annotate unseen labels based on train set visibility
    known_labels = _known_train_labels(bundle.train, normalizer)
    bundle.splits["val"] = _annotate_unseen_labels(bundle.val, known_labels, normalizer)
    bundle.splits["test"] = _annotate_unseen_labels(bundle.test, known_labels, normalizer)

    return bundle


def _known_train_labels(rows: list[ReviewExample], normalizer: LabelNormalizer) -> set[str]:
    labels: set[str] = set()
    for ex in rows:
        for g in ex.gold_aspects:
            if g.aspect and g.aspect != "unknown":
                labels.add(normalizer.normalize(g.aspect))
    return labels


def _annotate_unseen_labels(
    rows: list[ReviewExample],
    known_labels: set[str],
    normalizer: LabelNormalizer,
) -> list[ReviewExample]:
    from dataclasses import replace
    out: list[ReviewExample] = []
    for ex in rows:
        unseen = sorted(
            normalizer.normalize(g.aspect)
            for g in ex.gold_aspects
            if g.aspect and normalizer.normalize(g.aspect) not in known_labels
        )
        out.append(replace(ex, gold_unseen_labels=unseen))
    return out
