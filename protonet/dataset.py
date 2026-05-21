from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

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


def _load_protocol_splits(
    artifact_dir: Path,
    protocol_name: str,
    normalizer: LabelNormalizer,
) -> dict[str, list[ReviewExample]]:
    nested = artifact_dir / protocol_name
    if nested.exists() and nested.is_dir():
        return _load_split_dir(nested, normalizer)

    # Fallback for flattened or older artifact layout:
    out = {"train": [], "val": [], "test": []}
    for split in out:
        path = artifact_dir / f"{protocol_name}_{split}.jsonl"
        if not path.exists():
            continue
        for idx, row in enumerate(read_jsonl(path)):
            ex = example_from_raw(row, split=split)
            if not ex.row_id or ex.row_id == "unknown":
                from dataclasses import replace
                ex = replace(ex, row_id=f"{protocol_name}_{split}_{idx:06d}")

            norm_gold = []
            for g in ex.gold_aspects:
                norm_gold.append(
                    g.__class__(**{**g.to_dict(), "aspect": normalizer.normalize(g.aspect)})
                )
            out[split].append(ex.__class__(**{**ex.to_dict(), "gold_aspects": norm_gold}))
    return out


def load_dataset_bundle(artifact_dir: str | Path) -> DatasetBundle:
    artifact_dir = Path(artifact_dir)
    normalizer = LabelNormalizer.from_artifact(artifact_dir)
    bundle = DatasetBundle(artifact_dir=artifact_dir, normalizer=normalizer)
    bundle.splits = _load_split_dir(artifact_dir, normalizer)

    bundle.domain_holdout = _load_protocol_splits(artifact_dir, "domain_holdout", normalizer)
    bundle.grouped = _load_protocol_splits(artifact_dir, "grouped", normalizer)

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

    _collapse_protocol_tail_labels(bundle, normalizer)
    _annotate_protocol_unseen(bundle, normalizer)

    return bundle


def _fallback_parent_label(label: str) -> str:
    label = str(label or "").strip().lower()
    if not label:
        return "unknown"
    toks = [t for t in label.split("_") if t]
    if any(t in {"service", "staff", "server", "waiter", "waitress"} for t in toks):
        return "service_quality"
    if any(t in {"slow", "speed", "quick", "wait", "delay", "late", "delivery"} for t in toks):
        return "service_speed"
    if any(t in {"food", "dish", "meal", "dessert", "sushi", "pizza", "pasta", "bagels", "chicken"} for t in toks):
        return "food_quality"
    if any(t in {"price", "value", "cost", "worth"} for t in toks):
        return "value"
    if any(t in {"battery", "charge", "power"} for t in toks):
        return "battery_life"
    if any(t in {"display", "screen", "resolution"} for t in toks):
        return "display"
    if any(t in {"keyboard", "keys"} for t in toks):
        return "keyboard"
    if any(t in {"software", "app", "apps", "program", "programs", "windows"} for t in toks):
        return "software"
    if any(t in {"performance", "speed", "lag", "crash", "reliability"} for t in toks):
        return "performance"
    if any(t in {"ambience", "atmosphere", "place", "restaurant", "bar"} for t in toks):
        return "ambience"
    return "quality"


def _collapse_rows(rows: list[ReviewExample], rare_labels: set[str], normalizer: LabelNormalizer) -> list[ReviewExample]:
    from dataclasses import replace

    out: list[ReviewExample] = []
    for ex in rows:
        updated_gold = []
        for g in ex.gold_aspects:
            original = normalizer.normalize(g.aspect)
            target = _fallback_parent_label(original) if original in rare_labels else original
            updated_gold.append(replace(g, aspect=target))
        out.append(replace(ex, gold_aspects=updated_gold))
    return out


def _collapse_protocol_tail_labels(bundle: DatasetBundle, normalizer: LabelNormalizer, min_support: int = 2) -> None:
    def _train_counts(rows: list[ReviewExample]) -> Counter[str]:
        c: Counter[str] = Counter()
        for ex in rows:
            for g in ex.gold_aspects:
                a = normalizer.normalize(g.aspect)
                if a and a != "unknown":
                    c[a] += 1
        return c

    grouped_counts = _train_counts(bundle.splits["train"])
    grouped_rare = {a for a, n in grouped_counts.items() if n < min_support}
    bundle.splits = {k: _collapse_rows(v, grouped_rare, normalizer) for k, v in bundle.splits.items()}

    if any(bundle.domain_holdout.values()):
        domain_counts = _train_counts(bundle.domain_holdout["train"])
        domain_rare = {a for a, n in domain_counts.items() if n < min_support}
        bundle.domain_holdout = {k: _collapse_rows(v, domain_rare, normalizer) for k, v in bundle.domain_holdout.items()}


def _annotate_protocol_unseen(bundle: DatasetBundle, normalizer: LabelNormalizer) -> None:
    # Grouped protocol unseen labels are defined from grouped train inventory.
    grouped_known = _known_train_labels(bundle.splits["train"], normalizer)
    bundle.splits["val"] = _annotate_unseen_labels(bundle.splits["val"], grouped_known, normalizer)
    bundle.splits["test"] = _annotate_unseen_labels(bundle.splits["test"], grouped_known, normalizer)

    # Domain-holdout unseen labels are defined from domain-holdout train inventory.
    if any(bundle.domain_holdout.values()):
        domain_known = _known_train_labels(bundle.domain_holdout["train"], normalizer)
        bundle.domain_holdout["val"] = _annotate_unseen_labels(bundle.domain_holdout["val"], domain_known, normalizer)
        bundle.domain_holdout["test"] = _annotate_unseen_labels(bundle.domain_holdout["test"], domain_known, normalizer)


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
