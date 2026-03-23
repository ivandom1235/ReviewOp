from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Literal, Mapping, Sequence

DATASET_FAMILIES = ("reviewlevel", "episodic")
DATA_SOURCES = ("backend_raw", "input_dir")
SplitName = Literal["train", "val", "test"]
DataSourceName = Literal["backend_raw", "input_dir"]


@dataclass(frozen=True)
class SentenceRow:
    sentence_id: str
    sentence: str
    aspect: str
    split: str


@dataclass(frozen=True)
class SplitDiagnostics:
    split: str
    num_rows: int
    num_sentences: int
    num_labels: int
    label_counts: Dict[str, int]
    train_label_coverage: float
    unseen_labels_vs_train: List[str]


@dataclass(frozen=True)
class DatasetDiagnostics:
    dataset_family: str
    data_source: str
    split_stats: Dict[str, SplitDiagnostics]
    train_labels: List[str]
    warnings: List[str]
    is_degenerate_validation: bool
    label_merge_enabled: bool
    label_merge_map: Dict[str, str]
    merged_label_counts: Dict[str, int]
    unknown_merge_sources: List[str]
    merge_summary: Dict[str, object]


@dataclass(frozen=True)
class DatasetBundle:
    dataset_family: str
    data_source: str
    split_paths: Dict[str, str]
    splits: Dict[str, "SentenceDataset"]
    diagnostics: DatasetDiagnostics


class SentenceDataset:
    """Normalized dataset with one (sentence, aspect) row per implicit label."""

    def __init__(self, rows: Sequence[SentenceRow]) -> None:
        self.rows: List[SentenceRow] = list(rows)
        if not self.rows:
            raise ValueError("Dataset is empty")

    @classmethod
    def from_csv(cls, path: str | Path, split: str = "unknown") -> "SentenceDataset":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        rows: List[SentenceRow] = []
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, raw in enumerate(reader, start=1):
                sentence = str(raw.get("sentence", "")).strip()
                aspect = str(raw.get("aspect", "")).strip()
                if not sentence or not aspect:
                    continue
                rows.append(
                    SentenceRow(
                        sentence_id=f"{split}_csv_{idx:06d}",
                        sentence=sentence,
                        aspect=aspect,
                        split=split,
                    )
                )
        return cls(rows)

    @classmethod
    def from_backend_jsonl(cls, path: str | Path, split: str) -> "SentenceDataset":
        """Load data from JSON or JSONL file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Try loading as a single JSON array first (to support .json files)
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if content.startswith("[") and (content.endswith("]") or content.strip().endswith("]")):
                data = json.loads(content)
                if isinstance(data, list):
                    return cls._from_list(data, split)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # Fallback to line-by-line JSONL parsing
        rows: List[SentenceRow] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                    rows.extend(cls._parse_obj(obj, split, line_no))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no} in {file_path}") from exc

        return cls(rows)

    @classmethod
    def _from_list(cls, data: List[Dict], split: str) -> "SentenceDataset":
        rows: List[SentenceRow] = []
        for i, obj in enumerate(data, start=1):
            rows.extend(cls._parse_obj(obj, split, i))
        return cls(rows)

    @classmethod
    def _parse_obj(cls, obj: Dict, split: str, line_no: int) -> List[SentenceRow]:
        if not isinstance(obj, dict):
            return []
        
        rows: List[SentenceRow] = []
        # Support various ID keys
        record_id = str(obj.get("id", obj.get("example_id", obj.get("review_id", f"{split}_{line_no:06d}"))))
        
        labels = obj.get("implicit_labels")
        if isinstance(labels, list):
            for label_idx, label in enumerate(labels, start=1):
                aspect = str((label or {}).get("implicit_aspect", "")).strip()
                sentence = str((label or {}).get("evidence_sentence", "")).strip()
                if not sentence or not aspect:
                    continue
                rows.append(
                    SentenceRow(
                        sentence_id=f"{record_id}_{label_idx}",
                        sentence=sentence,
                        aspect=aspect,
                        split=split,
                    )
                )
            return rows

        # Flat format
        flat_aspect = str(obj.get("implicit_aspect", "")).strip()
        flat_sentence = str(obj.get("evidence_sentence", "")).strip()
        if flat_aspect and flat_sentence:
            row_id = str(obj.get("episode_id", obj.get("id", obj.get("example_id", f"{split}_{line_no:06d}"))))
            rows.append(
                SentenceRow(
                    sentence_id=row_id,
                    sentence=flat_sentence,
                    aspect=flat_aspect,
                    split=split,
                )
            )
        return rows

    def to_csv(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sentence", "aspect"])
            writer.writeheader()
            for row in self.rows:
                writer.writerow({"sentence": row.sentence, "aspect": row.aspect})
        return out

    def group_by_aspect(self, dedupe_sentences: bool = False) -> Dict[str, List[str]]:
        grouped: DefaultDict[str, List[str]] = defaultdict(list)
        for row in self.rows:
            grouped[row.aspect].append(row.sentence)
        if not dedupe_sentences:
            return dict(grouped)
        deduped: Dict[str, List[str]] = {}
        for aspect, sentences in grouped.items():
            deduped[aspect] = list(dict.fromkeys(sentences))
        return deduped

    def unique_aspects(self) -> List[str]:
        return sorted({row.aspect for row in self.rows})

    def unique_sentences_with_labels(self) -> Dict[str, List[str]]:
        by_sentence: DefaultDict[str, set[str]] = defaultdict(set)
        for row in self.rows:
            by_sentence[row.sentence].add(row.aspect)
        return {sentence: sorted(labels) for sentence, labels in by_sentence.items()}

    def summary(self) -> Dict[str, int]:
        aspect_counts: Dict[str, int] = defaultdict(int)
        for row in self.rows:
            aspect_counts[row.aspect] += 1
        return dict(sorted(aspect_counts.items(), key=lambda x: x[0]))

    def remap_aspects(self, merge_map: Mapping[str, str] | None = None) -> tuple["SentenceDataset", Dict[str, int]]:
        mapping = {str(k).strip(): str(v).strip() for k, v in (merge_map or {}).items() if str(k).strip() and str(v).strip()}
        if not mapping:
            return SentenceDataset(self.rows), {}
        remapped: List[SentenceRow] = []
        merge_counts: Dict[str, int] = Counter()
        for row in self.rows:
            target = mapping.get(row.aspect, row.aspect)
            if target != row.aspect:
                merge_counts[f"{row.aspect}->{target}"] += 1
            remapped.append(
                SentenceRow(
                    sentence_id=row.sentence_id,
                    sentence=row.sentence,
                    aspect=target,
                    split=row.split,
                )
            )
        return SentenceDataset(remapped), dict(sorted(merge_counts.items()))


def load_label_merge_map(
    enabled: bool = True,
    merge_config_path: str | Path | None = None,
) -> Dict[str, str]:
    if not enabled:
        return {}
    if merge_config_path:
        path = Path(merge_config_path)
    else:
        path = Path(__file__).resolve().parents[1] / "configs" / "label_merge_map.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in payload.items() if str(k).strip() and str(v).strip()}


def normalize_dataset_family(dataset_family: str) -> str:
    normalized = str(dataset_family).strip().lower()
    if normalized not in DATASET_FAMILIES:
        raise ValueError(f"Unsupported dataset family '{dataset_family}'. Expected one of: {', '.join(DATASET_FAMILIES)}")
    return normalized


def normalize_data_source(data_source: str | None) -> str:
    normalized = str(data_source or "backend_raw").strip().lower()
    if normalized not in DATA_SOURCES:
        raise ValueError(f"Unsupported data source '{data_source}'. Expected one of: {', '.join(DATA_SOURCES)}")
    return normalized


def _repo_root(root_dir: str | Path | None = None) -> Path:
    return Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]


def resolve_split_path(
    split: SplitName,
    dataset_family: str,
    data_source: str = "backend_raw",
    input_dir: str | Path | None = None,
    root_dir: str | Path | None = None,
) -> Path:
    family = normalize_dataset_family(dataset_family)
    source = normalize_data_source(data_source)
    repo_root = _repo_root(root_dir)

    if source == "backend_raw":
        prefix = "implicit_reviewlevel" if family == "reviewlevel" else "implicit_episode"
        return repo_root / "backend" / "data" / "implicit" / "raw" / f"{prefix}_{split}.jsonl"

    base_input_dir = Path(input_dir) if input_dir else (repo_root / "ProtoBackend" / "input")
    family_dir = base_input_dir / family
    candidate_names = [
        f"{split}.jsonl",
        f"{split}.json",
        f"implicit_{family}_{split}.jsonl",
        f"implicit_{family}_{split}.json",
        f"implicit_reviewlevel_{split}.jsonl" if family == "reviewlevel" else f"implicit_episode_{split}.jsonl",
        f"implicit_reviewlevel_{split}.json" if family == "reviewlevel" else f"implicit_episode_{split}.json",
    ]
    for name in candidate_names:
        candidate = family_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing split file for dataset_family={family} split={split} in {family_dir}. "
        f"Tried: {', '.join(candidate_names)}"
    )


def load_split(
    split: SplitName,
    dataset_family: str,
    data_source: str = "backend_raw",
    input_dir: str | Path | None = None,
    root_dir: str | Path | None = None,
) -> SentenceDataset:
    file_path = resolve_split_path(
        split=split,
        dataset_family=dataset_family,
        data_source=data_source,
        input_dir=input_dir,
        root_dir=root_dir,
    )
    return SentenceDataset.from_backend_jsonl(path=file_path, split=split)


def build_dataset_diagnostics(
    dataset_family: str,
    data_source: str,
    splits: Dict[str, SentenceDataset],
    label_merge_enabled: bool = True,
    label_merge_map: Mapping[str, str] | None = None,
    merged_label_counts: Mapping[str, int] | None = None,
    unknown_merge_sources: Sequence[str] | None = None,
    min_val_sentences: int = 25,
    min_train_labels: int = 2,
) -> DatasetDiagnostics:
    train_labels = splits["train"].unique_aspects()
    train_label_set = set(train_labels)
    split_stats: Dict[str, SplitDiagnostics] = {}
    warnings: List[str] = []

    for split_name, dataset in splits.items():
        label_counts = dataset.summary()
        split_labels = sorted(label_counts)
        unseen = sorted(set(split_labels) - train_label_set) if split_name != "train" else []
        coverage = 0.0
        if split_labels:
            coverage = (len(set(split_labels) & train_label_set) / len(split_labels)) * 100.0
        split_stats[split_name] = SplitDiagnostics(
            split=split_name,
            num_rows=len(dataset.rows),
            num_sentences=len(dataset.unique_sentences_with_labels()),
            num_labels=len(split_labels),
            label_counts=label_counts,
            train_label_coverage=round(coverage, 4),
            unseen_labels_vs_train=unseen,
        )
        if split_name != "train" and unseen:
            warnings.append(f"{split_name} contains labels unseen in train: {', '.join(unseen)}")

    if split_stats["train"].num_labels < min_train_labels:
        warnings.append(
            f"train has only {split_stats['train'].num_labels} labels; prototype calibration will be unstable"
        )
    if split_stats["val"].num_sentences < min_val_sentences:
        warnings.append(
            f"val has only {split_stats['val'].num_sentences} unique sentences; validation metrics may be noisy"
        )
    if split_stats["val"].num_labels and split_stats["val"].train_label_coverage < 50.0:
        warnings.append(
            f"val train-label coverage is only {split_stats['val'].train_label_coverage:.2f}%"
        )

    is_degenerate_validation = (
        split_stats["val"].num_sentences < min_val_sentences
        or split_stats["val"].num_labels == 0
        or (split_stats["val"].num_labels > 0 and split_stats["val"].train_label_coverage == 0.0)
    )
    return DatasetDiagnostics(
        dataset_family=normalize_dataset_family(dataset_family),
        data_source=normalize_data_source(data_source),
        split_stats=split_stats,
        train_labels=train_labels,
        warnings=warnings,
        is_degenerate_validation=is_degenerate_validation,
        label_merge_enabled=bool(label_merge_enabled),
        label_merge_map=dict(sorted((label_merge_map or {}).items())),
        merged_label_counts=dict(sorted((merged_label_counts or {}).items())),
        unknown_merge_sources=sorted(set(str(x) for x in (unknown_merge_sources or []))),
        merge_summary={
            "pre_merge_num_train_labels": len(set((label_merge_map or {}).keys()) | set(train_labels)),
            "post_merge_num_train_labels": len(train_labels),
            "num_merged_rows": int(sum((merged_label_counts or {}).values())),
        },
    )


def load_dataset_bundle(
    dataset_family: str,
    data_source: str = "backend_raw",
    input_dir: str | Path | None = None,
    root_dir: str | Path | None = None,
    label_merge_enabled: bool = True,
    label_merge_map: Mapping[str, str] | None = None,
    label_merge_config: str | Path | None = None,
) -> DatasetBundle:
    merge_map = (
        {str(k).strip(): str(v).strip() for k, v in label_merge_map.items()}
        if label_merge_map is not None
        else load_label_merge_map(enabled=label_merge_enabled, merge_config_path=label_merge_config)
    )
    splits = {
        split: load_split(
            split=split,  # type: ignore[arg-type]
            dataset_family=dataset_family,
            data_source=data_source,
            input_dir=input_dir,
            root_dir=root_dir,
        )
        for split in ("train", "val", "test")
    }
    merged_counts_total: Dict[str, int] = Counter()
    if label_merge_enabled and merge_map:
        for split in ("train", "val", "test"):
            remapped, merge_counts = splits[split].remap_aspects(merge_map)
            splits[split] = remapped
            merged_counts_total.update(merge_counts)
    observed_labels = {label for ds in splits.values() for label in ds.unique_aspects()}
    unknown_sources = sorted([src for src in merge_map.keys() if src not in observed_labels and src not in merge_map.values()])
    split_paths = {
        split: str(
            resolve_split_path(
                split=split,  # type: ignore[arg-type]
                dataset_family=dataset_family,
                data_source=data_source,
                input_dir=input_dir,
                root_dir=root_dir,
            )
        )
        for split in ("train", "val", "test")
    }
    diagnostics = build_dataset_diagnostics(
        dataset_family=dataset_family,
        data_source=data_source,
        splits=splits,
        label_merge_enabled=label_merge_enabled,
        label_merge_map=merge_map,
        merged_label_counts=dict(merged_counts_total),
        unknown_merge_sources=unknown_sources,
    )
    return DatasetBundle(
        dataset_family=normalize_dataset_family(dataset_family),
        data_source=normalize_data_source(data_source),
        split_paths=split_paths,
        splits=splits,
        diagnostics=diagnostics,
    )


def diagnostics_to_dict(diagnostics: DatasetDiagnostics) -> Dict[str, object]:
    return {
        "dataset_family": diagnostics.dataset_family,
        "data_source": diagnostics.data_source,
        "train_labels": diagnostics.train_labels,
        "warnings": diagnostics.warnings,
        "is_degenerate_validation": diagnostics.is_degenerate_validation,
        "label_merge_enabled": diagnostics.label_merge_enabled,
        "label_merge_map": diagnostics.label_merge_map,
        "merged_label_counts": diagnostics.merged_label_counts,
        "unknown_merge_sources": diagnostics.unknown_merge_sources,
        "merge_summary": diagnostics.merge_summary,
        "split_stats": {
            split: {
                "num_rows": stat.num_rows,
                "num_sentences": stat.num_sentences,
                "num_labels": stat.num_labels,
                "label_counts": stat.label_counts,
                "train_label_coverage": stat.train_label_coverage,
                "unseen_labels_vs_train": stat.unseen_labels_vs_train,
            }
            for split, stat in diagnostics.split_stats.items()
        },
    }


def load_default_backend_split(split: str, root_dir: str | Path | None = None, dataset_family: str = "reviewlevel") -> SentenceDataset:
    return load_split(
        split=split,  # type: ignore[arg-type]
        dataset_family=dataset_family,
        data_source="backend_raw",
        root_dir=root_dir,
    )


def load_input_split(
    split: str,
    dataset_family: str,
    input_dir: str | Path | None = None,
) -> SentenceDataset:
    return load_split(
        split=split,  # type: ignore[arg-type]
        dataset_family=dataset_family,
        data_source="input_dir",
        input_dir=input_dir,
    )


def export_default_backend_splits(output_dir: str | Path, root_dir: str | Path | None = None) -> List[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for split in ["train", "val", "test"]:
        dataset = load_default_backend_split(split=split, root_dir=root_dir)
        output_path = out_dir / f"implicit_{split}.csv"
        dataset.to_csv(output_path)
        written.append(output_path)
    return written


def compare_label_spaces(datasets: Iterable[SentenceDataset]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for dataset in datasets:
        counter.update(dataset.unique_aspects())
    return dict(sorted(counter.items()))
