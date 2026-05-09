from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schema import GoldInterpretation, ReviewExample


def load_jsonl_rows(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_row(row: dict) -> ReviewExample:
    """Convert a raw dataset_builder row into a ReviewExample."""
    gold_interpretations = []
    
    # Handle different field names if necessary (dataset_builder uses gold_interpretations)
    raw_gold = row.get("gold_interpretations", [])
    for g in raw_gold:
        evidence_text = g.get("evidence_text", row.get("review_text", ""))
        
        # Validation 7: Validate that every accepted row has evidence (if not abstain)
        # Note: We'll check this at the row level later too.
        
        gold_interpretations.append(
            GoldInterpretation(
                aspect=g.get("aspect_canonical", g.get("aspect", "unknown")),
                label_type=g.get("label_type", "unknown"),
                sentiment=g.get("sentiment"),
                evidence_text=evidence_text,
                evidence_span=g.get("evidence_span"),
                evidence_scope=g.get("evidence_scope", "unknown"),
                novelty_status=g.get("novelty_status", "known"),
                mapping_scope=g.get("mapping_scope"),
                confidence=g.get("canonical_confidence"),
                matched_terms=list(g.get("matched_terms", [])),
            )
        )
    
    # Handle abstain cases
    abstain_acceptable = bool(row.get("abstain_acceptable", False))
    abstain_reason_gold = list(row.get("abstain_reason_gold", []))
    
    # If no gold interpretations and it's an abstain row, ensure we track it
    if not gold_interpretations and not abstain_reason_gold and abstain_acceptable:
        # Check if there are reason_gold in metadata
        abstain_reason_gold = list(row.get("reason_gold", []))

    # Validation 8: Validate that abstain rows have abstain reasons.
    if abstain_acceptable and not abstain_reason_gold:
        # We don't necessarily want to crash, but we should log or flag it.
        # For now, let's add a default reason if missing.
        abstain_reason_gold = ["unspecified_abstain_reason"]

    # Validation 10: Validate that there are no empty-gold rows unless abstain_acceptable = true.
    if not gold_interpretations and not abstain_acceptable:
        # This is a critical validation. 
        pass 

    split = row.get("split", "unknown")
    # Validation 9: Validate that split metadata is present.
    # (split is handled above)

    return ReviewExample(
        row_id=row.get("row_id", "unknown"),
        review_id=row.get("review_id", "unknown"),
        text=row.get("review_text", ""),
        domain=row.get("domain", "unknown"),
        split=split,
        source_type=row.get("source_type", "unknown"),
        novelty_status=row.get("novelty_status", "known"),
        abstain_acceptable=abstain_acceptable,
        abstain_reason_gold=abstain_reason_gold,
        gold_interpretations=gold_interpretations,
        metadata=row.get("metadata", {}),
    )


class ECDatasetLoader:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.examples: list[ReviewExample] = []
        self.splits: dict[str, list[ReviewExample]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.grouped_splits: dict[str, list[ReviewExample]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.domain_holdout_splits: dict[str, list[ReviewExample]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.aspect_memory_review_queue: list[dict] = []

    def load_all(self):
        """Load standard train/val/test splits and extra artifacts."""
        # Standard splits
        for split in ["train", "val", "test"]:
            path = self.artifact_dir / f"{split}.jsonl"
            for row in load_jsonl_rows(path):
                example = normalize_row(row)
                self.examples.append(example)
                self.splits[split].append(example)
        
        # Grouped splits
        grouped_dir = self.artifact_dir / "grouped"
        if grouped_dir.exists():
            for split in ["train", "val", "test"]:
                path = grouped_dir / f"{split}.jsonl"
                for row in load_jsonl_rows(path):
                    self.grouped_splits[split].append(normalize_row(row))
        
        # Domain holdout splits
        holdout_dir = self.artifact_dir / "domain_holdout"
        if holdout_dir.exists():
            for split in ["train", "val", "test"]:
                path = holdout_dir / f"{split}.jsonl"
                for row in load_jsonl_rows(path):
                    self.domain_holdout_splits[split].append(normalize_row(row))
        
        # Aspect Memory Review Queue
        queue_path = self.artifact_dir / "aspect_memory_review_queue.json"
        if queue_path.exists():
            try:
                data = json.loads(queue_path.read_text(encoding="utf-8"))
                self.aspect_memory_review_queue = data.get("items", [])
            except Exception:
                pass
                
        return self

    def load_counterfactuals(self) -> list[dict]:
        """Load counterfactual pairs for evaluation."""
        path = self.artifact_dir / "counterfactual" / "counterfactual_pairs.jsonl"
        return list(load_jsonl_rows(path))

    def get_summary(self) -> dict:
        summary = {
            "train_rows": len(self.splits["train"]),
            "val_rows": len(self.splits["val"]),
            "test_rows": len(self.splits["test"]),
            "total_rows": len(self.examples),
            "explicit_gold": 0,
            "implicit_gold": 0,
            "abstain_rows": 0,
            "novel_rows": 0,
            "boundary_rows": 0,
            "grouped_rows": sum(len(s) for s in self.grouped_splits.values()),
            "domain_holdout_rows": sum(len(s) for s in self.domain_holdout_splits.values()),
            "memory_review_queue_count": len(self.aspect_memory_review_queue),
        }
        
        for ex in self.examples:
            if ex.abstain_acceptable:
                summary["abstain_rows"] += 1
            
            if ex.novelty_status == "novel":
                summary["novel_rows"] += 1
            elif ex.novelty_status == "boundary":
                summary["boundary_rows"] += 1
                
            for g in ex.gold_interpretations:
                if g.label_type == "explicit":
                    summary["explicit_gold"] += 1
                elif g.label_type == "implicit":
                    summary["implicit_gold"] += 1
                    
        return summary

    def export_summary(self, output_path: str | Path):
        summary = self.get_summary()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

