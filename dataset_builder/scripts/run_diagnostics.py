from __future__ import annotations
import json
import argparse
import sys
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class DiagnosticConfig:
    fixtures_dir: Path
    expectations_path: Path
    output_root: Path
    mode: str = "all"
    strict: bool = False

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]

class DiagnosticValidator:
    def validate_row(self, row: Dict[str, Any], expectations: Dict[str, Any]) -> ValidationResult:
        errors = []
        gold = row.get("gold_interpretations", [])
        
        # 1. Source Type Check
        if "must_include_source_type" in expectations:
            actual_sources = {i.get("source_type") for i in gold}
            for expected in expectations["must_include_source_type"]:
                if expected not in actual_sources:
                    errors.append(f"Expected source_type {expected} not found in {list(actual_sources)}")

        # 2. Canonical Check
        if "must_include_canonical" in expectations:
            actual_canonicals = {i.get("aspect_canonical") for i in gold}
            for expected in expectations["must_include_canonical"]:
                if expected not in actual_canonicals:
                    errors.append(f"Expected canonical label {expected} not found in {list(actual_canonicals)}")

        # 3. Evidence Invariant: text == review_text[start:end]
        review_text = row.get("review_text", "")
        for i in gold:
            span = i.get("evidence_span")
            text = i.get("evidence_text")
            if span and len(span) == 2:
                start, end = span
                expected_text = review_text[start:end]
                if text != expected_text:
                    errors.append(f"Evidence invariant failure: '{text}' != '{expected_text}' (span {start}:{end})")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_accounting(self, stats: Dict[str, int]) -> ValidationResult:
        errors = []
        input_rows = stats.get("input_rows", 0)
        exported = stats.get("exported_rows", 0)
        rejected = stats.get("rejected_rows", 0)
        discarded = stats.get("discarded_rows", 0)
        
        sum_rows = exported + rejected + discarded
        if sum_rows != input_rows:
            errors.append(f"Accounting reconciliation failure: input({input_rows}) != sum({sum_rows}) [exp:{exported} + rej:{rejected} + disc:{discarded}]")
            
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

class DiagnosticRunner:
    def __init__(self, config: DiagnosticConfig):
        self.config = config
        self.validator = DiagnosticValidator()

    def load_fixtures(self) -> List[Dict[str, Any]]:
        fixtures = []
        for fixture_file in self.config.fixtures_dir.glob("*.jsonl"):
            with open(fixture_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        row["_fixture_file"] = fixture_file.name
                        fixtures.append(row)
        return fixtures

    def run(self):
        print(f"Starting diagnostic run in mode: {self.config.mode}")
        fixtures = self.load_fixtures()
        if not fixtures:
            print("No fixtures found!")
            sys.exit(2)

        # 1. Prepare Config
        from dataset_builder.config import BuilderConfig
        from dataset_builder.orchestrator.pipeline import run_builder_pipeline
        from dataset_builder.schemas.raw_review import RawReview

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output = self.config.output_root / f"diag_{ts}"
        run_output.mkdir(parents=True, exist_ok=True)

        # 2. Map Fixtures to RawReviews
        raw_reviews = [
            RawReview(
                review_id=f.get("id", f"diag_{i}"),
                group_id=f.get("id", f"diag_{i}"), # 1:1 for diagnostics
                domain=f.get("domain", "electronics"),
                domain_family="electronics" if f.get("domain") == "electronics" else "restaurant",
                text=f.get("review_text", ""),
                source_name=f.get("_fixture_file", "unknown"),
                metadata={"fixture_type": f.get("fixture_type")}
            ) for i, f in enumerate(fixtures)
        ]

        # 3. Load Expectations
        with open(self.config.expectations_path, 'r', encoding='utf-8') as f:
            global_expectations = yaml.safe_load(f)

        # 4. Run Pipeline
        cfg = BuilderConfig(
            input_dir=self.config.fixtures_dir,
            input_paths=tuple(self.config.fixtures_dir.glob("*.jsonl")),
            output_dir=run_output,
            overwrite=True,
            sample_size=len(raw_reviews)
        )

        try:
            print(f"Executing builder pipeline on {len(raw_reviews)} fixtures...")
            results = run_builder_pipeline(cfg, raw_reviews=raw_reviews)
            
            # 5. Validate Results
            # results contains 'quality' which is a QualityReport
            # and 'counts' which is split counts
            # We'll need to load the exported rows to validate them individually
            
            exported_rows = []
            for split in ["train", "val", "test"]:
                p = run_output / f"{split}.jsonl"
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                exported_rows.append(json.loads(line))

            # Row-level validation
            total_errors = []
            for row in exported_rows:
                # Find matching fixture for expectations
                fixture = next((f for f in fixtures if f["id"] == row["review_id"]), {})
                expectations = fixture.get("expected_behavior", {})
                
                v_res = self.validator.validate_row(row, expectations)
                if not v_res.is_valid:
                    total_errors.extend([f"Row {row['review_id']}: {e}" for e in v_res.errors])

            # Accounting validation
            stats = {
                "input_rows": len(raw_reviews),
                "exported_rows": results["counts"].get("total", 0),
                "rejected_rows": 0, # TODO: Get from pipeline if available
                "discarded_rows": 0 # TODO: Get from pipeline if available
            }
            # Behavioral Hard Gate: Must have at least one learned path match in diagnostic run
            source_types_seen = [i.get("source_type") for row in exported_rows for i in row.get("gold_interpretations", [])]
            learned_count = sum(1 for st in source_types_seen if st == "implicit_learned")
            
            if self.config.strict and learned_count == 0:
                total_errors.append(f"Learned path hard gate failure: 0 implicit_learned interpretations detected among {len(source_types_seen)} interpretations.")

            # Simplified accounting for now: sum of splits == results['counts']['total']
            v_acc = self.validator.validate_accounting(stats)
            if not v_acc.is_valid:
                total_errors.extend(v_acc.errors)

            # 6. Final Report
            summary = {
                "status": "passed" if not total_errors else "failed",
                "timestamp": ts,
                "mode": self.config.mode,
                "total_fixtures": len(raw_reviews),
                "exported_rows": len(exported_rows),
                "errors": total_errors
            }
            
            summary_path = run_output / "diagnostic_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

            print(f"Diagnostic run complete. Status: {summary['status'].upper()}")
            if total_errors:
                print(f"Errors found: {len(total_errors)}")
                for err in total_errors[:10]:
                    print(f"  - {err}")
                if self.config.strict:
                    sys.exit(1)
            
        except Exception as e:
            print(f"Diagnostic execution failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--expectations-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["all", "per-fixture", "mixed"], default="all")
    parser.add_argument("--strict", action="store_true")
    
    args = parser.parse_args()
    
    config = DiagnosticConfig(
        fixtures_dir=args.fixtures_dir,
        expectations_path=args.expectations_path,
        output_root=args.output_root,
        mode=args.mode,
        strict=args.strict
    )
    
    runner = DiagnosticRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
