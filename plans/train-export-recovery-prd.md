# PRD: Train Export Recovery and Benchmark Quality Cleanup

## Problem Statement

The current dataset build can produce a usable benchmark slice, but the train export is collapsing far below target size. In the latest mixed run, train export fell to single digits even though grounding, grouped split hygiene, and strict contamination checks were strong. The export pipeline is applying a stack of strict filters and balance constraints that are too aggressive for the surviving data, so valid rows are being removed late in the pipeline instead of being preserved earlier.

At the same time, the benchmark slice still has quality blockers that prevent promotion: implicit purity is too low, duplicate logical rows are too common, and the worst-domain regression remains above threshold.

The project needs a way to recover train size without relaxing the core safety gates, while also improving benchmark quality so the resulting artifact is both trainable and promotable.

## Solution

Keep the strict quality gates intact, but change the export behavior so train recovery happens earlier and the sentiment-balancing stage cannot shrink the train set into unusability. Recover more valid rows before the final strict pruning pass, and make the size recovery logic explicit when the available rows cannot satisfy all soft constraints.

Separately, improve benchmark selection so the exported benchmark has higher implicit purity, lower duplicate logical row rate, and better domain-family coverage. These benchmark cleanups should be treated as a distinct track so train recovery can proceed without loosening promotion rules.

## User Stories

1. As a dataset maintainer, I want train export to remain above a usable minimum, so that the artifact can support downstream model training.
2. As a dataset maintainer, I want strict leakage and contamination gates to remain enforced, so that recovered rows do not weaken the dataset.
3. As a dataset maintainer, I want rows that are valid but borderline to be recovered earlier in the pipeline, so that they are not discarded by late-stage pruning.
4. As a dataset maintainer, I want the sentiment balancer to respect train-size viability, so that it does not destroy the train set while enforcing ratio constraints.
5. As a dataset maintainer, I want the export report to explain why rows were lost, so that I can see whether the bottleneck is filtering, balancing, or top-up failure.
6. As a dataset maintainer, I want top-up recovery to pull from valid candidates before the final size check, so that recoverable rows can restore the train floor.
7. As a dataset maintainer, I want the pipeline to distinguish hard failures from soft review failures, so that recoverable rows are handled differently from irrecoverable rows.
8. As a dataset maintainer, I want benchmark purity to exceed the promotion threshold, so that the benchmark can be used as a reliable evaluation artifact.
9. As a dataset maintainer, I want duplicate logical rows in the benchmark to be reduced, so that the benchmark reflects genuine coverage rather than repetition.
10. As a dataset maintainer, I want worst-domain performance to stop blocking promotion, so that the selected benchmark slice is not disproportionately weak in one domain.
11. As a dataset maintainer, I want benchmark domain-family coverage to remain broad, so that the artifact still stresses cross-domain behavior.
12. As a dataset maintainer, I want grounding quality to stay strong, so that any recovery work preserves evidence-based labeling.
13. As a dataset maintainer, I want grouped split leakage to remain zero, so that the benchmark remains clean across train, val, and test.
14. As a dataset maintainer, I want the train export to remain non-general and leakage-free, so that recovered rows do not reintroduce contamination.
15. As a dataset maintainer, I want the build summary to surface the real bottleneck, so that I can tell whether future failures are caused by size pressure or quality pressure.

## Implementation Decisions

- Preserve the existing strict contamination and leakage gates as non-negotiable constraints.
- Treat train-size recovery as a first-class concern separate from benchmark promotion checks.
- Allow valid borderline rows to be reconsidered before the final train pruning step, rather than only after late-stage elimination.
- Keep sentiment ratio enforcement, but prevent it from shrinking the train export below a usable floor without explicit reporting.
- Expand recovery and top-up logic only for rows that already satisfy grounding and domain validity.
- Keep benchmark curation separate from train recovery so benchmark blockers can be fixed without altering train safety rules.
- Make benchmark purity, duplicate-rate reduction, and domain-family balance explicit selection criteria.
- Keep the report output stable enough to compare runs across iterations.

## Testing Decisions

- Test only externally visible behavior: final row counts, split counts, and promotion/blocking outcomes.
- Verify that train export stays above the usable floor while train leakage remains zero.
- Verify that the sentiment balancer no longer collapses the train export when valid rows are available for recovery.
- Verify that benchmark implicit purity crosses threshold and duplicate logical rows fall below the allowed maximum.
- Verify that grouped split leakage stays zero after any recovery or curation changes.
- Verify that the report still explains which stage removed rows and why.
- Use report-level and artifact-level checks as the main validation surface.

## Out of Scope

- Relaxing the core leakage or contamination thresholds.
- Redesigning the ontology or aspect taxonomy.
- Changing the underlying model architecture.
- Optimizing runtime performance unless it is directly tied to recovery behavior.
- Rewriting unrelated parts of the dataset builder or protonet pipeline.
- Adding new benchmark families or expanding the source corpus.

## Further Notes

The train-export problem and the benchmark-quality problem are related but should be solved as two distinct concerns. The train path needs viability first; the benchmark path needs promotability second. The implementation should preserve strictness where it protects data quality, and add recovery only where the current pipeline is over-pruning valid rows.
