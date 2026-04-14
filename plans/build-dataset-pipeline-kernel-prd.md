# PRD: Deepen `build_dataset.py` into a pipeline kernel

## Problem

`dataset_builder/code/build_dataset.py` has become the orchestration bottleneck for the dataset builder. It currently owns configuration handling, split construction, train recovery, benchmark selection, report assembly, and export writing in one long execution path.

That shape makes the code hard to test and hard to reason about:
- behavior is spread across many helpers with a large amount of cross-calling
- report assembly is interleaved with pipeline execution
- small changes in one stage can accidentally affect unrelated stages
- the file is too large to treat as a clean orchestration layer

## Goals

- Make the pipeline easier to understand by introducing a clear kernel boundary.
- Keep the public CLI and output formats stable.
- Move stage coordination behind a smaller interface that can be tested with focused boundary tests.
- Separate pure report assembly from side-effecting export/write steps.
- Preserve current quality behavior while refactoring structure.

## Non-goals

- No change to dataset semantics unless required to preserve existing behavior.
- No new quality policy or threshold changes.
- No rewrite of the stage-specific modules unless the refactor needs a small local helper extraction.

## User stories

- As a maintainer, I can inspect the pipeline through a smaller entrypoint rather than a monolithic function.
- As a test writer, I can verify pipeline report assembly without running a full build.
- As a refactorer, I can move pipeline logic into a kernel-like module without changing the emitted artifacts.
- As a reviewer, I can understand which stage owns which responsibility without scanning the whole file.

## Architectural direction

- Keep `build_dataset.py` as the CLI and compatibility entrypoint.
- Introduce a kernel boundary that receives preprocessed inputs and stage configuration, then returns structured pipeline outputs.
- Make report assembly a pure transformation over the pipeline outputs.
- Keep stage-specific logic in the existing domain modules where practical.
- Treat exports and file writes as the outermost side-effect layer.

## Success criteria

- The main orchestration path is visibly smaller and delegates to a kernel boundary.
- At least one end-to-end pipeline behavior is covered by a test that exercises a public-facing boundary.
- Report assembly can be tested without invoking the full dataset build.
- Existing smoke outputs remain stable after the refactor slice.
