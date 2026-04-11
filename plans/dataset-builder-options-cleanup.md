# Plan: Dataset Builder Options Cleanup

> Source PRD: ReviewOp dataset_builder README and CLI behavior

## Architectural decisions

- **CLI surface**: Keep a small set of primary flags for common runs and preserve advanced flags only where they control distinct runtime behavior.
- **Run modes**: `research` remains the default full build path; `debug` remains the sampled/preview path.
- **LLM selection**: Explicit provider selection must override environment-based defaults.
- **Compatibility**: Preserve existing flag aliases only when they prevent breaking documented workflows.

---

## Phase 1: Verify option behavior end to end

**User stories**: Run dataset_builder with common and advanced option combinations without silent fallbacks or invalid defaults.

### What to build

Exercise the public CLI in representative modes:
- standard build
- sampled debug build
- preview/dry run
- zip-only
- OpenAI provider selection
- fallback provider selection via processor/environment

### Acceptance criteria

- [ ] Common documented commands succeed or fail with a clear error.
- [ ] Provider selection resolves to the requested backend.
- [ ] Sampled debug runs use debug mode and do not trip research-only guards.
- [ ] Existing aliases still behave as documented.

---

## Phase 2: Classify flags by necessity

**User stories**: Reduce CLI clutter while keeping the knobs that change real behavior.

### What to build

Review every dataset_builder option and categorize it as:
- core
- advanced but still needed
- redundant alias
- dead or legacy-only

### Acceptance criteria

- [ ] Each option has a clear classification.
- [ ] Redundant aliases are identified.
- [ ] Dead or duplicate flags are documented for removal or consolidation.

---

## Phase 3: Simplify the public CLI

**User stories**: Use fewer flags for the same supported workflows.

### What to build

Remove or consolidate redundant options where behavior is duplicated, and keep the smallest practical set of primary controls in the README.

### Acceptance criteria

- [ ] The README lists only the needed day-to-day flags.
- [ ] Advanced options remain available where they meaningfully change behavior.
- [ ] Removed flags have a migration note or replacement.

---

## Phase 4: Lock in regressions

**User stories**: Prevent future drift between README, parser, and runtime behavior.

### What to build

Add regression coverage for the remaining CLI contract, focusing on the option combinations that were simplified or retained as core behavior.

### Acceptance criteria

- [ ] Tests cover the supported CLI paths.
- [ ] A failing option mismatch is caught before release.
- [ ] Documentation and parser stay aligned.
