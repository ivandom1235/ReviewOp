# Plan: Deepen `build_dataset.py` into a pipeline kernel

> Source PRD: `plans/build-dataset-pipeline-kernel-prd.md`

## Architectural decisions

Durable decisions that apply across all phases:

- **Entry point**: `build_dataset.py` remains the CLI and compatibility wrapper.
- **Kernel boundary**: a dedicated pipeline boundary owns orchestration state and stage coordination.
- **Report shape**: the report stays JSON-compatible and preserves existing field names.
- **Side effects**: file writes and export serialization stay at the outer edge of the pipeline.
- **Stage modules**: domain-specific behavior continues to live in the existing stage modules unless a small extraction is needed for testability.

---

## Phase 1: Extract report assembly boundary

**User stories**:
- As a test writer, I can verify pipeline report assembly without running a full build.
- As a maintainer, I can inspect pipeline outputs through a smaller interface.

### What to build

Pull the report construction logic behind a pure boundary that accepts the already-computed pipeline state and returns the report payload. Keep the emitted report fields and blocking decisions unchanged.

### Acceptance criteria

- [ ] A focused test covers report assembly from a minimal synthetic pipeline state.
- [ ] The CLI output remains JSON-compatible and preserves existing report keys.
- [ ] The orchestration layer no longer inlines the full report dictionary construction.

---

## Phase 2: Introduce kernel-owned pipeline state

**User stories**:
- As a maintainer, I can understand the full pipeline flow without scanning a large function.
- As a refactorer, I can move stage logic without changing downstream behavior.

### What to build

Introduce a kernel boundary that owns the sequence of stage outputs and returns a structured result bundle. The outer entrypoint should delegate to that boundary instead of manually threading every stage artifact through one long function.

### Acceptance criteria

- [ ] The main pipeline path delegates orchestration to a kernel boundary.
- [ ] Stage outputs are grouped into a structured result rather than scattered locals.
- [ ] Existing smoke behavior and output artifacts remain stable.

---

## Phase 3: Separate side effects from orchestration

**User stories**:
- As a reviewer, I can tell where computation ends and file writing begins.
- As a test writer, I can exercise the kernel without touching the filesystem.

### What to build

Move artifact writing, report persistence, and export serialization into the outermost layer. Keep the kernel focused on computation and result assembly.

### Acceptance criteria

- [ ] The kernel can run without writing files.
- [ ] Side-effecting export code is isolated from the core pipeline flow.
- [ ] Report and artifact writes happen in a narrow wrapper around the kernel.

---

## Phase 4: Reduce `build_dataset.py` to wiring

**User stories**:
- As a maintainer, I can use `build_dataset.py` as a thin entrypoint.
- As a refactorer, I can reason about the pipeline from a smaller surface area.

### What to build

Shrink `build_dataset.py` so it primarily parses inputs, resolves config, invokes the kernel, and handles final persistence. Leave the stage logic and report construction in dedicated boundaries.

### Acceptance criteria

- [ ] The file is materially smaller and easier to navigate.
- [ ] The public CLI remains unchanged.
- [ ] Regression coverage protects the kernel boundary and the report payload.
