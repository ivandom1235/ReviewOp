# Plan: Train Export Recovery and Benchmark Quality Cleanup

> Source PRD: `plans/train-export-recovery-prd.md`

## Architectural decisions

- **Train safety gates**: Keep leakage, contamination, and grounding constraints intact.
- **Train recovery flow**: Recovery should happen before final train-size enforcement and before any late-stage shrinkage that can make the train export unusable.
- **Benchmark curation flow**: Benchmark selection remains a separate path from train recovery and continues to enforce promotion quality checks.
- **Reporting contract**: The build report must continue to expose stage counts, rejection reasons, recovery outcomes, and promotion blockers.

---

## Phase 1: Reproduce the collapse in a thin diagnostic slice

**User stories**: 1, 4, 5, 6, 7, 15

### What to build

Trace a representative run through the full export path and confirm where the train set loses rows. Focus on the boundaries between review filtering, leakage filtering, top-up recovery, sentiment balancing, and final sizing. The goal is to establish the exact stage that turns a usable train set into a single-digit export.

### Acceptance criteria

- [ ] The report clearly shows train row counts before and after each export stage.
- [ ] The dominant loss point is identifiable from the report alone.
- [ ] The current promotion blockers are still visible and separated from the train-size issue.

---

## Phase 2: Preserve valid rows before final shrinkage

**User stories**: 1, 3, 4, 6, 7

### What to build

Introduce a recovery slice that keeps valid borderline rows in play longer, especially rows that are grounded, domain-valid, and non-general. The export should favor preserving usable rows over allowing a late-stage balance rule to exhaust the train set.

### Acceptance criteria

- [ ] Train export remains above the minimum viable floor when recoverable rows exist.
- [ ] Leakage and contamination gates remain unchanged.
- [ ] The report shows recovered rows separately from initially accepted rows.

---

## Phase 3: Make sentiment balancing viability-aware

**User stories**: 1, 4, 5, 15

### What to build

Keep sentiment balance constraints, but prevent them from forcing the train set below a usable threshold. The balancing step should be aware of how much train capacity it can safely consume and should report when it is constrained by scarcity rather than silently over-pruning.

### Acceptance criteria

- [ ] Balance constraints are still enforced when enough rows exist.
- [ ] The train export no longer collapses to a trivial size under scarcity.
- [ ] The report shows whether balancing was constrained by available data.

---

## Phase 4: Separate benchmark cleanup from train recovery

**User stories**: 8, 9, 10, 11, 12, 13

### What to build

Refine benchmark selection so it favors higher implicit purity, fewer duplicate logical rows, and healthier domain-family coverage. Keep this work isolated from train recovery so the benchmark can be promoted independently of train-size changes.

### Acceptance criteria

- [ ] Benchmark implicit purity meets the promotion threshold.
- [ ] Duplicate logical row rate drops materially.
- [ ] Worst-domain regression is reduced enough to clear the promotion gate.
- [ ] Grouped split leakage remains zero.

---

## Phase 5: Lock in report-level regression checks

**User stories**: 5, 14, 15

### What to build

Add or refine report-level checks that make the failure mode obvious on future runs: stage counts, shortage explanations, benchmark purity, duplication, and promotion status.

### Acceptance criteria

- [ ] The report explains whether failure came from filtering, recovery shortfall, or benchmark quality.
- [ ] The build result still flags promotion blockers explicitly.
- [ ] A future regression would be visible without manual inspection of raw artifacts.
