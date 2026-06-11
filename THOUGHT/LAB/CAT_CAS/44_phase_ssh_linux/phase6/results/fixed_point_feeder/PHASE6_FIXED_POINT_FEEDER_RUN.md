# Phase 6 Fixed-Point Feeder Run

Verdict: `PHASE6_FEEDER_BASELINES_READY__5_9V_DIRECTIONAL_NOT_DETERMINISTIC`

Purpose: push Phase 5.7-5.9 toward the Phase 6 fixed-point substrate spec without claiming a physical Mode C crossing.

Target generator note: this dry run uses `M = 2.00 * sqrt(N)`, still a constant-factor `M ~ sqrt(N)` public Fourier table, because `1.00 * sqrt(N)` did not reliably produce unique small-n targets under the fixed `M/4` threshold.

## A/B Baseline Dry Run

| n | N | M | unique accept | accept count | A evals | B restore | best score x=d |
|---|---|---|---------------|--------------|---------|-----------|----------------|
| 8 | 256 | 32 | 1 | 1 | 103 | 1 | 1 |
| 10 | 1024 | 64 | 1 | 1 | 171 | 1 | 1 |
| 12 | 4096 | 128 | 1 | 1 | 1686 | 1 | 1 |
| 14 | 16384 | 256 | 1 | 1 | 4587 | 1 | 1 |
| 16 | 65536 | 512 | 1 | 1 | 19131 | 1 | 1 |

## 5.9V Basin Selector Audit

| selector | n | collapsed | mid | high | top basin | top rate | noncollapse | anti-high |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|
| branch_prelude | 4 | 0 | 3 | 1 | mid | 0.750 | 1.000 | 0.750 |
| cache_prelude | 4 | 2 | 2 | 0 | mid | 0.500 | 0.500 | 1.000 |
| collapse_prelude | 4 | 2 | 0 | 2 | high | 0.500 | 0.500 | 0.500 |
| quiet | 4 | 2 | 1 | 1 | collapsed | 0.500 | 0.500 | 0.750 |
| reset_p0 | 4 | 2 | 0 | 2 | collapsed | 0.500 | 0.500 | 0.500 |
| syscall_prelude | 4 | 0 | 2 | 2 | high | 0.500 | 1.000 | 0.500 |

## Gate Readout

- G1 restoration discipline: `PASS_FOR_MODE_B_DRY_RUN` (5/5 hash restores).
- G2 A/B baseline: `PASS_SOFTWARE_BASELINES_EXIST`; Mode B intentionally costs the same eval count as Mode A.
- Fixed-point target uniqueness: `PASS` for all generated dry-run targets.
- 5.9V basin selector: `DIRECTIONAL_NOT_DETERMINISTIC`; current evidence can bias basin family but cannot yet select a stable answer-bearing basin.
- G3 basin -> invariant: `ATTEMPTED_NOT_PASSED`; target-coupled VID+5/VID+6 matrices physically shaped prelude/workload from public `(k,b)` payloads, but public did not select a reproducible answer-bearing basin.
- G4 no-smuggle: `PASS_AS_REJECTION`; public, shuffled, wrong-target, and d-oracle controls were run. The strongest VID+6 selector was shuffled/nonpublic, not public.
- G5 controls: `PARTIAL_PASS_AS_REJECTION`; wrong/shuffled/oracle controls are physically coupled for the 5.9V feeder, while same-hash wrong-invariant remains a Phase 5.7/Phase 6 invariant-scoring control.
- G6 scaling: `BASELINE_ONLY`; A/B rows provide the scaling harness, not a Mode C curve.
- G7 audit: `ACTIVE`; this report refuses any crossing claim.

## Phase 5.7-5.9 Push Direction

- 5.7 should receive Phase 6 labels: basin id, invariant strength, and answer correlation against same-hash wrong-invariant nulls.
- 5.8 is sufficient as the boundary lifecycle object: borrow, couple, relax, read, uncompute, verify.
- 5.9V is still the bottleneck. It has now rejected the current public-prelude family under VID+5, VID+6, longer duration, and target-coupled workload shaping.

## Current Blocker

`PUBLIC_TARGET_COUPLING_DOES_NOT_SELECT_PUBLIC_BASIN`

Do not rerun the same public-prelude family. The next useful push requires a qualitatively different coupling mechanism, such as cache-set/address-topology coupling or a Phase 5.7 invariant scorer consuming the 5.9V basin labels.
