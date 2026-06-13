# Phase 5.7 Integrity Audit

**Status:** `PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED`

## Hardened Findings

- Removed class-label boundary scaling from `50_5_7_entropic_boundary/src/entropic_boundary_probe.c`.
- Removed the direct `load_scale()` boundary deformation constant.
- Runtime load deformation is derived from measured bounded memory/timing/worker observables.
- Runtime observables are averaged at the load-condition level before row generation, so per-row timing noise cannot masquerade as carrier geometry.
- Boundary thresholds are calibrated from training catalytic rows by load mode, then evaluated on holdout rows and null controls.
- Same-final-hash wrong-answer, wrong residual, destructive write, random reversible write, and shuffled schedule controls remain excluded.
- Raw jitter/boundary correlation is confounded by load level and is diagnostic only.
- Within-load carrier/boundary correlation beats within-load jitter/boundary correlation.

## Measured Runtime Gates

- Measured cache delta: `1.120241`
- Measured contention delta: `16.429834`
- Measured jitter delta: `0.770563`
- Independent load deformation source: `PASS_MEASURED_RUNTIME_OBSERVABLES`

## Integrity Verdict

`PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED`

The result justifies `PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED` only if the independent measured-runtime source gate passes with null exclusion and carrier correlation gates intact.
