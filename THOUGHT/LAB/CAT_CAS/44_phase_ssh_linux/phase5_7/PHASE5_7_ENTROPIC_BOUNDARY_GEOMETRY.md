# Phase 5.7: Entropic Boundary Geometry Probe

**Date:** 2026-06-08
**Status:** `PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED`
**Harness:** `session_scripts/phase5_7/entropic_boundary_probe.c`

## Result

- Rows: `432`
- Same-final-hash wrong-answer exclusion: `1.000000`
- Holdout accuracy: `1.000000`
- Balanced accuracy: `1.000000`
- Catalytic true-positive rate: `1.000000`
- Medium boundary delta: `0.097904`
- High boundary delta: `0.217625`
- Null exclusion: `1.000000`
- Measured cache delta: `1.120241`
- Measured contention delta: `16.429834`
- Measured jitter delta: `0.770563`
- Raw carrier/boundary correlation: `0.558488`
- Raw jitter/boundary correlation: `0.730879`
- Within-load carrier/boundary correlation: `0.996774`
- Within-load jitter/boundary correlation: `0.000000`
- Class-label boundary leakage: `PASS`
- Independent load deformation source: `PASS_MEASURED_RUNTIME_OBSERVABLES`

## Verdict

`PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED`

## Integrity Finding

The hardened harness no longer scales the boundary proxy by class label. Same-final-hash wrong-answer controls, wrong residual controls, and destructive/reversible nulls remain excluded by carrier/restoration/answer-boundary constraints. The load deformation term is now derived from measured runtime observables collected during bounded memory/timing/worker contention probes, not from a direct programmed load-scale constant.

## Interpretation

Phase 5.7 supports computational boundary deformation under measured bounded runtime load: the carrier boundary proxy deforms while answer-predictive exclusion survives, and within-load residual correlation tracks carrier structure more strongly than jitter. It does not claim physical holography, AdS/CFT, quantum coherence, physical Kuramoto, Landauer violation, zero heat, or thermodynamic entropy reduction.
