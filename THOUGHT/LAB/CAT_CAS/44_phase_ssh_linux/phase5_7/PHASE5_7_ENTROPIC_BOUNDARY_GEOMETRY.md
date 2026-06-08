# Phase 5.7: Entropic Boundary Geometry Probe

**Date:** 2026-06-08
**Status:** `PHASE5_7_ENTROPIC_BOUNDARY_ROADMAP_ADDED`

## Objective

Test whether operational noise, chaos, and CPU-load entropy behave as observable boundary deformation of the CAT_CAS full-carrier geometry rather than simple degradation.

## Starting Evidence

Phase 5.6 is confirmed as `PHASE5_6_POLYTOPE_GEOMETRY_CONFIRMED`:

- Full T0/T1/T2/T3 carrier rows are generated directly.
- Same-final-hash wrong-answer controls are excluded at `1.000000`.
- Held-out balanced accuracy and catalytic true-positive rate are `1.000000`.
- Static projection hierarchy and fine residual-boundary deformation gates pass.
- Load/entropy deformation is intentionally assigned to Phase 5.7.

## Hypothesis

```
operational entropy / contention / jitter
  -> observable boundary deformation
  -> richer admissible carrier geometry
  -> answer-predictive invariant remains intact
  -> null and wrong-answer histories remain outside
```

## Required Harness

`session_scripts/phase5_7/entropic_boundary_probe.c`

The harness should reuse the Phase 5.6 full-carrier generator and add bounded load modes.

## Required Outputs

- `phase5_7/results/entropic_boundary_summary.csv`
- `phase5_7/results/load_boundary_raw.csv`
- `phase5_7/results/null_boundary_exclusion.csv`
- `phase5_7/results/residual_deformation_under_load.csv`
- `phase5_7/results/phase5_7_stdout.txt`

## Decision Gate

Promote only if:

- same-final-hash wrong-answer exclusion remains `>=0.95`
- catalytic true-positive rate remains `>=0.80`
- balanced held-out accuracy remains `>=0.80`
- MEDIUM or HIGH load changes boundary proxy by `>=10%`
- null exclusion does not collapse
- effect tracks carrier/boundary features, not raw jitter alone

## Claim Boundary

Phase 5.7 may claim computational boundary deformation only. It must not claim physical holography, AdS/CFT, quantum coherence, physical Kuramoto, Landauer violation, zero heat, or thermodynamic entropy reduction.
