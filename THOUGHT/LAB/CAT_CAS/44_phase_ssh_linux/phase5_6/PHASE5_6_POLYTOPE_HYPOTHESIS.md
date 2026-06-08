# Phase 5.6: Polytope / Positive-Geometry Hypothesis

**Date:** 2026-06-08
**Harness:** `session_scripts/phase5_6/polytope_hypothesis.c`
**Verdict:** `PHASE5_6_POLYTOPE_GEOMETRY_CONFIRMED`

## Result

- Full carrier rows generated: `264`
- Predictive features: `72`
- Same-final-hash wrong-answer exclusion: `1.000000`
- Held-out accuracy: `1.000000`
- Balanced accuracy: `1.000000`
- Catalytic true-positive rate: `1.000000`
- Static projection hierarchy: `PASS` with `6` separating/informative subspaces
- Fine residual-boundary deformation: `PASS`
- Load/entropy geometry: `DEFERRED_TO_PHASE5_7`, not required for static Phase 5.6 confirmation

## Interpretation

The hardened harness now uses real T0/T1/T2/T3 carrier state. Same-final-hash wrong-answer controls are represented by identical restored final hash but different T2 answer boundary state. Projection hierarchy and fine residual-boundary perturbation gates now pass inside the static carrier geometry scope. This fixes the earlier proxy-data weakness. The result is still not a physical holography claim; it is evidence about a computational carrier geometry only.

PHASE5_6_POLYTOPE_GEOMETRY_CONFIRMED
