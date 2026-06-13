# Phase 5.7: Entropic Boundary Geometry Probe

**Date:** 2026-06-08
**Status:** `PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED`
**Harness:** `50_5_7_entropic_boundary/src/entropic_boundary_probe.c`

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

## Phase 6 Bridge

Phase 5.7 is the invariant/null discipline feeder for Phase 6, not the physical substrate run itself.

Mapping into `50_6_fixed_point_substrate/SPEC_PHASE6_FIXED_POINT_SUBSTRATE.md`:

- G1 restoration: supported at the logical/control level by restored tape and same-final-hash wrong-answer exclusion.
- G3 basin -> invariant: not complete; 5.7 can score invariant strength and answer correlation once 5.9V supplies a basin label.
- G5 controls: same-final-hash wrong-answer, wrong residual, destructive-write, and reversible-null machinery already exist and should be extended to same-hash wrong-invariant Phase 6 controls.
- Claim boundary: 5.7 remains computational carrier geometry only. It does not prove a physical Mode C crossing.

Completed 5.7 bridge push:

1. Add `basin_id` from the 5.9V carrier selector to each invariant row.
2. Add `fixed_point_d` and public target hash from the Phase 6 public `(k,b)` map.
3. Score whether invariant strength predicts `d` beyond shuffled-map and same-hash wrong-invariant nulls.
4. Emit a Phase 6-ready invariant table with:
   - restoration hash pass/fail
   - basin id
   - invariant family
   - invariant strength
   - answer / `d` correlation
   - null effect size

Completed scorer artifact:

`50_5_7_entropic_boundary/results/phase6_invariant_scorer/PHASE5_7_PHASE6_INVARIANT_SCORER_RUN.md`

Final bridge verdict:

`PHASE5_7_PHASE6_PUBLIC_INVARIANT_REJECTED_BY_5_9V_CONTROLS`

The scorer consumed the real 5.9V target-coupled VID+5 and VID+6 basin labels, emitted `50_5_7_entropic_boundary/results/phase6_invariant_scorer/phase5_7_phase6_invariant_scores.csv`, and found 0 public candidates beyond shuffled/wrong-target controls. Classify the current survivor as `RESIDUAL_ARTIFACT_ONLY`, not a Phase 6 crossing candidate.
