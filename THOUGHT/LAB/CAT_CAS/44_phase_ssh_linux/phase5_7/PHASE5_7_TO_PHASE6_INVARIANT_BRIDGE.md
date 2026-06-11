# Phase 5.7 to Phase 6 Invariant Bridge

Verdict: `PHASE5_7_READY_AS_PHASE6_INVARIANT_SCORER__WAITING_ON_5_9V_BASIN_LABELS`

## Purpose

Phase 5.7 becomes the Phase 6 invariant/null scoring layer. It does not select the physical basin and does not claim Mode C. Its job is to decide whether a restored-tape survivor is answer-predictive or merely residual artifact.

## Existing Evidence

- Same-final-hash wrong-answer exclusion: `1.000000`
- Holdout accuracy: `1.000000`
- Balanced accuracy: `1.000000`
- Catalytic true-positive rate: `1.000000`
- Within-load carrier/boundary correlation: `0.996774`
- Within-load jitter/boundary correlation: `0.000000`
- Class-label boundary leakage: `PASS`
- Independent load deformation source: `PASS_MEASURED_RUNTIME_OBSERVABLES`

## Phase 6 Mapping

| Phase 6 Gate | 5.7 Contribution | Current State |
|---|---|---|
| G1 restoration | restored tape and same-final-hash controls | logical support present |
| G3 basin -> invariant | invariant scorer once basin labels exist | waiting on 5.9V labels |
| G5 controls | wrong-answer, wrong-residual, destructive/reversible null machinery | extend to same-hash wrong-invariant |
| G7 audit | leakage and null discipline | active |

## Required Extension

Add a Phase 6 invariant table with one row per run:

- `target_public_hash`
- `n`
- `fixed_point_d`
- `selector`
- `basin_id`
- `restoration_hash_pass`
- `invariant_family`
- `invariant_strength_t0_t1_t2_t3`
- `answer_correlation`
- `same_hash_wrong_invariant_score`
- `shuffled_map_score`
- `null_effect_size`

## Stop Condition

5.7 is complete as a bridge only when the invariant scorer consumes a real 5.9V/Phase 6 basin label and shows whether the survivor predicts `d` beyond same-hash wrong-invariant and shuffled-map controls.
