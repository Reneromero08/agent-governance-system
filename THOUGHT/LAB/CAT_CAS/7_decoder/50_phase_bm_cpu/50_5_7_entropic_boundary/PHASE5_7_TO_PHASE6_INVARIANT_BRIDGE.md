# Phase 5.7 to Phase 6 Invariant Bridge

Verdict: `PHASE5_7_PHASE6_PUBLIC_INVARIANT_REJECTED_BY_5_9V_CONTROLS`

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
| G3 basin -> invariant | invariant scorer consumed 5.9V target-coupled basin labels | attempted/rejected |
| G5 controls | wrong-answer, wrong-residual, destructive/reversible null machinery plus 5.9V shuffled/wrong-target controls | pass as rejection |
| G7 audit | leakage and null discipline | active |

## Completed Extension

Created:

- `50_5_7_entropic_boundary/results/phase6_invariant_scorer/PHASE5_7_PHASE6_INVARIANT_SCORER_RUN.md`
- `50_5_7_entropic_boundary/results/phase6_invariant_scorer/phase5_7_phase6_invariant_scores.csv`

The Phase 6 invariant table now includes:

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

## Stop Condition Result

5.7 is complete as a bridge: the invariant scorer consumed real 5.9V/Phase 6 basin labels and showed that the current public survivor does not predict `d` beyond shuffled/wrong-target controls.

Final classification:

`RESIDUAL_ARTIFACT_ONLY`
