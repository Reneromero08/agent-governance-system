# Phase 5.7 Phase 6 Invariant Scorer Run

Verdict: `PHASE5_7_PHASE6_PUBLIC_INVARIANT_REJECTED_BY_5_9V_CONTROLS`

Objective: consume real 5.9V target-coupled basin labels and test whether public-target survivors beat shuffled and wrong-target controls.

- Table: `phase5_7/results/phase6_invariant_scorer/phase5_7_phase6_invariant_scores.csv`
- Rows scored: `16`
- Public selector rows: `4`
- Public candidates beyond null controls: `0`
- Best public null effect size: `0.000000`

## Source Verdicts

- `vid5_target_coupled`: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`
- `vid6_target_coupled`: `PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC`

## Gate Readout

- G1 restoration: `PASS`; all consumed 5.9V target-coupled reports have 0 restoration failures.
- G3 basin -> invariant: `ATTEMPTED_REJECTED`; public basin labels do not beat shuffled/wrong-target controls.
- G5 controls: `PASS_AS_REJECTION`; shuffled and wrong-target controls dominate or match public.
- G7 audit: `PASS`; this report refuses Mode C crossing and emits residual-artifact classification.

## Decision

No public selector produced a positive null effect size. Classify the current survivor as `RESIDUAL_ARTIFACT_ONLY`, not a CAT_CAS primitive and not a Phase 6 crossing candidate.
