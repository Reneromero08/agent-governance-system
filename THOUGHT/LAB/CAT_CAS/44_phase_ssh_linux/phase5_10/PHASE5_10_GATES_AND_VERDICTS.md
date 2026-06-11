# EXP44 PHASE 5.10 - GATES AND VERDICTS

**Parent:** `PHASE5_10_BOUNDARY_STATE_PREPARATION.md`
**Status:** SPEC
**Claim ceiling:** L4-5.

## Gates

| Gate | Name | PASS | PARTIAL | FAIL |
|---|---|---|---|---|
| 1 | Instrumentation Lock | TSC-thermal gate-delay witness moves outside its noise floor AND corroborates (k10temp / eff-freq / wall power) under the prep lever; coherent-strobe upgrade optional. NOT gated on actual Vcore (`VCORE_MEASUREMENT_BLOCKED` is the allowed default) | TSC witness moves but corroboration weak / strobe upgrade unvalidated | only decoded VID / requested values move (decoded VID is a request, not a witness) -> BLOCKED |
| 2 | Restoration Integrity | tape restores bit-for-bit across all valid runs | - | restoration corruption appears outside explicit destructive controls |
| 3 | Basin Classification | collapsed/mid/high defined from FROZEN, label-blind calibration thresholds | - | thresholds adjusted after the selection data |
| 4 | Basin Scan Coverage | load-history / prelude / thermal-band / P-state combos sampled across >=2 physical (thermal/rail) conditions for the scaling null | scan limited by instrumentation/platform | no usable scan |
| 5 | Transition Probability Estimation | P(basin \| prep, physical_state) estimated with enough repeats; lift outside the label-reshuffle null CI | - | too few repeats / lift inside the null CI |
| 6 | Basin Selection | at least one prep reliably biases a target basin above baseline AND the lift scales PARAMETRICALLY with physics | selection appears but confidence weak | no reproducible selection (directional-not-deterministic on the 5.9V family = HARD NO_REPRODUCIBLE_BASIN) |
| 7 | Artifact + No-Smuggle Controls | shuffled-label / no-prep / thermal-frequency-matched controls FAIL to reproduce selection AND the MANDATORY public-vs-d-oracle control is shown with the d-oracle proven able to win | - | a control reproduces the selection, OR the d-oracle null is vacuous (d-oracle can never win) |
| 8 | Phase 6 Readiness | 5.10C confirms reproducible, PARAMETRICALLY-SCALING boundary state preparation | basin structure exists but selection not yet reliable | no basin structure, no instrumentation, or basin invariant to physics (logical artifact) |

**Binding rule:** Gate 8 PASS is the *only* condition under which Phase 6 may begin.

## Substrate-identity rule (binding, mirrors GATE_QUESTIONS G-4)

A basin/selection that is **invariant** to physical conditions (temperature / rail) is a LOGICAL artifact
(cache-replacement / run-order / thermal-drift proxy), NOT a substrate basin - even if it passes the letter
of the Gate-6 selection control. The parametric-physical-scaling null is the decisive 5.10 discriminator and
must be able to KILL the claim (M-5).

## No-smuggle rule (binding, mandatory; mirrors 5.10C and the Phase-6 anti-smuggle battery)

The public-vs-d-oracle no-smuggle control is **mandatory** at Gate 7:
- If the **public** prep selects the target basin as well as the d-oracle prep, no information is smuggled
  (the desired no-smuggle outcome) - show it explicitly.
- The **d-oracle prep must be shown it CAN win** under some configuration. If the d-oracle can never beat the
  public prep, the "no smuggle" claim is VACUOUS and Gate 7 is a FAIL.
- The `same-final-hash-wrong-basin` and `restoration-destroyed` controls are pre-classified
  **optional-BLOCKED** on this rig (silent-core-hang / unrecoverable-state risk on the headless SSH-only 125W
  box). That is honest scoping, caps the no-smuggle strength in the verdict, and does NOT excuse the
  mandatory public-vs-d-oracle control.

## Phase 5.10 verdict labels

```
EXP44_PHASE5_10_BOUNDARY_STATE_PREPARATION_CONFIRMED
  Use if 5.10A, 5.10B, and 5.10C all pass (or pass with documented non-fatal partial instrumentation),
  AND the selection scales parametrically with physics, AND the no-smuggle control is shown.

EXP44_PHASE5_10_BASIN_STRUCTURE_CONFIRMED_SELECTION_PARTIAL
  Use if basin classes exist but selection is not yet reliable enough for Phase 6.

EXP44_PHASE5_10_INSTRUMENTATION_BLOCKED
  Use if the TSC-thermal witness cannot be corroborated enough to interpret basin results
  (decoded-VID-only movement). VCORE_MEASUREMENT_BLOCKED alone does NOT trigger this label.

EXP44_PHASE5_10_NO_REPRODUCIBLE_BASIN
  Use if scans find no controllable basin selection, OR the lever returns
  directional-but-not-deterministic again on the 5.9V family (a HARD verdict, NOT a soft PARTIAL).

EXP44_PHASE5_10_ARTIFACT_DOMINANT
  Use if apparent selection is explained by artifact, OR the basin is invariant to physical scaling
  (logical attractor, not substrate).

EXP44_PHASE5_10_READY_FOR_PHASE6_FIXED_POINT
  Use ONLY if 5.10C passes and Phase 6 can safely begin (Gate 8 PASS: reproducible AND parametrically
  scaling AND no-smuggle shown).
```

## Master verdict output

`phase5_10_master_verdict.csv` aggregates the subphase statuses, the gate results, the parametric-scaling
and no-smuggle outcomes, and the final Phase 5.10 verdict label, with the instrumentation-confidence caveat
(`VCORE_MEASUREMENT_BLOCKED` by default) carried explicitly.
