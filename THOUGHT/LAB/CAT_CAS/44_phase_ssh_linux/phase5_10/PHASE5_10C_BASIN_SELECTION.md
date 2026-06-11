# EXP44 PHASE 5.10C - BASIN SELECTION

**Parent:** `PHASE5_10_BOUNDARY_STATE_PREPARATION.md`
**Prerequisite:** 5.10A complete; 5.10B with `basin_thresholds_frozen.json` frozen
**Status:** SPEC
**Claim ceiling:** L4-5.
**This is the gating subphase: Phase 6 may not run until 5.10C passes.**

## REALIZED RESULT (this session)

**NOT REACHED.** 5.10B found **no basin to select** (the apparent load-history basin was a drift
artifact, `ARTIFACT_CONFIRMED` under full randomization), so there is nothing for 5.10C to steer. The
5.10A witness **exists** (driven compute-only timing lock-in), but the rail carries **no preparable
history** - state preparation has no target. 5.10C therefore does not pass, and the Phase 6 Prerequisite
Gate (Gate 8) is **NOT satisfied**: Phase 6 does not run on this rig. Claim ceiling L4-5.

## Question

Can the system **intentionally** prepare collapsed / mid / high carrier basins using the chosen prep
lever, with the selection scaling parametrically with physics and all controls failing?

## Purpose

Turn basin observation (5.10B) into controlled **boundary state preparation**. This is the point where
the prep lever becomes *state preparation*, not *work toward an answer*. There is still NO fixed-point
map here - that is Phase 6.

## Lever (inherited from 5.10B; binding)

Select with the **gated electrical PDN load-history / dI/dt envelope** (LEAD lever). Voltage/VID is a
clamped covariate; frequency-detuning is a structural zero - neither is a selection knob. Read on the fast
PDN channel via the coherent 2.67 MHz strobe, witnessed by the 5.10A TSC-thermal probe.

## Protocol

At fixed VID / P-state / tape size / thermal band:
```
1. Choose a target basin: collapsed | mid | high
2. Choose a load-history / dI/dt prep expected to bias the target basin
3. Run randomized repeats
4. Record the observed basin (frozen-threshold classifier from 5.10B)
5. Estimate transition probabilities:  P(observed_basin | prep, physical_state)
6. Compare against controls
7. Test parametric physical scaling (does the selection lift move with temperature / rail? - 5.10B null)
```

## Repeats

```
Minimum:   20 repeats per prep class
Preferred: 50 repeats per prep class
```

Lift must be **outside the binomial CI of a label-reshuffle null**, with multiple-comparison correction
across prep classes. (n=10 directional rates are not enough - that was the 5.9V gap.)

## Required prep / prelude classes

```
load_history: up-from-idle
load_history: down-from-high
didt_envelope: on-resonance aggressor
didt_envelope: off-resonance aggressor   (also serves as a control)
quiet
reset_p0
```

## Required controls (each must FAIL to reproduce the selection)

```
shuffled prelude/prep labels
randomized run order
wrong prep (off-resonance / mismatched envelope)
no-prep baseline
temperature-matched control
effective-frequency-matched control
tape-size matched control
```

### No-smuggle control (MANDATORY - mirrored from the Phase-6 anti-smuggle battery)

5.10 has no fixed-point map `f` and no invariant `d` yet, but the **public-vs-d-oracle no-smuggle control
is mandatory here** so the smuggling surface is closed before Phase 6 inherits it:

```
public prep    : a prep that knows ONLY the public/observable configuration
d-oracle prep  : a (constructed) prep that is ALLOWED to know the eventual target/invariant
```

Binding requirements:
```
- The public-vs-d-oracle separation control is MANDATORY (not optional). If the public prep already
  selects the target basin as well as the d-oracle prep, the selection is congruent with the substrate's
  native dynamics and carries no smuggled information - that is the desired no-smuggle outcome and must be
  shown explicitly.
- The d-oracle control must be shown it CAN win. If the d-oracle prep can never beat the public prep under
  ANY configuration, the "no smuggle" claim is VACUOUS (you proved nothing could leak because nothing could
  ever select). Demonstrate a configuration where the d-oracle prep DOES separate, so the null has teeth.
- This control echoes the 5.9V verdict PUBLIC_TARGET_COUPLING_DOES_NOT_SELECT_PUBLIC_BASIN: a public prep
  that fails to select is a NULL on coupling, not a pass.
```

### Pre-classified UNREACHABLE controls (optional-BLOCKED on this rig)

The following controls are **pre-classified optional-BLOCKED** because they cannot be run safely on the
headless, SSH-only, 125W rig (silent-core-hang / unrecoverable-state risk; no external recovery path):

```
same final hash but wrong basin          -> optional-BLOCKED (cannot force the wrong-basin/same-hash
                                            state without risking a silent core hang)
restoration-destroyed control            -> optional-BLOCKED (destructive; only if a future bench rig with
                                            external recovery exists)
```

Marking these optional-BLOCKED is honest scoping, not a pass. It caps the strength of the no-smuggle claim
and must be stated in the verdict. It does NOT excuse the MANDATORY public-vs-d-oracle control above.

## Success criterion (boundary state preparation CONFIRMED)

All of:
```
- at least one prep selects a basin above baseline (lift outside the label-reshuffle null CI)
- selection survives randomized repeats
- selection survives the artifact controls
- the MANDATORY public-vs-d-oracle no-smuggle control is shown (with the d-oracle proven able to win)
- the selection lift scales PARAMETRICALLY with a physical condition (temperature / rail) - 5.10B null;
  an invariant-to-physics selection is a LOGICAL artifact, not substrate preparation
- tape restores bit-for-bit on all non-destructive runs
- worker lifetime remains clean (0 join failures / migrations / thermal aborts)
- actual physical state is instrumented or honestly partial-instrumented (5.10A; VCORE_BLOCKED is allowed)
- controls FAIL to reproduce the selection
```

## Required output: `phase5_10c_transition_matrix.csv`

Columns:
```
target_basin
prep                        # load-history / didt-envelope / prelude class
physical_state
n_runs
collapsed_count
mid_count
high_count
p_collapsed
p_mid
p_high
selection_lift_vs_baseline
p_value_or_bootstrap_ci
parametric_scaling          # scales_with_physics | invariant_logical | undetermined
no_smuggle_status           # public_selects | d_oracle_only | vacuous (d-oracle never wins)
restoration_failures
artifact_status
verdict
```

## Allowed statuses

```
PHASE5_10C_BOUNDARY_STATE_PREPARATION_CONFIRMED
PHASE5_10C_BASIN_SELECTION_PARTIAL
PHASE5_10C_NO_REPRODUCIBLE_SELECTION
PHASE5_10C_ARTIFACT_DOMINANT
PHASE5_10C_INSTRUMENTATION_BLOCKED
PHASE5_10C_FAILED
```

## Gates touched

- **Gate 5 (Transition Probability Estimation):** PASS iff P(basin | prep, physical_state) is estimated
  with enough repeats (>= the minimum above) and lift is outside the label-reshuffle null CI.
- **Gate 6 (Basin Selection):** PASS iff at least one prep reliably biases a target basin above baseline AND
  the lift scales parametrically with physics; PARTIAL if selection appears but confidence is weak; FAIL if
  no reproducible selection (directional-but-not-deterministic on the 5.9V family = the HARD
  `EXP44_PHASE5_10_NO_REPRODUCIBLE_BASIN`, not a soft PARTIAL).
- **Gate 7 (Artifact + No-Smuggle Controls):** PASS iff shuffled-label, no-prep, thermal/frequency-matched
  controls FAIL to reproduce the selection AND the mandatory public-vs-d-oracle no-smuggle control is shown
  with the d-oracle proven able to win (a vacuous d-oracle null = Gate-7 FAIL).
- **Gate 8 (Phase 6 Readiness):** PASS only if 5.10C confirms reproducible, parametrically-scaling boundary
  state preparation.

## A8 note

A clean basin selection is the desired result, but if the selection lift is large and surprising, treat it
with A8 suspicion: re-check that the prep is state preparation and not a hidden encoding of the eventual
Phase-6 answer (there is no map here yet, so the smuggling surface is small - but the mandatory
public-vs-d-oracle control above is precisely what closes it). Confirm the basin metric is not secretly a
proxy for run order, thermal drift, or spike artifacts, and that the selection scales with physics rather
than being a config-invariant logical attractor.
