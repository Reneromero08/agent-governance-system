# EXP50 PHASE 5.10B - PREP-LEVER / PRELUDE BASIN SCAN

(historical filename retains "VOLTAGE"; the voltage lever is RETIRED - see lever re-key below)

**Parent:** `PHASE5_10_BOUNDARY_STATE_PREPARATION.md`
**Prerequisite:** 5.10A complete (PASS or documented PARTIAL/BLOCKED)
**Status:** SPEC
**Claim ceiling:** L4-5.

## REALIZED RESULT (this session)

**No retained basin.** The apparent load-history difference (up-from-idle vs down-from-high) that looked
like a preparable basin was a **drift artifact**: under full randomization of run order it did not
survive. `ARTIFACT_CONFIRMED` - the structure was a load-history/thermal-drift proxy, not a
parametrically-scaling substrate basin (the sharpened scaling null, as designed, killed it). This is the
HARD verdict `EXP50_PHASE5_10_NO_REPRODUCIBLE_BASIN`, not a soft PARTIAL: the rail carries the 5.10A
witness but no preparable history. Claim ceiling L4-5.

## Question

How does the chosen prep lever (electrical PDN load-history) map into carrier basins, and does any
apparent basin structure **scale parametrically with a physical condition** or is it a logical artifact?

## Purpose

Map the substrate basin landscape **before** attempting selection. 5.10B observes; 5.10C steers. Do
not attempt intentional selection here, and do not freeze thresholds against the selection goal.

## Prep-lever re-key (load-bearing, supersedes the voltage-era design)

The original 5.10B led with **voltage / VID** and treated frequency as a knob. Both are retired:
- **Voltage / VID is clamped and dead** -> demoted to a **logged covariate only**, never the lever.
- **Frequency-detuning is a STRUCTURAL ZERO** (cores are PLL-slaved to one crystal; infinite-stiffness
  phase clamp) -> not a lever, not a knob; do not sweep it as a prep variable.

### LEAD prep variable: gated electrical PDN load-history (dI/dt envelope)

The one prep variable genuinely actuatable on the clamped, fanless, SSH-only rig is a
software-controllable **dI/dt aggressor / load-history envelope** on the shared PDN:

```
LEAD lever  = gated power-virus / load-history envelope on the shared PDN
              (up-from-idle vs down-from-high to the SAME final config; frequency-swept dI/dt aggressor;
               power-virus dwell). Read on the FAST PDN electrical channel via the coherent strobe,
               witnessed by the 5.10A TSC-thermal probe (and, when available, an external rail probe).
```

Thermal band and load history are the physical axes that the witness can actually resolve. Note the
thermal-bistability branch (degrade cooling to push loop gain beta>=1) is **DEAD on this rig** (no PWM
node, fan1:0 RPM, read-only k10temp, unsafe to degrade cooling on a headless 125W part); do NOT design
the lead scan around reaching the saddle-node fold. Thermal band enters as a **logged covariate axis** for
the parametric-scaling null (below), not as an actuated lever.

### Demoted to logged covariates (NOT levers)

```
requested VID / decoded VID   - clamped; logged, never swept as a prep variable
P-state                       - logged
frequency / effective freq    - logged as a covariate and witness; NOT detuned as a lever (structural zero)
thermal band                  - logged covariate axis (used for the scaling null, not actuated)
```

## Variables (logged per run)

```
requested VID                 (covariate)
decoded VID                   (covariate)
measured Vcore                (VCORE_MEASUREMENT_BLOCKED unless bench DMM; covariate)
P-state                       (covariate)
effective frequency           (witness / covariate)
load_history                  (LEAD lever: up-from-idle | down-from-high | flat)
didt_envelope                 (LEAD lever: aggressor pattern id / sweep frequency / dwell)
prelude type                  (candidate carrier, logged - NOT the primary lever)
thermal band                  (covariate axis for the scaling null)
tape size
worker mode
run order
restoration integrity
tsc_thermal_delay             (5.10A canonical witness)
strobe_locked                 (was readout phase-locked to the 2.67 MHz strobe)
```

## Prelude types (candidate carriers, logged - secondary to the load-history lever)

```
quiet
reset_p0
cache_prelude
syscall_prelude
branch_prelude
integer_churn_prelude
memory_sweep_prelude
thermal_prelude
pstate_transition_prelude
```

## Measurement timescale (binding)

Measure at the **THERMAL timescale**: seconds-scale dwell per setting with dwell-time extrapolation, NOT
fast single-shot sampling. The canonical witness (TSC gate-delay) settles in seconds at ~1.5 bits per
settling time; a fast read under-resolves it. The FAST PDN electrical channel is read only through the
coherent strobe (phase-locked averaging), which is itself a slow-integration measurement built on top of
the seconds-scale dwell. Report dwell time and the extrapolation for every basin estimate.

## Fixed tape sizes (primary)

```
2048
4096
```

**Do not mix tape-size scaling with basin selection.** Tape-size is held fixed per scan block; scaling
is a separate later study, not part of basin discovery.

## Basin classes

```
collapsed
mid
high
```

Basin classification must be defined from **calibrated carrier metrics**, not eyeballed.

## Candidate carrier metrics (for classification)

```
boundary thickness
boundary carrier amplitude
timing CV
stable-window thickness
spike-free thickness
mean radius
D_eff
spectral entropy
tsc_thermal_delay_regime    (the 5.10A canonical-witness band)
restoration state
```

## Threshold freezing rule (binding)

```
1. Run a dedicated calibration set.
2. Derive collapsed/mid/high thresholds from the calibration set.
3. FREEZE them in basin_thresholds_frozen.json.
4. Classify all future runs with the frozen thresholds.
```

**Do not redefine thresholds after seeing any selection result** (Gate 3 FAIL if violated).

## SHARPENED parametric-physical-scaling null (binding; adopt as primary discriminator)

A true substrate basin must show **parametric physical scaling**: its size/separation must move
**monotonically** with a physical condition (die temperature / cooling / rail). A logical-state attractor
(cache-replacement, run-order, thermal-drift proxy) is **invariant** to those physical axes. So:

```
- Repeat the scan across at least two thermal bands (warm vs cold, whatever the fanless rig reaches passively)
  and, when reachable, two rail/strobe conditions.
- A candidate basin counts as PHYSICAL ONLY IF its class separation scales monotonically with the physical
  axis. If the basin structure is invariant to temperature / rail, classify it as a LOGICAL artifact, not a
  substrate basin -> PHASE5_10B_ARTIFACT_DOMINANT (logical), even if the clusters look clean.
```

This null must be able to KILL the basin claim (M-5). An apparent basin that does not scale parametrically
is "the CPU as a CPU through a side door", not a substrate basin.

## WHAT IS NEW VS 5.9V (state explicitly; the lever changed)

5.9V swept a **voltage/public-prelude** lead variable on the FAST timescale and returned
`DIRECTIONAL_NOT_DETERMINISTIC` (public_kb_prelude == quiet; the public-prelude basin select did NOT
reproduce). 5.10B is **not** a re-run of that family. What is new:

```
- LEVER:      voltage/VID (clamped, dead)  ->  gated electrical PDN load-history / dI/dt envelope.
- TIMESCALE:  fast single-shot             ->  thermal-timescale dwell + dwell-time extrapolation.
- READOUT:    raw TSC                       ->  TSC-thermal witness + coherent 2.67 MHz strobe averaging.
- DISCRIMINATOR (NEW, primary): the sharpened parametric-physical-scaling null - a basin must move with
  temperature/rail or it is logical, not substrate. 5.9V had no such scaling test.
```

If 5.10B merely re-runs the voltage/public-prelude family on the fast timescale, it is OUT OF SCOPE and does
not count.

## Pre-committed hard verdict (binding; freeze before any run)

If the scan returns **directional-but-not-deterministic again on the same family** (i.e. a lever biases a
basin in repeated runs but the lift does not survive the binomial-CI label-reshuffle null with
multiple-comparison correction - the exact 5.9V failure mode), the verdict is the **HARD**
`EXP50_PHASE5_10_NO_REPRODUCIBLE_BASIN`, **NOT** a soft PARTIAL. This is committed here, before data, so it
cannot be softened post-hoc.

## Required output: `phase5_10b_basin_scan.csv`

Columns:
```
run_id
vid_request                 # covariate
vid_decoded                 # covariate
vcore_measured              # BLOCKED unless bench DMM
pstate                      # covariate
effective_frequency         # witness / covariate
load_history                # LEAD lever
didt_envelope               # LEAD lever
prelude                     # candidate carrier
thermal_band                # covariate axis for scaling null
tape_size
worker_mode
tsc_thermal_delay           # 5.10A canonical witness
strobe_locked
boundary_thickness
carrier_amplitude
timing_cv
spike_rate
stable_thickness
spike_free_thickness
basin_class
parametric_scaling          # scales_with_physics | invariant_logical | undetermined
restoration_ok
hash_match
notes
```

## Allowed statuses

```
PHASE5_10B_BASIN_SCAN_COMPLETE          # scan ran; classes populated under frozen thresholds; scaling tested
PHASE5_10B_BASIN_SCAN_PARTIAL           # coverage limited by instrumentation/platform
PHASE5_10B_NO_BASIN_STRUCTURE           # carrier shows no separable basin structure
PHASE5_10B_ARTIFACT_DOMINANT            # apparent structure explained by artifact (spike/order/thermal drift)
                                        #   OR basin invariant to physical scaling -> logical, not substrate
PHASE5_10B_INSTRUMENTATION_BLOCKED      # cannot interpret without 5.10A witnesses
```

## Gates touched

- **Gate 3 (Basin Classification):** PASS iff classes are defined from frozen calibration thresholds.
- **Gate 4 (Basin Scan Coverage):** PASS iff load-history / prelude / thermal-band / P-state combinations are
  sufficiently sampled across at least two physical (thermal/rail) conditions for the scaling null; PARTIAL
  if limited by instrumentation/platform constraints.
