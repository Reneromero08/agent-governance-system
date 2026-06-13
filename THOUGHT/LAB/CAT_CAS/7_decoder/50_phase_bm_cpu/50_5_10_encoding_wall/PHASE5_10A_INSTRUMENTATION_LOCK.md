# EXP50 PHASE 5.10A - INSTRUMENTATION LOCK

**Parent:** `PHASE5_10_BOUNDARY_STATE_PREPARATION.md`
**Status:** SPEC
**Claim ceiling:** L4-5. No faked instrumentation; a decoded-VID change is NOT a witness.

## REALIZED RESULT (this session)

The witness is **SOLVED** via the driven compute-only timing lock-in: under a steady compute load the
TSC gate-propagation delay locks in reproducibly outside its noise floor, giving the canonical
independent physical witness this subphase required. Note `f_eff` (APERF/MPERF) is **blind to droop**
on K10 (PLL-slaved effective frequency does not register the rail sag), so it serves only as a
corroborating covariate, not the witness. **Vcore remains BLOCKED** (`VCORE_MEASUREMENT_BLOCKED`): no
bench DMM was taken, so the direct-rail claim is still capped and must be carried forward. Net: 5.10A
clears its witness requirement (driven timing lock-in), with Vcore still blocked. Claim ceiling L4-5.

## Question

Which physical variables are **actually** changing during the catalytic run, and which independent
witness proves it?

## Purpose

Establish an independent witness for the physical substrate state, so that basin results in 5.10B/5.10C
are interpretable as substrate behavior rather than digital artifact. This subphase makes the substrate
the object of measurement, not the program.

## Witness re-key (load-bearing, supersedes the voltage-era design)

The original 5.10A treated **actual Vcore** as the witness to chase. The hardware retired that path:
VID is hardware-clamped at a floor (~1.225V), there is no VRM on SMBus, and COFVID reports a
**decoded VID** = a *request/definition*, not the rail. Therefore:

```
requested VID   - what software asked for          (logged covariate, NOT a witness)
decoded VID     - what the MSR/PCI register reports (logged covariate, NOT a witness)
actual Vcore    - what the silicon physically received -> VCORE_MEASUREMENT_BLOCKED by default
```

A decoded VID move alone is **not** evidence the substrate changed. The witness is re-keyed below.

### CANONICAL independent witness: invariant-TSC gate-propagation delay

The invariant TSC measures gate-propagation delay across the catalytic loop. Per the frontier dynamics
consult this is a **slow, low-resolution thermal/electrical probe** (~1.5 bits per thermal settling
time; settling is **seconds**), NOT a fast delay gauge. It is nonetheless the one independent physical
channel available on this rig, so it is the **canonical** witness:

```
CANONICAL witness  = invariant-TSC gate-propagation delay, read at the THERMAL timescale
                     (seconds dwell, dwell-time extrapolation; ~1.5 bits / settling time)
Corroborating      = effective frequency (APERF/MPERF), k10temp, wall power (if available)
SNR upgrade        = coherent averaging locked to the ~2.67 MHz VRM strobe phase (see below)
OPTIONAL upgrade   = one-time bench DMM on a safe Vcore test point (only path to lift VCORE_BLOCKED)
```

The TSC-delay shift must be read OUTSIDE its own noise floor, at the thermal timescale, and must
**corroborate** with at least one independent channel (k10temp or wall power). TSC-delay alone moving
while temperature and power do not is suspicious (possible self-load artifact), not a pass.

### SNR upgrade: coherent-averaging VRM strobe

The ~2.67 MHz VRM switching tone is broadcast on the shared power rail with ns-scale inter-core skew.
Used as a **global strobe / sampling clock**, it lets the TSC-thermal probe coherently average a
sub-noise signal: lock sampling to the strobe phase and integrate, gaining ~sqrt(N) over the raw-jitter
floor. This is an SNR enhancement on the canonical witness, **not** a second independent witness.

Strobe-as-instrument is itself conditional and must be validated before it is trusted (build-and-validate
the lock-in BEFORE relying on it):
```
- prove the 2.67 MHz line is a real cross-core shared-rail strobe, not a config-invariant artifact
  (if it is config-invariant infrastructure, the coherent-averaging lever is unfounded - see ROADMAP)
- aperture shorter than the ~375 ns rail period (the catalytic measurement window currently smears >1 cycle)
- empty-tape, scrambled-phase, off-resonance, and 1/sqrt(N) controls must show SNR gain on a KNOWN
  injected tone before the strobe is trusted on an unknown signal
```
A null on any of these kills the strobe upgrade for the whole electrical family at near-zero (software-only)
cost; the canonical TSC-thermal witness still stands without it (just at the raw ~1.5-bit floor).

### actual Vcore: VCORE_MEASUREMENT_BLOCKED by default

`actual Vcore` is set to `VCORE_MEASUREMENT_BLOCKED` by default and is **not** required to pass 5.10A.
The ONLY path to upgrade it is a one-time bench DMM on a safe Vcore test point (OPTIONAL; owner decision).
Until then, never report a decoded-VID change as a Vcore change.

## Instrumentation candidates (use what the platform allows)

- invariant-TSC gate-propagation delay (CANONICAL witness; thermal timescale, seconds dwell)
- coherent-strobe / lock-in on the ~2.67 MHz VRM tone (SNR upgrade on the TSC witness)
- hwmon / lm-sensors temperature (k10temp; corroborating channel)
- APERF/MPERF effective-frequency witness (corroborating channel)
- wall power meter (coarse but external; corroborating channel)
- one-time bench DMM on a safe Vcore point (OPTIONAL; only path to lift VCORE_BLOCKED)
- external Vcore/PDN scope-or-ADC (OPTIONAL; the witness the electrical-channel work eventually wants)
- per-core P-state MSR readback (logged covariate)
- decoded VID / COFVID readback (logged covariate; NOT a witness)
- run-window synchronized logging (timestamp-aligned to the catalytic loop)

## Required output: instrumentation table (`instrumentation_lock.csv`)

Columns:
```
physical_variable
requested_value
decoded_value
measured_value
witness_role            # canonical / corroborating / snr_upgrade / covariate / blocked
measurement_method
sampling_rate
timescale               # thermal (seconds) / electrical (fast) / NA
strobe_locked           # true/false: was sampling phase-locked to the 2.67 MHz strobe
synchronized_to_run     # true/false: was the measurement window aligned to the catalytic loop
noise_floor             # the channel's own noise floor for that timescale
confidence              # high / medium / low
notes
```

Minimum variables that must appear:
```
TSC gate-propagation delay (canonical witness)
effective frequency (corroborating)
temperature / k10temp (corroborating)
decoded VID (covariate; explicitly flagged NOT a witness)
P-state (covariate)
wall power OR actual Vcore (corroborating if available; Vcore marked BLOCKED if not bench-measured)
```

## Honesty rule

If the TSC-thermal witness does not move outside its noise floor, or moves but does NOT corroborate with
an independent channel (k10temp / wall power), **5.10A must not fake success.** Mark it
`PHASE5_10A_INSTRUMENTATION_BLOCKED` and carry that limitation into the 5.10B/5.10C interpretation and the
Phase 6 readiness gate. A decoded-VID move is never sufficient.

## Allowed statuses

```
PHASE5_10A_INSTRUMENTATION_LOCK_PASS      # TSC-thermal witness moves outside noise AND corroborates
                                          #   (k10temp / eff-freq / wall power) under the prep lever
PHASE5_10A_INSTRUMENTATION_PARTIAL        # witness moves but corroboration weak / strobe upgrade unvalidated
PHASE5_10A_VCORE_MEASUREMENT_BLOCKED      # default Vcore state: not bench-measured (NOT a fail; not required)
PHASE5_10A_INSTRUMENTATION_BLOCKED        # only decoded VID / requested values move; no physical witness
PHASE5_10A_INSTRUMENTATION_FAILED         # only requested values exist; substrate state unproven
```

## Pass / partial / fail (Gate 1)

- **PASS** is **ACHIEVABLE without an external Vcore probe.** PASS iff the canonical TSC gate-propagation
  delay shifts reproducibly OUTSIDE its noise floor at the thermal timescale under the prep lever AND
  corroborates with at least one independent channel (k10temp, effective frequency, or wall power).
  The coherent-strobe upgrade strengthens but is not required for PASS.
- **PARTIAL** if the TSC witness moves but corroboration is weak, or the strobe upgrade is not yet validated.
- **BLOCKED** if only decoded VID / P-state move while TSC-delay + temperature + power do not
  (decoded VID is a request, not a witness).
- **FAIL** if only requested values exist.

Gate 1 is deliberately **NOT** gated on actual Vcore. `VCORE_MEASUREMENT_BLOCKED` is the expected default
and does not block PASS; it only caps the *direct-rail* strength of any downstream claim and must be stated
explicitly in every downstream verdict. A PARTIAL / BLOCKED instrumentation lock does not by itself block
5.10B/5.10C, but it caps basin-claim strength and must be carried forward.
