# EXP50 PHASE 5.10 - BOUNDARY STATE PREPARATION

**Subtitle:** Instrumented Voltage / Prelude Basin Control
**Status:** SPEC (design only; hardware run pending on the Phenom)
**Platform:** AMD Phenom II X6 1090T (K10, 45nm SOI), bare-metal, catcas host
**Claim ceiling:** L4-5. No faked instrumentation, no post-hoc thresholds, no crash-as-success.

---

## 0. Why 5.10 exists (the split)

Phase 5.x asked: **can software stress DEFORM the boundary?** Answer (5.8/5.9A/5.9B/5.9C):
the catalytic boundary is robust - area-law confirmed, 0 restoration failures across millions of
trials, the true instability edge was not reached inside the safe controllable envelope, and the one
new signal was that boundary thickness tracks sustained timing CV (a carrier-level property, not a
transient glitch).

The next meaningful variable is therefore **not more software pressure**. It is **physical state
preparation**. Phase 5.10 opens that regime with a hard conceptual split that must not be conflated:

| Phase | Question |
|---|---|
| **5.10 (this spec)** | Can the silicon **prepare** reproducible boundary states (carrier basins)? |
| **Phase 6** | Can a **prepared** boundary state **carry/select** the fixed-point invariant? |

Failure modes differ, which is the whole reason for the split:
- If **5.10 fails**, the substrate cannot yet prepare reliable basins (a *preparation* failure).
- If **5.10 passes but Phase 6 fails**, basins exist but do not carry the fixed point (a *crossing* failure).
- If **5.10 is skipped**, Phase 6 is **uninterpretable** - a null result could be either failure, and a
  positive result could be a basin artifact.

**HARD RULE: Phase 6 MUST NOT RUN until 5.10C passes.** (Re-gated in `phase6/SPEC_PHASE6_FIXED_POINT_SUBSTRATE.md`.)

## 1. CAT_CAS frame (use these terms directly)

The catalytic tape is the **holographic boundary**. The restoration loop (borrow dirty tape, couple,
relax, uncompute, verify SHA in==out) is the **closed boundary condition**. A **carrier basin** is a
reproducible physical state the boundary settles into. **Boundary state preparation** is the bridge
from 5.x (deform/observe the boundary) to Phase 6 (use a prepared basin to test fixed-point selection).
This is not generic CPU benchmarking; it is preparing the **substrate** itself.

## 2. Definition of success (and non-success)

Success is **NOT**: CPU crash; lower decoded VID; an arbitrary voltage request; a raw timing change;
a one-off basin event.

Success **IS**:
```
prelude / physical state
  -> reproducible basin transition probabilities
  -> tape restores bit-for-bit
  -> controls fail to reproduce the selection
  -> the actual physical variable is measured (or honestly marked partial/blocked)
```

## 3. The three subphases (see the per-subphase specs)

- **5.10A INSTRUMENTATION LOCK** (`PHASE5_10A_INSTRUMENTATION_LOCK.md`)
  Which physical variables are *actually* changing? Establish external/independent witnesses.
  Decoded VID in an MSR is only a *request/definition* - prove the silicon physically received the
  state change during the catalytic run, or mark it blocked. No faked Vcore.
- **5.10B VOLTAGE / PRELUDE BASIN SCAN** (`PHASE5_10B_VOLTAGE_PRELUDE_BASIN_SCAN.md`)
  Map (VID / Vcore / prelude / thermal / P-state) -> carrier basin. Define basin classes from
  **frozen calibration thresholds** before any selection data is seen.
- **5.10C BASIN SELECTION** (`PHASE5_10C_BASIN_SELECTION.md`)
  Turn observation into control: can a chosen prelude intentionally bias collapsed / mid / high above
  baseline, surviving randomized repeats and artifact controls, with the tape restoring.

Gates and verdict labels: `PHASE5_10_GATES_AND_VERDICTS.md`.
Handoff to Phase 6: `PHASE5_10_TO_PHASE6_HANDOFF.md`.

## 4. Discipline (binding for all subphases)

- **Restoration is sacred.** Any non-control run that fails to restore the tape bit-for-bit (SHA out
  != SHA in) is VOID and excluded; restoration corruption outside an explicit destructive control is a
  Gate-2 FAIL.
- **Freeze thresholds first.** Basin-class thresholds are calibrated on a calibration set and frozen
  in `basin_thresholds_frozen.json` BEFORE selection runs. Redefining thresholds after seeing the
  selection result is a Gate-3 FAIL.
- **Instrument honestly.** If actual Vcore cannot be witnessed, say so (PARTIAL / BLOCKED) - do not
  treat a decoded-VID change as proof the substrate changed.
- **Controls must fail.** A selection counts only if shuffled-label, no-prelude, and thermal/frequency
  matched controls do NOT reproduce it.
- This is a SPEC. The analysis pipeline (basin classifier, transition-matrix estimator, artifact
  comparisons, frozen-threshold calibrator) is buildable now and is real engineering. The silicon run
  is on the Phenom.

## 5. Future harness files (planned; implement only if repo convention expects stubs)

`run_phase5_10a_instrumentation_lock.sh`, `run_phase5_10b_basin_scan.sh`,
`run_phase5_10c_basin_selection.sh`, `phase5_10_basin_scan.c`, `analyze_phase5_10.py`,
`aggregate_phase5_10.py`. Expected outputs: `instrumentation_lock.csv`, `vcore_measurement_log.csv`,
`frequency_effective_audit.csv`, `phase5_10b_basin_scan.csv`, `phase5_10c_transition_matrix.csv`,
`basin_thresholds_frozen.json`, `artifact_controls_5_10.csv`, `phase5_10_master_verdict.csv`,
`REPORT_PHASE5_10.md`.
