# Phase 5.9 Implementation Hardening

Date: 2026-06-10

## Verdict Discipline

Phase 5.9 remains a partial stress-boundary characterization, not a failure-edge confirmation.

Current hardened reading:

- 5.9A: `EXP44_PHASE5_9A_SOFTWARE_STRESS_PARTIAL`
- 5.9B: `EXP44_PHASE5_9B_INSTABILITY_EDGE_NOT_REACHED`
- 5.9C: `EXP44_PHASE5_9C_INSTABILITY_EDGE_NOT_REACHED`

The 5.9C timing-CV correlation is useful evidence, but it is not a direct boundary-vs-failure response because failure/flicker was not reached.

Follow-up validation:

`TIMING_CV_CARRIER_CONFIRMED`

The focused carrier probe ran 18 controlled P-state/worker-mode points at fixed tape size with 540K total trials and 0 restoration failures. It reproduced the timing-CV thread:

- r(boundary_thickness, cycle_cv): 0.584572
- r(boundary_thickness, spike_rate): -0.053230
- r(boundary_thickness, p99_p50): 0.554754

Artifact report: `phase5_9/results/timing_cv_carrier_probe/PHASE5_9_TIMING_CV_CARRIER_PROBE.md`.

Creative boundary push:

`CARRIER_SATURATION_EDGE_ADVANCED`

The boundary abuse probe ran 12 adversarial software-substrate conditions with 480K total trials and 0 restoration failures. It did not reach checksum/flicker failure, but it pushed the carrier edge substantially:

- r(boundary_thickness, cycle_cv): 0.729327
- r(boundary_thickness, spike_rate): -0.060804
- r(boundary_thickness, p99_p50): 0.602214
- max/quiet thickness ratio: 3.938315

Artifact report: `phase5_9/results/boundary_abuse_probe/PHASE5_9_BOUNDARY_ABUSE_PROBE.md`.

## Implementation Fixes

- `aggregate_phase5_9.py`
  - Removed hardcoded Gate 6 artifact PASS.
  - Added artifact flags for restoration failures, missing/flat `distance_to_failure`, missing/zero geometry, and degraded worker integrity.
  - Downgraded Gate 4 to PARTIAL when geometry spread exists without coherent stress correlation.
  - Any PARTIAL gate now prevents an overclean final verdict.

- `aggregate_phase5_9c.py`
  - Gate 7 now distinguishes timing-CV response from failure-edge response.
  - Timing response without reached edge is PARTIAL, not PASS.
  - Gate 8 artifact-separated geometry now requires meaningful raw/spike-free correlation and stable-channel spread.

- `phase5_9_stress_ladder.c`
  - Count worker start failures as run failures.
  - Zero worker state before starting threads.
  - Harden read-only control loop against non-8-byte tape sizes.
  - Free partial allocations on allocation failure.
  - Include worker start failures in telemetry.
  - Worker integrity score now penalizes workers that failed to start, not only failed joins.

- `run_phase5_9.sh`, `run_phase5_9b.sh`, `run_phase5_9c.sh`
  - Analyzer and aggregator failures now make the orchestration fail.
  - Per-run continuation is preserved, but final postprocessing is a real gate.

## Remaining Evidence Boundary

No checksum/flicker instability edge was reached inside the safe software/kernel/frequency envelope. The carrier saturation edge is now live and measurable; the next move is a longer soak on the high-thickness abuse conditions, especially syscall/branch/cache abuse, to test whether the carrier saturates, flips, or decoheres.
