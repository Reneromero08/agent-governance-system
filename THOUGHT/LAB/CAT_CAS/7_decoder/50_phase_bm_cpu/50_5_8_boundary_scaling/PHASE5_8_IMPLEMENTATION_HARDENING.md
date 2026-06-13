# Phase 5.8 Implementation Hardening

Date: 2026-06-10

## Verdict Correction

Current hardened verdict after targeted rerun:

`EXP50_PHASE5_8_AREA_LAW_CONFIRMED`

The prior `EXP50_PHASE5_8_AREA_LAW_CONFIRMED` label was temporarily too strong because the artifact audit was PARTIAL. The result supported a real silicon boundary and volume-beating boundary scaling, but strict area-law confirmation requires:

- Gate 5 frequency deformation proof present.
- Gate 8 area-law wins tracked separately from log-law wins.
- Gate 9 artifact audit clean PASS.

The targeted P0-locked rerun cleared the named cache artifact:

| Tape | CACHE/NONE thickness ratio |
|------|----------------------------|
| 1024 | 1.794370 |
| 4096 | 1.232016 |

Artifact report: `phase5_8/results/freq_locked_cache_probe/PHASE5_8_FREQ_LOCKED_CACHE_ARTIFACT_PROBE.md`.

## Implementation Fixes

- `aggregate_phase5_8.py`
  - Split strict area wins from area-or-log wins.
  - Prevented PARTIAL Gate 9 from promoting to strict `AREA_LAW_CONFIRMED`.
  - Added explicit gate fields for area wins, log wins, area-or-log wins, strict area pass, control distinctness, and cache anomaly.

- `phase5_8_boundary_rdtsc.c`
  - Count worker start failures as real run failures.
  - Zero worker state before starting threads.
  - Harden read-only control loop against non-8-byte tape sizes.
  - Free partial allocations on allocation failure.
  - Include worker start failures in telemetry.

- `run_phase5_8.sh`
  - Analyzer and aggregator failures now make the orchestration fail.
  - Failed runs still continue within the matrix, but the full script exits nonzero if any run or postprocess step fails.

## Remaining Evidence Boundary

The original T1024-T4096 cache/frequency drift anomaly is resolved for P0-locked interleaved NONE/CACHE ordering. The next evidence boundary is not 5.8 artifact cleanup; it is the 5.9 timing-CV carrier thread.
