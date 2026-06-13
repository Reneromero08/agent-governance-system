# Phase 5.8 — Bare-Metal Holographic Boundary Probe

**Status:** COMPLETE (2026-06-09), hardened review 2026-06-10
**Lead:** CAT_CAS Exp50 Implementation Agent + DeepSeek
**Target:** catcas @ 192.168.137.100 — AMD Phenom II X6 1090T (K10, 45nm SOI)
**Verdict:** EXP50_PHASE5_8_AREA_LAW_CONFIRMED
**Hardening note:** Strict `AREA_LAW_CONFIRMED` was restored after a P0-locked cache artifact rerun cleared the large-tape cache anomaly. CACHE/NONE thickness ratios: T1024=1.794370, T4096=1.232016.
**Spec:** Bare Metal CPU Entropy.md, Entropy_2.md, Entropy_3.md, Entropy_4.md (Shizzle Obsidian vault)

## Objective

Phase 5.8 moves the entropic boundary probe from Python/OS-level timing into bare-metal C/RDTSC timing on the AMD Phenom II platform. The goal is to test whether the holographic carrier boundary persists when we move down from Python timing into silicon-facing cycle timing.

## Prior Chain

- EXP 42.28: Load-induced timing variance (contaminated by synthetic Gaussian null)
- EXP 42.29: Intrinsic execution-boundary cloud (hardware load changes intrinsic boundary geometry, catalytic restoration survives)
- Phase 5.7: Entropic boundary confirmed at OS/Python level (`PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED`)

## Hypothesis

When Python and scheduler overhead are removed, the boundary does not disappear. Instead, the boundary becomes visible as cycle-level timing geometry:

- RDTSC/RDTSCP jitter
- Cache-line contention geometry
- Frequency-detuning deformation
- Voltage-state deformation if accessible
- Thermal/cycle variance structure
- Spectral deformation of execution timing
- Intrinsic boundary-cloud dimensionality changes

## The Bridge

```
digital cache boundary
→ C/RDTSC timing boundary
→ frequency-detuned silicon boundary
→ voltage-sensitive analog boundary
```

## Non-Negotiable Rules

1. Do not rewrite the premise as metaphor.
2. Do not reduce the phase to "just benchmarking."
3. Do not replace the holographic-boundary frame with generic performance analysis.
4. Preserve the interpretation; harden the experiment; let the silicon decide.
5. Do not fake a pass. Do not declare the boundary confirmed unless the hard gates pass.

## Experimental Design

- C harness with RDTSC/RDTSCP serialized timing on measurement core (default Core 3)
- Catalytic tape operation: T' = T XOR K, reverse: T'' = T' XOR K, invariant: T'' == T
- Tape sizes: 256, 512, 4096, 32768 bytes (64-byte aligned, mlock'd, pre-touched)
- Worker load modes: cache hammer (20MB+ L3-spilling), integer churn, mixed pressure, thermal (optional)
- CPU affinity via sched_setaffinity, no migration during measurement
- Operating-point sweep: frequency detuning (100-3200 MHz via DID), VID-labeled
- FNV-1a 64-bit checksum for restoration verification
- Buffered CSV output — no printf/allocation inside timing loop
- Python analyzer: windowed boundary features (64-1024 sample windows), intrinsic geometry, area-law scaling

## Verdict Gates

| Gate | Name | Description |
|------|------|-------------|
| Gate 1 | Raw Silicon Timing Validity | RDTSC/RDTSCP works, affinity holds, no migration |
| Gate 2 | Catalytic Restoration Survival | All hash/checksum matches, restore_failures=0, logical_bits_erased=0 |
| Gate 3 | Intrinsic Boundary Geometry | No synthetic null defines result; boundary cloud from measured data |
| Gate 4 | Load Boundary Deformation | Worker count changes intrinsic geometry beyond raw variance |
| Gate 5 | Frequency/Detuning Deformation | DID operating point changes boundary geometry at matched worker_count |
| Gate 6 | Voltage Boundary Deformation | VID changes geometry (DEFERRED if voltage data unavailable) |
| Gate 7 | Digital-to-Silicon Transition | Boundary persists in C/RDTSC after Python/OS removal; operating-point geometry measurable |
| Gate 8 | Area-Law Scaling | Boundary metric scaling beats volume-law on >=2 independent metrics; strict area-law wording requires area wins separately from log wins |
| Gate 9 | Scheduler/OS Artifact Audit | Trial order does not explain result; no migration; flatline regimes marked |

## Verdict Labels

- `EXP50_PHASE5_8_SILICON_BOUNDARY_CONFIRMED` — Core silicon-boundary gates pass; Gate 6 may be properly deferred; Gate 9 may be PARTIAL only if explicitly non-fatal
- `EXP50_PHASE5_8_AREA_LAW_CONFIRMED` — All silicon-boundary gates pass, Gate 5 frequency proof exists, Gate 9 is clean, and Gate 8 has independent area-law wins
- `EXP50_PHASE5_8_DIGITAL_TO_SILICON_TRANSITION_CONFIRMED` — Boundary persists with reduced workers, detuning produces structured deformation
- `EXP50_PHASE5_8_PARTIAL_BOUNDARY_DEFORMATION` — Geometry changes under load but operating-point or area-law evidence incomplete
- `EXP50_PHASE5_8_NOISE_ONLY` — Raw variance changes but no coherent intrinsic geometry deformation
- `EXP50_PHASE5_8_ARTIFACT_DOMINANT` — Migration, scheduler effects, trial order, or timing overhead explain result
- `EXP50_PHASE5_8_BLOCKED_BY_PLATFORM` — C/RDTSC cannot run reliably, affinity cannot be fixed
- `EXP50_PHASE5_8_BOUNDARY_REJECTED` — No intrinsic boundary geometry persists in C/RDTSC

## Execution Summary (2026-06-09 — Phase 5.8R hardened re-run)

**34 total runs:** 15 condition matrix (5 tape sizes x 3 worker modes) + 4 controls + 15 frequency sweep (5 P-states x 3 tape sizes)

| Category | Count | Details |
|----------|-------|---------|
| Condition matrix | 15 | NONE, CACHE, MIXED at 256/512/1024/2048/4096 bytes |
| Controls | 4 | EMPTY, NOP, IRREVERSIBLE, READONLY |
| Frequency sweep | 15 | 800/1600/2400/3200/3600 MHz x 256/1024/4096 bytes |
| **Total** | **34** | **~1,090,000 catalytic trials, 0 restoration failures, 0 worker join failures** |

**Gate results:**

| Gate | Result | Key evidence |
|------|--------|-------------|
| 1: Raw Silicon Timing | PASS | 34/34 runs, RDTSCP serialized, affinity held |
| 2: Restoration Survival | PASS | ~1.09M trials, 0 failures |
| 3: Intrinsic Boundary Geometry | PASS | True eigendecomposition, 390 windows/run |
| 4: Load Boundary Deformation | PASS | Cache changes geometry (deformation ratios 0.68-1.89x) |
| 5: Frequency Deformation | PASS | 15-run MSR P-state sweep, geometry varies with frequency |
| 6: Voltage Deformation | DEFERRED_NOT_FAILED | K10 lacks per-core VID |
| 7: Digital-to-Silicon Transition | PASS | Catalytic survives all 34 conditions |
| 8: Area-Law Scaling | PASS | Area+log beats volume on 4/4 metrics; strict area/log split now recorded |
| 9: Artifact Audit | PASS | P0-locked T1024/T4096 rerun cleared cache anomaly |

See REPORT_PHASE5_8_FINAL.md for the full 19-section consolidated report.

## Files

| File | Location | Purpose | Status |
|------|----------|---------|--------|
| `phase5_8_common.h` | `50_5_8_boundary_scaling/src/` | Shared definitions, RDTSC primitives | FINAL |
| `phase5_8_workers.h` | `50_5_8_boundary_scaling/src/` | Worker thread interface | FINAL |
| `phase5_8_workers.c` | `50_5_8_boundary_scaling/src/` | Worker implementations (hardened) | FINAL |
| `phase5_8_boundary_rdtsc.c` | `50_5_8_boundary_scaling/src/` | Main C harness | FINAL |
| `analyze_phase5_8.py` | `50_5_8_boundary_scaling/src/` | Python analyzer (hardened) | FINAL |
| `Makefile` | `50_5_8_boundary_scaling/src/` | Build system | FINAL |
| `run_phase5_8.sh` | `50_5_8_boundary_scaling/src/` | Orchestration (hardened) | FINAL |
| `README.md` | `50_5_8_boundary_scaling/` | This file | FINAL |
| `REPORT_PHASE5_8_FINAL.md` | `50_5_8_boundary_scaling/` | Consolidated final report (19 sections, all gates) | FINAL |
| `PHASE5_8_DESIGN.md` | `50_5_8_boundary_scaling/` | Design document | FINAL |
| `PHASE5_8_SUMMARY.md` | `50_5_8_boundary_scaling/` | Full experimental report | FINAL |
| `PHASE5_8_TELEMETRY.md` | `50_5_8_boundary_scaling/` | Run telemetry data | FINAL |
| Runtime CSVs | `50_5_8_boundary_scaling/results/` | Generated on Phenom II | REMOTE |

## Next Actions After Phase 5.8

- COMPLETE: Verdict `EXP50_PHASE5_8_AREA_LAW_CONFIRMED`
- Proceed to Phase 5.9 — Analog Silicon Boundary Entry (controlled VID/VRM/firmware voltage sweep)
- Gate 9 closure artifact: `50_5_8_boundary_scaling/results/freq_locked_cache_probe/PHASE5_8_FREQ_LOCKED_CACHE_ARTIFACT_PROBE.md`

## Related Artifacts

- Roadmap: `../ROADMAP.md` (Phase 5.8 section at lines 1267-1332)
- Phase 5.7: `../50_5_7_entropic_boundary/` (entropic boundary at OS/Python level)
- Phase 5.6: `../50_5_6_polytope_geometry/` (polytope geometry confirmed)
- Spec source: `/mnt/c/Users/rene_/Documents/Shizzle Obsidian/Shizzle/AGI/AGS/WIP/Todo/Bare Metal CPU/Bare Metal CPU Entropy.md`
