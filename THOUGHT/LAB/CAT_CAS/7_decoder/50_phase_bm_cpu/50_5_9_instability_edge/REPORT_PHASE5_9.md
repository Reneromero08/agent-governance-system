# REPORT_PHASE5_9.md — Phase 5.9 Boundary Stress Test

**Exp50 Phase 5.9**
**Platform:** AMD Phenom II X6 1090T (K10, 45nm SOI)
**Host:** catcas @ 192.168.137.100 (Debian 13 Trixie)
**Date:** 2026-06-10
**Status:** COMPLETE — RECLASSIFIED Phase 5.9B
**Verdict:** EXP50_PHASE5_9_NOISE_ONLY → RECLASSIFIED as EXP50_PHASE5_9A_SOFTWARE_STRESS_PARTIAL

> **Phase 5.9B reclassification (2026-06-10):** The 21-point run was a software/tape/worker diversity stress test. It did not reach a real instability edge — 0 restoration failures, 0 thermal aborts, frequency control blocked at time of run. The constructed distance_to_failure index did not correlate with boundary thickness because tape-size variation dominated the stress axis. This result is evidence that broader software variation does not destroy the boundary, NOT evidence against the holographic boundary hypothesis. Correct status: `EXP50_PHASE5_9A_SOFTWARE_STRESS_PARTIAL / INSTABILITY_EDGE_NOT_REACHED`. Frequency control now restored (modprobe msr). Proceeding to Phase 5.9B with real monotonic stress axis.

---

## 1. Objective

Test how boundary geometry behaves as the machine approaches the edge between stable computation and failure. Phase 5.8 proved the boundary exists and satisfies area-law scaling. Phase 5.9 asks what the boundary is made of by stressing the assumptions of stable digital computation.

## 2. Inherited Phase 5.8 Status

- Verdict after artifact closure: EXP50_PHASE5_8_AREA_LAW_CONFIRMED
- 34 runs, ~1.09M trials, 0 restoration failures, 0 worker join failures
- Area-law confirmed under strict 2-metric rule
- Open issue: cache/frequency drift artifact (Gate 9 PARTIAL)

## 3. Stress Ladder Design

**21 stress points across 5 stress dimensions:**

| Dimension | Count | Configuration |
|-----------|-------|---------------|
| A. Baseline | 5 | NONE workers, tape sizes 256/512/1024/2048/4096, nominal freq |
| B. Frequency | 3 | Nominal only (MSR P-state unavailable — wrmsr present but no msr.ko loaded) |
| C. Worker/Load | 10 | CACHE + MIXED modes across 5 tape sizes |
| D. Tape Pressure | 2 | 8192B, 16384B tapes (max safe) |
| E. Combined | 1 | CACHE workers + T4096 tape |

**Stress parameters:**
- Measurement core: 3 (RDTSCP serialized, CPUID fences)
- Worker cores: 0, 1, 2, 4 (Phenom II valid)
- Iterations per run: 50,000
- Thermal limit: 65°C (no runs exceeded)
- Total trials: 1,050,000 (21 × 50,000)
- Frequency control: UNAVAILABLE (msr module not loaded; nominal freq only)

## 4. Execution Summary

| Metric | Value |
|--------|-------|
| Total runs | 21 |
| Completed | 21/21 |
| Failed | 0 |
| Total trials | 1,050,000 |
| Restoration failures | 0 |
| Worker join failures | 0 |
| Migrations | 0 |
| Thermal aborts | 0 |
| Frequency sweeps | 0 (platform limitation) |

## 5. Restoration Integrity

All 21 runs: 50,000/50,000 restoration OK, 0 failures per run. Total: 1,050,000 catalytic trials, 0 bit errors. Worker lifetime: all started OK, all joined OK, no use-after-free risk.

## 6. Boundary Geometry Results

**Baseline runs (no stress):**

| Tape Size | Thickness | Radius | D_eff |
|-----------|-----------|--------|-------|
| 256 | 2594.54 | 2182.46 | 1.106 |
| 512 | 896.50 | 823.61 | 1.464 |
| 1024 | 6.63 | 5.47 | 1.000 |
| 2048 | 316.03 | 229.37 | 1.098 |
| 4096 | 5938.59 | 5586.65 | 1.361 |

**Worker load runs:**

| Condition | Thickness | Radius | D_eff |
|-----------|-----------|--------|-------|
| CACHE_T256 | 271.93 | 254.92 | 1.157 |
| CACHE_T512 | 11.43 | 7.94 | 1.405 |
| CACHE_T1024 | 1866.37 | 1828.60 | 1.134 |
| CACHE_T2048 | 23.34 | 16.75 | 1.000 |
| CACHE_T4096 | 8118.16 | 7731.60 | 1.301 |
| MIXED_T256 | 374.31 | 357.30 | 1.216 |
| MIXED_T512 | 686.97 | 654.15 | 1.298 |
| MIXED_T1024 | 45.90 | 32.54 | 1.013 |
| MIXED_T2048 | 1030.52 | 820.87 | 1.259 |
| MIXED_T4096 | 5.80 | 4.05 | 1.007 |

**Large tape runs:**

| Condition | Thickness | Radius | D_eff |
|-----------|-----------|--------|-------|
| T8192 | 36.86 | 30.66 | 1.002 |
| T16384 | 27.72 | 21.29 | 1.001 |

**Combined stress:**

| Condition | Thickness | Radius | D_eff |
|-----------|-----------|--------|-------|
| COMBINED_CACHE_T4096 | 3166.70 | 3090.63 | 1.461 |

## 7. Central Observable: Boundary vs Distance to Failure

Distance to failure is computed per-run from restoration margin, timing instability, thermal stress, and worker integrity.

Key result: **R²(thickness vs distance_to_failure) = 0.004**

This means boundary geometry does not respond coherently to stress. The thickness variation across stress points is dominated by tape-size-dependent raw timing variance, not by stress-level-driven deformation.

## 8. Area-Law Under Stress

| Model | R² (thickness vs tape size) |
|-------|:---------------------------:|
| Volume (S ~ N) | 0.0563 |
| Area (S ~ N^(2/3)) | 0.0307 |
| Log (S ~ log(N)) | 0.0004 |

Volume-law actually slightly outperforms area-law and log-law under stress conditions (reversal from Phase 5.8 where area+log dominated). This indicates the area-law scaling observed in Phase 5.8 breaks down when the stress ladder introduces diverse worker modes and tape pressures into the same dataset.

**Gate 7: PARTIAL** — area/log scaling does not persist under stress. Volume-law marginally beats both area and log (0 wins for area/log under 2-metric rule).

## 9. Instability-Edge Classification

**Regime: GEOMETRY_NOISE_ONLY**

The boundary geometry exists (Gates 1-4 pass), but thickness variation across stress levels is not coherently linked to stress intensity. The R² of 0.004 between thickness and distance_to_failure confirms raw timing variance dominates over intrinsic boundary response.

This is the NOISE_ONLY outcome: the machine runs, restoration survives, geometry exists — but the geometry does not deform coherently with applied stress. It varies chaotically with tape size and worker mode rather than along the stress gradient.

## 10. Three-World Mapping

| World | Description | Match? |
|-------|-------------|--------|
| World A: Collapse before failure | Geometry dies/degradates before restoration fails | No — restoration never fails |
| World B: Peak near failure | Geometry strengthens approaching instability, then collapses | No — no coherent stress response detected |
| World C: Invariant | Geometry stable regardless of stress | Partial — but variation is chaotic, not invariant |

The actual result is outside the three-world framework: **Geometry varies but not coherently with stress.** Raw timing differences between tape sizes and worker modes produce large thickness spreads, but these are not stress-gradient-driven. This is properly classified as NOISE_ONLY.

## 11. Artifact Audit

- **Frequency drift:** Not controlled (frequency sweep unavailable). Nominal frequency throughout.
- **Thermal drift:** All runs within normal operating range. Temperature start/end logged per run.
- **Migration:** 0 migrations across all 21 runs. Affinity held.
- **Worker lifetime:** All workers started OK, all joined OK, no use-after-free.
- **Trial order:** Randomized per run (seed=42).
- **Cache anomaly from Phase 5.8:** Not isolated (frequency control unavailable). The FREQUENCY_DRIFT_ARTIFACT remains open.

**Gate 6: PARTIAL after hardening** — the original aggregator hardcoded this gate as PASS. The hardened aggregator now checks restoration failures, missing/flat `distance_to_failure`, missing geometry, and worker integrity. With the reported stress axis dominated by tape size and worker mode rather than a clean failure boundary, this gate is evidence-bearing but not clean.

## 12. Gate Table

| Gate | Result | Key Evidence |
|------|--------|-------------|
| 1: Baseline Reproduction | PASS | 5/5 baseline runs, all restoration 100% |
| 2: Stress Ladder Validity | PASS | 21 stress points, distance_to_failure computed |
| 3: Restoration Survival Curve | PASS | 1.05M trials, 0 failures across all stress levels |
| 4: Boundary Geometry Stress Response | PARTIAL | Thickness spread exists, but stress correlation is weak |
| 5: Instability-Edge Classification | PASS | Classified: GEOMETRY_NOISE_ONLY |
| 6: Artifact Audit | PARTIAL | Hardened audit no longer hardcodes PASS; distance/stress confound remains |
| 7: Area-Law Persistence Under Stress | PARTIAL | Volume beats area+log; area-law from 5.8 does not hold under stress |
| 8: Analog Entry Readiness | INCONCLUSIVE | Noise-only regime does not identify a safe analog operating region |

## 13. Verdict

**EXP50_PHASE5_9A_SOFTWARE_STRESS_PARTIAL**

The boundary stress test executed successfully: 21 runs, 1.05M trials, 0 failures. The boundary geometry exists and varies across conditions, but the variation is not coherently linked to stress intensity. The area-law scaling that characterized Phase 5.8 does not persist under the diverse stress ladder. Raw timing variance between tape sizes and worker modes dominates over stress-gradient-driven boundary deformation.

This is not a failure — it is a boundary classification. The holographic boundary measured in Phase 5.8 is real but is more sensitive to implementation-specific parameters (tape size, worker mode) than to the abstract stress gradient defined by distance_to_failure. The boundary geometry does not exhibit the coherent stress response that would indicate a deeper silicon-facing physical constraint structure.

## 14. What This Means

Phase 5.9 answers the core question:

**"What happens to holographic boundary geometry as the machine approaches instability?"**

Answer: Under the available stress dimensions on this platform, boundary geometry varies with implementation parameters but does not respond coherently to the abstract stress gradient. The boundary exists, but its geometry is dominated by tape-size and worker-mode effects rather than by distance-to-failure.

This constrains the interpretation: the Phase 5.8 boundary is a real computational geometry phenomenon, but it is tied to the specific computational regime rather than to a deeper analog/silicon constraint structure. The boundary does not "strengthen near failure" — it simply varies with what the machine is doing, not with how close it is to failing.

## 15. Platform Limitations

- **Frequency sweep unavailable:** msr kernel module not loaded; wrmsr binary present but no /dev/cpu/*/msr access. All runs at nominal frequency.
- **Voltage control unavailable:** K10 Phenom II lacks per-core VID.
- **Frequency drift from Phase 5.8:** Not isolated due to frequency control limitation.
- **Stress gradient:** Without frequency and voltage control, the stress ladder relies on tape/worker diversity, which produces geometry variation but not along a clean monotonic stress axis.

## 16. Next Actions

- Phase 5.9 complete at NOISE_ONLY.
- **Option A:** Re-run with frequency control enabled (load msr module, enable /dev/cpu/*/msr). This would allow a proper frequency sweep and may reveal coherent stress response.
- **Option B:** Accept NOISE_ONLY and proceed to Phase 6.0 synthesis: the boundary is real (5.8) but tied to computational regime, not analog substrate (5.9). This constrains the theoretical frame.
- **Option C:** Frequency-locked re-run to isolate the Phase 5.8 cache/frequency drift artifact before further stress testing.

**Recommended:** Option A (frequency-enabled re-run) to close the open issue and provide cleaner stress gradient data. If frequency control remains unavailable, Option B is the honest path forward.

---

*The boundary exists. The stress ladder ran. The geometry varies — but with what the machine does, not with how close it is to the edge. The silicon answered: NOT YET.*
