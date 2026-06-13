# PHASE 5.8 — Design Document

## Bare-Metal Holographic Boundary Probe
**Exp50 Phase 5.8**
**Target:** AMD Phenom II X6 1090T (K10, 45nm SOI)
**Date:** 2026-06-09

## 1. Architecture

### 1.1 C Harness (`phase5_8_boundary_rdtsc.c`)
Main measurement harness. Features:
- Serialized RDTSC/RDTSCP timing (CPUID barrier before/after)
- Reversible catalytic tape operations (XOR forward/reverse)
- Aligned, memory-locked buffers (posix_memalign + mlock)
- CPU affinity pinning via sched_setaffinity
- Buffered CSV output (no printf inside timing loop)
- 8 control modes: empty, nop, irreversible, readonly, shuffled, synthetic, trial-order, migration
- Randomized trial order (Fisher-Yates shuffle)
- FNV-1a 64-bit checksum verification

### 1.2 Worker Module (`phase5_8_workers.c`)
Deterministic load workers:
- **Cache hammer:** 20MB aligned buffer, XOR/rotate stride through memory
- **Integer churn:** LCG/xor/rotate/multiply loop, no memory pressure
- **Mixed:** Half cache hammer, half integer churn
- **Thermal:** Long-running integer churn (optional)
- Atomic stop flag, per-worker core affinity via pthread attributes

### 1.3 Analyzer (`analyze_phase5_8.py`)
Python post-processing:
- Windowed boundary feature extraction (64-1024 sample windows)
- Intrinsic geometry metrics (radius, PCA, effective dimension)
- Spectral power analysis (low/mid/high frequency bands)
- Area-law scaling analysis
- Digital-to-silicon transition analysis
- Verdict gate audit

## 2. Measurement Protocol

### 2.1 Timing Primitive
```
CPUID (serialize)
RDTSC (start)
<measured operation>
RDTSCP (end, serialized)
CPUID (serialize)
```

Overhead measured via empty-body loop, recorded but not silently subtracted.

### 2.2 Catalytic Operation
```
T' = T XOR K   (forward)
T'' = T' XOR K  (reverse, self-inverse)
Invariant: T'' == T
```

FNV-1a 64-bit checksum before and after each trial.

### 2.3 Data Flow
```
C Harness → raw_cycles.csv → Python Analyzer → window_features.csv
                                            → geometry_stats.csv
                                            → projection_stats.csv
                                            → verdict_gate_audit.csv
```

## 3. Controls

| Control | Purpose |
|---------|---------|
| Empty timing | RDTSC overhead baseline |
| NOP loop | Non-catalytic execution baseline |
| Irreversible write | Destructive overwrite comparison |
| Read-only | Memory read without XOR |
| Shuffled labels | Test load vs random structure |
| Synthetic cloud | Statistical null (not primary result) |
| Trial order audit | Drift vs condition predictor |
| Migration audit | CPU core change detection |

## 4. Verdict Gates

| Gate | Criteria |
|------|----------|
| 1. Raw Silicon Timing | RDTSC works, affinity holds, no migration |
| 2. Restoration Survival | All checksums match, bits_erased = 0 |
| 3. Intrinsic Boundary Geometry | No synthetic null, measured cloud built |
| 4. Load Boundary Deformation | worker_count changes geometry |
| 5. Frequency Boundary Deformation | DID/frequency changes geometry |
| 6. Voltage Boundary Deformation | VID changes geometry (deferred if unavailable) |
| 7. Digital-to-Silicon Transition | Boundary persists after Python/OS removal |
| 8. Area-Law Scaling | Area/log beats volume on 2+ metrics |
| 9. Artifact Audit | Trial order/migration don't explain result |

## 5. Operating Points

### A. Digital baseline
Nominal frequency, nominal voltage

### B. Frequency-detuned sweep
3200, 1600, 800, 400, 200, 100 MHz (DID-divisible states)

### C. Runtime VID sweep (if available)
Requires accessible VID channel; K10 platform has no accessible runtime VID below 1.225V floor

### D. Sub-threshold / analog sweep
Only if validated hardware route exists; not yet available

## 6. Non-Negotiable Constraints

- No synthetic Gaussian null defines the main result
- No printf/allocation inside timing loop
- No hash computation inside timing loop
- Do not fake sub-threshold mode
- Do not label a run sub-threshold unless measured voltage confirms it
- Do not claim physical holography, AdS/CFT, or quantum coherence
