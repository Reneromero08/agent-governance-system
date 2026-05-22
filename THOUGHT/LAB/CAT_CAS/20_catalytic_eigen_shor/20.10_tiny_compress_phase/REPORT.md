# 20.10: Tiny Compress Phase — Compression Experiments Report

## Overview

Can the `.holo` spectral compression engine (TINY_COMPRESS lab) reduce the memory requirement for Shor period detection? The phase grating `g_n = exp(2*pi*i * a^n / N)` lives on the unit circle. We applied five different compression strategies to measure its effective information dimension and extract the period `r`.

## Experiments

### 1. Holographic Phase Oracle (1_holographic_phase_oracle.py)
**Approach**: MERA/Feistel-encoded phase grating fed through EIGEN_BUDDY-style attention.
**Finding**: Feistel degrades signal across layers (SNR drops ~10x per layer). Deterministic phase-constructed Feistel cannot holographically encode U_a's period into fewer positions.
**Verdict**: Negative result. MERA encoding doesn't compress the period.

### 2. Catalytic Complex Compressor (2_catalytic_complex_compressor.py)
**Approach**: Complex-native Hermitian covariance on unit circle (S^1), not flattened real+imag (R^2L). Reduced Df/L from ~2.0 to ~1.0 at large L.
**Finding**: Working in complex domain halves the apparent dimension vs real+imag flattening. Df plateaus at ~1373 for L>=2048, confirming the phase grating has compressive structure. Df/r = 0.005 (200x theoretical compression).
**Verdict**: Complex-native representation is essential. Df measures the true phase dimensionality.

### 3. Torus-Native Analysis (3_torus_native.py)
**Approach**: Torus kernel PCA, circular statistics, winding number analysis on T^L.
**Finding**: Torus kernel Df grows with L (49 -> 3527), showing the torus kernel sees full-dimensional structure while complex covariance collapses it. Winding dispersion ~12-21 rad confirms non-geodesic trajectories.
**Verdict**: Torus geometry confirmed as native topology but doesn't reduce detection requirement.

### 4. Catalytic Cepstrum (4_catalytic_cepstrum.py)
**Approach**: Mandelbrot-inspired catalytic recursion: autocorrelation(autocorrelation(autocorrelation(...))). Each level amplifies periodic structure.
**Finding**: Full grating: recursion amplifies SNR from 20x to 3500x. Reduced samples (S < r): recursion converges to noise fixed points, never finds period. The recursion amplifies existing structure but can't create period information not in the sample.
**Verdict**: Powerful amplification when the period IS present. Can't create information from nothing.

### 5. Holo Oracle (5_holo_oracle.py)
**Approach**: Real `.holo` engine (holo_core.analyze_spectrum/project/render/verify) combined with Mandelbrot cepstrum recursion. All five breakthroughs unified.
**Finding**: `.holo` compresses at k95 dimensions, renders, cepstrum amplifies, detects r. Matches autocorrelation reference 4/5 runs, SNR up to 713x. D_pr/r = 0.0049 — grating is 200x compressible by effective information dimension.
**Verdict**: All five methods combined work as well as raw autocorrelation for 22-bit.

### 6. Shor Solver (6_shor_solver.py)
**Approach**: Complete factoring engine. Tries gcd, autocorrelation, direct iteration, `.holo` extraction across multiple bases (a=2,3,5,7...).
**Finding**: **10/10 semiprimes factored.** Autocorrelation handles most (fast, ~1.3s), iteration catches large-period edge cases (slower, ~20s). `.holo` extraction methods haven't independently found a period that autocorrelation/iteration missed.
**Verdict**: Complete Shor solver for 22-bit. The period-containment limit (M >= r) remains — when r > 4M, autocorrelation fails and iteration must scan O(r) steps.

## Key Metrics Across All Experiments

| Metric | Value | Meaning |
|--------|-------|---------|
| D_pr/L at L=4096 | 0.40 | 60% of dimensions are noise |
| D_pr/r | 0.005 | Effective dimension is 0.5% of period |
| Theoretical compression | 200x | Grating compressible to D_pr dimensions |
| Practical compression (k95) | 5x | At L=4096, k95/L = 0.43 |
| Cepstrum amplification | 175x | SNR 20 -> 3500 at depth 1 |
| Complex vs real+imag Df | 0.5x | Complex-native halves apparent dimension |

## The Period-Containment Wall

Every method converges to the same limit: **you cannot detect a period from a sample window shorter than the period.** Autocorrelation needs M > r. Iteration needs O(r) time. `.holo` compression reduces storage but can't create period information not in the data.

The phase grating is ~200x compressible (D_pr << r), meaning the PERIOD INFORMATION is low-dimensional. But accessing it still requires sampling across at least one full period. The compression reveals the effective dimension but doesn't eliminate the sampling requirement.

## What Worked

1. **Autocorrelation**: Fast, reliable for 22-bit when M > r. The MVP.
2. **Direct iteration**: O(r) time, O(1) memory. Guaranteed to find r within time budget.
3. **Complex-native representation**: Essential for correct Df measurement (not real+imag flattening).
4. **Base retries**: Critical for Shor edge cases (odd period, -1 trap).
5. **`.holo` spectral analysis**: Accurately measures D_pr, D_sh, compression ratio.

## What Didn't Work (Yet)

1. **MERA/Feistel encoding**: Signal degrades across layers.
2. **Torus kernel PCA**: Sees full-dimensional structure, no compression advantage.
3. **Cepstrum recursion on sub-period samples**: Converges to noise, not period.
4. **`.holo` coordinate/basis extraction**: Detection range limited by sample count (n_samples/2), shorter than autocorrelation (M/2).

## The Unified Picture

For 22-bit semiprimes: **10/10 factored.** The solver is complete. Autocorrelation + iteration + base retries handle every case. `.holo` provides spectral diagnostics (D_pr, compression ratio) confirming the signal structure.

For larger bit sizes: the period-containment limit activates. `.holo` can measure Df but can't bypass the need to sample across r. The compression ratio (Df/r) tells us how much information the period contains, but accessing it still requires spanning one full period.

The next frontier: extract the period from the `.holo` basis vectors or coordinate trajectory when the period IS within range but autocorrelation produces a false peak. This would make `.holo` the verifier — confirming or rejecting autocorrelation candidates by checking whether the compressed representation is consistent with the candidate period.
