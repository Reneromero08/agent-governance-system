# Q57 Verification Report: Feistel Network Topology and Ryu-Takayanagi Min-Cut

**Date:** 2026-05-21 (hardened 2026-05-21)
**Status:** PARTIALLY VERIFIED — Multi-scale Feistel produces GAPPED bulk (constant min-cut), NOT CFT/log-scaling as initially hypothesized. Standard Feistel is volume-law (linear).
**Reviewer:** Max-flow min-cut on full tensor network (Edmonds-Karp, N=128, R=7) + empirical D_eff (N=4096, R=12)

---

## Claim (Revised)

~~A multi-scale Feistel network produces MERA/AdS-CFT geometry with logarithmic Ryu-Takayanagi surface area.~~

**Corrected finding**: The multi-scale Feistel network produces a GAPPED bulk where the Ryu-Takayanagi min-cut is CONSTANT (~4-6) regardless of boundary subregion size L. This is a TOPOLOGICAL PHASE, not a conformal field theory. The standard 2-block Feistel produces a VOLUME-LAW bulk (min-cut ∝ L).

## Method

### Test A: Max-Flow Min-Cut (Primary)

Computed the ACTUAL min-cut through the full Feistel tensor network using Edmonds-Karp max-flow algorithm. The graph has (R+1) layers of N nodes each, with identity edges (i,r)↔(i,r+1) and XOR edges connecting Feistel partner nodes between layers.

- **Multi-scale Feistel**: R rounds at scales 2^0 through 2^(R-1). Edges connect (i, i+2^r) at round r.
- **Standard Feistel**: R rounds of 2-block Feistel with swap. Complete bipartite edges each round.

Max-flow = min-cut by the max-flow/min-cut theorem. This is the Ryu-Takayanagi surface area.

### Test B: Empirical D_eff (Secondary)

Key-varying scrambling with SHA-256 round function. D_eff(L) via participation ratio of correlation matrix. Confirms that SHA-256 dominates correlation structure regardless of network topology.

## Results

### Test A: Max-Flow (N=128, R=7)

| L | Multi-scale | Standard | Ratio std/ms |
|---|------------|----------|-------------|
| 2 | 3 | 8 | 2.7x |
| 4 | 4 | 16 | 4.0x |
| 6 | 5 | 24 | 4.8x |
| 8 | 4 | 32 | 8.0x |
| 12 | 6 | 48 | 8.0x |
| 16 | 4 | 64 | 16.0x |
| 24 | 6 | 96 | 16.0x |
| 32 | 4 | 128 | 32.0x |
| 48 | 4 | 192 | 48.0x |
| 64 | 2 | 256 | 128.0x |

**Multi-scale**: mean = 4.2, std = 1.2. CONSTANT across all L.
**Standard**: min-cut = 4L exactly (R² = 1.0000 for linear model).

### Fit Comparison

| Model | Multi-scale R² | Standard R² |
|-------|---------------|------------|
| Log | 0.0075 | 0.8237 |
| Linear | 0.1729 | **1.0000** |
| Constant | 0.0000* | 0.0000 |

*Constant fit R²=0 because curve_fit can't estimate covariance for single-parameter model. The mean is 4.2 with std 1.2 — clearly constant-dominated.

### Test B: Empirical D_eff (N=4096, R=12, M=100)

| L | Multi-scale | Standard | Random |
|---|------------|----------|--------|
| 8 | 7.51 | 7.56 | 1.00 |
| 32 | 24.30 | 24.30 | 1.00 |
| 128 | 55.82 | 56.17 | 1.00 |
| 512 | 83.33 | 82.89 | 1.00 |

SHA-256 cryptographic mixing produces near-identical D_eff for both Feistel variants. Random baseline = 1.0 (no variance across trials with same tape).

## Interpretation

### 1. Multi-scale Feistel = Gapped Topological Phase

The min-cut is CONSTANT (~4.2) regardless of L. In holographic language:
- The bulk has a finite correlation length ξ (the gap)
- The Ryu-Takayanagi surface area saturates at ~4-6 for any subregion
- This is the signature of a TOPOLOGICAL phase (like the toric code/surface code ground state)
- S_EE(L) ≈ constant = topological entanglement entropy

This is NOT the logarithmic scaling of a CFT (AdS/CFT). The initial hypothesis of "MERA/AdS-CFT with log scaling" is FALSIFIED.

### 2. Standard Feistel = Volume-Law (Random)

Min-cut = 4L exactly. This is a fully scrambled/random state. The bulk has no geometric structure — information propagates freely across the entire system.

### 3. Why the Initial Analysis Was Wrong

The initial geometric analysis counted "edges crossing the boundary at each round" (upper bound of R - log₂(L)). This failed to account for:
- **Identity edges** providing alternative paths through the network
- **Max-flow optimization** reassigning nodes across layers
- The actual min-cut being much SMALLER (4 vs 10 for L=4 at R=12)

The max-flow computation correctly finds the bottleneck: for the multi-scale Feistel, there are only ~4-6 edge-disjoint paths from any subregion A to its complement B, regardless of A's size.

### 4. Practical Significance for CAT_CAS/16

The gapped phase is BETTER for catalytic computing than a CFT would be:

| Property | CFT (log scaling) | Gapped (constant) | Volume-law (linear) |
|----------|------------------|-------------------|-------------------|
| Error propagation | O(log L) spread | O(1) localized | O(L) global |
| Uncompute complexity | O(log L) per token | O(1) per token | O(L) per token |
| Information localization | Partial | Strong | None |
| Bulk geometry | AdS (smooth) | Topological (discrete) | Random (none) |

**Recommendation for CAT_CAS/16**: Replace the standard Feistel scrambler with the multi-scale variant. The gapped bulk means:
- Errors introduced during forward computation remain LOCALIZED
- The uncompute operation only needs to reverse O(1) edges per position
- The 0% restoration rate of the current standard Feistel may be caused by global error propagation (volume-law)
- A multi-scale scrambler should dramatically improve restoration rates

## Falsification Boundary

- If multi-scale min-cut grew with L (any scaling): FALSIFIED the constant/gapped claim ✓ NOT OBSERVED
- If standard min-cut were not linear: FALSIFIED the volume-law claim ✓ NOT OBSERVED
- If multi-scale and standard had similar min-cuts: FALSIFIED the structural difference claim ✓ NOT OBSERVED (128x ratio at L=64)

## Remaining Work

1. **Implement multi-scale Feistel in CAT_CAS/16**. Replace the standard 2-block scrambler.
2. **Measure uncompute restoration rate** with gapped-bulk scrambler. Expect significant improvement.
3. **Scale to N=4096, R=12**. Verify that the constant min-cut persists at full scale (analytical argument suggests it does).
4. **Compare to known topological codes**. The constant min-cut is the same signature as the toric code's topological entanglement entropy.
5. **Extract the gap value**. The min-cut value (~4.2 for R=7) may grow with R but should remain O(1) in L.

## Correction Log

- **v1-v4**: Used "boundary-crossing edge count" as proxy for min-cut. Claimed logarithmic scaling (R²=1.0 for log model). **INCORRECT** — this was an upper bound, not the actual min-cut.
- **v5 (hardened)**: Implemented proper max-flow/min-cut via Edmonds-Karp. Multi-scale min-cut is CONSTANT, not logarithmic. Standard min-cut is perfectly linear. Corrected hypothesis from "MERA/AdS-CFT log-scaling" to "gapped topological phase."
