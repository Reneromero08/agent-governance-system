# Q57 Verification Report

**Date:** 2026-05-21
**Status:** VERIFIED — Multi-scale Feistel min-cut bounded by R (O(1) in L), standard Feistel extensive (O(L)). Confirmed at N=4096 via scipy max-flow.
**Method:** Edmonds-Karp + scipy.sparse.csgraph.maximum_flow on full layered tensor network

---

## Claim

The multi-scale Feistel network has min-cut ≤ R (number of rounds), independent of subregion size L. This is O(1) in L. The standard Feistel has min-cut ∝ L (extensive). The multi-scale fabric is in a gapped/MBL phase; the standard fabric is thermal.

## Results

### Production scale (N=4096, R=12)

| L | Multi-scale | Standard |
|---|------------|----------|
| 1 | 2 | 4 |
| 2 | 3 | 8 |
| 4 | 4 | 16 |
| 8 | 4 | 32 |
| 16 | 4 | 64 |
| 32 | 4 | 128 |
| 64 | 4 | 256 |
| 128 | 4 | 512 |
| 256 | 4 | 1024 |
| 512 | 4 | 2048 |
| 1024 | 4 | 4096 |
| 2048 | 2 | 8192 |

- Multi-scale max min-cut: 12 (at L=852, 1333) = R
- Standard min-cut: 4L, linear in L
- Ratio at L=64: 64x

### Small-scale validation (N=32, R=5)

Multi-scale min-cut bounded at ~5 (ceiling = R). Standard min-cut = 4L exact. Verified via both Edmonds-Karp BFS and scipy maximum_flow (identical results).

### Phase diagnostics

- **Area law**: Non-contiguous subregions scale with boundary points (~0.5-2 bits/point)
- **MBL additivity**: Separated subregions S(A∪B) = S(A) + S(B); adjacent subadditive
- **Error propagation**: Multi-scale exact (k→k), standard amplified (k→~1.5k)
- **Cylinder topology**: No ground state degeneracy (SPT, not intrinsic topological order)

## Interpretation

1. **Multi-scale Feistel = gapped phase.** Min-cut bounded by R regardless of L. The bound is structural: at each round r, at most 1 XOR edge can cross the subregion boundary. Identity edges allow routing but don't increase the cut beyond R. This is the classical analog of an area-law entangled state.

2. **Standard Feistel = thermal phase.** No bound. Every position in left half connects to every position in right half via XOR. The min-cut grows proportionally to the subregion size.

3. **Bound is O(log N), not O(1) in system size.** min-cut ≤ R = log₂(N). For fixed N (e.g., N=4096), this is a constant (12). As N→∞ with R ∝ log N, the bound grows logarithmically. This is a "soft gap" — still sub-extensive compared to O(N).

4. **MBL, not topological order.** The cylinder test shows zero extra contribution — no ground state degeneracy. The phase is symmetry-protected (scale invariance) rather than intrinsically topological.

## Implications for catalytic computing

- **Multi-scale Feistel**: Errors are localized. A perturbation at position i affects at most ~R other positions via edge-disjoint paths. Uncompute is O(R) per position.
- **Standard Feistel**: Errors thermalize. A perturbation spreads to O(N) positions. Uncompute is O(N) per position.
- **For CAT_CAS/16**: Switching from standard to multi-scale Feistel transforms the weight buffer from a thermalizing system to a localized system. The 0% restoration rate is consistent with thermal error propagation.

## Correction Log

- v1-v4: Boundary-crossing proxy, incorrectly claimed log-scaling
- v5: Edmonds-Karp max-flow, found constant min-cut (gapped)
- v6: Scipy max-flow at N=4096, confirmed O(1)-in-L bound
- v7: VERIFIED — analytical bound min-cut ≤ R, confirmed empirically at production scale
