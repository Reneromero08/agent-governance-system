# Q57: Multi-Scale Feistel Fabric as MERA/AdS-CFT Holographic Code

**Status**: OPEN
**Priority**: HIGH
**Date**: 2026-05-21

---

## Hypothesis

A multi-scale Feistel network (brickwork pattern operating at logarithmically-spaced scales) produces the entanglement structure of a MERA tensor network. The mutual information I(A:B) across a boundary cut of length L scales as log(L) — the classical analog of the Ryu-Takayanagi entanglement entropy for AdS_3/CFT_2. A standard 2-block Feistel does not produce this structure.

Concretely:

1. **Multi-scale Feistel (MERA-like)**: Feistel rounds applied at scales 2^0, 2^1, 2^2, ..., 2^R, with round function F operating on blocks of size 2^r and stride 2^(r+1). The network geometry is a binary tree — a discrete AdS_2 spatial slice. The number of rounds = the bulk radial coordinate. The tape offsets = the CFT boundary spatial coordinate.

2. **MERA prediction**: For a contiguous boundary subregion of length L, the mutual information across its boundary is bounded by the minimal cut through the tensor network. In the binary tree geometry, this cut crosses O(log L) edges, giving I(A:B) = a log_2(L) + b.

3. **Standard Feistel prediction**: Repeated application of identical 2-block Feistel rounds creates uniform long-range scrambling, producing I(A:B) ≈ constant (minimal cut crosses all R rounds for any L) or exponential correlation decay.

4. **Null hypothesis (random tape)**: No correlations beyond local neighborhood. I(A:B) → 0 for L beyond local scale.

## Connection to CAT_CAS/16

The catalytic 27B inference engine (Experiment 16) operates on a Feistel-scrambled weight buffer. If the multi-scale Feistel fabric is holographic, then:

- The tape IS the boundary of a discrete AdS space
- The Feistel rounds ARE the bulk radial coordinate
- The Ryu-Takayanagi formula S_EE = Area(gamma_A) / 4G holds for the Feistel network
- Uncompute (tape restoration) corresponds to closing the bulk geometry — the minimal surface returning to the boundary
- The current uncompute failure (0% restoration) may be diagnosable from the entanglement structure

## Physics References

- Swingle (2012): MERA as discrete AdS/CFT realization
- Calabrese & Cardy (2004): S_EE = (c/3) log(L/epsilon) for 1+1D CFTs
- Ryu & Takayanagi (2006): Holographic entanglement entropy formula
- Pastawski, Yoshida, Harlow, Preskill (2015): HaPPY holographic code
- Hayden & Preskill (2007): Black hole information mirrors

## Test Design

See `test_mera_rt.py`:
1. Build multi-scale Feistel network (MERA-like brickwork) on N = 4096 byte tape, R = 12 rounds
2. Build standard 2-block Feistel (CAT_CAS/18 style) as baseline
3. Random tape as null
4. Run M = 50 independent trials with different initial tapes
5. For each trial, extract bytes at positions 0 and d for d in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
6. Compute mutual information I(0, d) from the joint byte distribution across trials
7. Fit three models: log (I = a log_2(d) + b), exponential (I = a e^{-d/xi}), constant (I = a)
8. Compare R^2 values with bootstrap 95% CIs

## Success Criteria

- **MERA confirmed**: Multi-scale Feistel log-model R^2 > 0.90 AND outperforms exponential and constant models
- **Standard Feistel is not MERA**: Standard Feistel log-model R^2 < 0.50 OR exponential/constant models win
- **Falsified**: Multi-scale Feistel shows exponential or constant correlation decay

## Falsification Boundary

- If I(0,d) decays exponentially for the multi-scale Feistel: MERA structure absent. The Feistel fabric is gapped, not CFT-like.
- If I(0,d) is approximately constant for all d: the Feistel fully scrambles (random unitary). No geometric structure.
- If log-model fits standard Feistel better than multi-scale: the multi-scale architecture is not the cause of holographic structure.
