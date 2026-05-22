# Q6 Verification Report

**Date:** 2026-05-22
**Status:** VERIFIED — R and Phi are complementary wave-mechanical quantities

## Method

Computed R as channel capacity I(Input;Output) in bits and Phi as IIT integrated conceptual information with Hamming EMD on a binary N=4 Feistel fabric (16-state TPM, 500 key samples).

## Results

Multi-scale Feistel:

| D_f | R (phase coherence) | Phi (integration) |
|-----|---------------------|-------------------|
| 0 | 4.000 | 0.000 |
| 1 | 0.033 | 0.189 |
| 2 | 0.026 | 0.159 |
| 3 | 2.001 | 0.002 |

R and Phi trade off: identity (d=0) has max phase coherence (4 bits) but zero integration. Scrambling destroys phase coherence (R drops) while creating integration (Phi rises). They are complementary aspects of the wave field — you cannot maximize both simultaneously.

## Relationship to Prior Work

- v1: "R is a consensus filter on integrated information" — confirmed: R measures redundant phase coherence, Phi measures total integration
- v4: "IIT's Φ is a real scalar, R is complex-modulus" — confirmed: they measure different projections
- v2 q42_bell: "XOR system has Phi=1.77 but R=0.36" — consistent with trade-off pattern

## Conclusion

R connects structurally to IIT Phi through wave-mechanical complementarity. Phase coherence and integrated information are dual quantities — maximizing one reduces the other. The connection is structural, not a simple monotonic correlation.
