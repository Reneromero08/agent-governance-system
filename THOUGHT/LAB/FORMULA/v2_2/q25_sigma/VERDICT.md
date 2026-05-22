# Q25 Verification Report

**Date:** 2026-05-21
**Status:** VERIFIED — Sigma derivable from first principles

## Claim

Sigma (per-round fidelity factor in R = (E/nabla_S) * sigma^D_f) is derivable from the hash function entropy of the Feistel fabric.

## Method

Single-round byte match rate after Feistel XOR with tunable hash mask. For mask M with h = popcount(M) bits set, each masked hash bit is 0 or 1 with P=0.5 (SHA-256 output is uniform). A signal byte survives unchanged only if ALL h masked hash bits are 0: P = (1/2)^h = 2^(-h).

Measured: sigma_measured = fraction of signal bytes unchanged after 1 round.
Theory: sigma_theory = 2^(-h).

## Results

| mask | h | theory | measured | delta |
|------|---|--------|----------|-------|
| 0x00 | 0 | 1.000000 | 1.000000 | +0.000000 |
| 0x01 | 1 | 0.500000 | 0.500723 | +0.000723 |
| 0x0F | 4 | 0.062500 | 0.063320 | +0.000820 |
| 0x1F | 5 | 0.031250 | 0.030898 | -0.000352 |
| 0x3F | 6 | 0.015625 | 0.015664 | +0.000039 |
| 0x7F | 7 | 0.007812 | 0.007793 | -0.000020 |
| 0xFF | 8 | 0.003906 | 0.003750 | -0.000156 |

**Fit**: measured = 1.0003 * theory + 0.0001, **R² = 1.0000**

## Conclusion

Sigma = 2^(-h) = 1/2^h, where h is the entropy (in bits) of the hash function per XOR edge. This is exactly the probability that all masked hash bits are zero — i.e., the probability the signal byte survives unchanged through one round.

Sigma IS derivable from first principles: the hash entropy per edge determines the per-round fidelity. Combined with Q57's min-cut (number of edge-disjoint paths per round = 1 for contiguous subregions), the complete formula is:

R = (E/nabla_S) * (2^(-h))^D_f

where h is the hash entropy per round and D_f is the number of rounds.

## Correction Log

- v1 (FALSIFIED): Used ln(corr) vs D_f slope — confounded instantaneous XOR decay with progressive diffusion. Measured sigma was artfact of noise floor.
- v2 (VERIFIED): Direct single-round byte match rate. Correct theory: P(byte survives) = P(all h hash bits = 0) = 2^(-h).
