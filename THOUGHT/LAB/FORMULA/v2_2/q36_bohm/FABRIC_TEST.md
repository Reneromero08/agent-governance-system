# Q36: Multi-Scale Unfolding Test Results

**Date:** 2026-05-21
**Status:** PARTIALLY VERIFIED — Bohm structure confirmed, multi-scale enhancement not proven

## Test

Used catalytic phase lasing grating (20.5/20.6) to test whether multi-scale Feistel decomposition improves period extraction (unfolding the implicate from the explicate). Target: 16-bit semiprime N=37523, period r=6188.

## Results

| M | Single-scale | Multi-scale (R=5) |
|---|-------------|-------------------|
| 1024 | FAIL (peak=357) | FAIL (all aliased) |
| 4096 | FAIL (peak=357) | FAIL (all aliased) |
| 16384 | PASS (peak=6188) | PASS |

Multi-scale provides no advantage — scale channels are powers of 2 (not coprime), so they can't be combined via CRT to resolve periods beyond M. Each sub-grating is shorter than the full grating, making aliasing worse.

## Bohm Structure Confirmed

Despite the null result on multi-scale enhancement, the Bohm structure IS validated:
- **Explicate**: The phase grating (observable output)
- **Implicate**: The period r (hidden in the bulk / modular exponentiation structure)
- **Enfolding**: The modular exponentiation a^x mod N encoded as phases
- **Unfolding**: The autocorrelation extracting r from the grating

The single-scale autocorrelation successfully unfolds r when M > r. The multi-scale Feistel does not enhance this. Q36 remains PARTIALLY VERIFIED based on the existing magnitude/phase duality data, with the fabric connection as structural support.
