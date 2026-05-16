# Phase 3f: Tensor Network Compression (quimb MPS vs SVD)

Date: 2026-05-16 | Status: **COMPLETE — SVD MORE EFFICIENT FOR 2D IMAGES**

---

## Summary

Matrix Product State (MPS) compression via quimb's `MatrixProductState.from_dense`
was compared against per-image SVD on 10 synthetic 32x32 images (rank 2).
SVD achieves perfect reconstruction at k=2 (7.9x compression). MPS achieves
perfect reconstruction at chi=8 (1.3x compression). SVD is more efficient
because it exploits the 2D matrix structure directly; MPS flattens to 1D,
creating artificial long-range entanglement.

## Method

- **Library**: quimb v1.14.0, pure Python tensor network library
- **MPS**: `MatrixProductState.from_dense(vec, max_bond=chi)` → `to_dense()`
- **SVD baseline**: per-image SVD truncated to k components
- **Images**: 10 synthetic images (sinusoidal, Gaussian, checkerboard), 32x32

## Results

| chi/k | SVD PSNR | MPS PSNR | Winner |
|-------|---------|---------|--------|
| 1 | 27.1 dB | n/a | — |
| 2 | 120.0 dB | 18.4 dB | SVD |
| 4 | 120.0 dB | 107.0 dB | SVD |
| 8 | 120.0 dB | 120.0 dB | Tie |
| 16+ | 120.0 dB | 120.0 dB | Tie |

Both achieve perfect reconstruction at sufficient rank/bond. SVD needs k=2;
MPS needs chi=8. The factor-of-4 difference arises because 1D flattening
creates artificial entanglement between pixels that are neighbors in 2D
but distant in the 1D chain.

## Compression Efficiency

| Method | Stored values | Compression ratio |
|--------|-------------|-------------------|
| SVD k=2 | 130 | 7.9x |
| MPS chi=8 | ~768 | 1.3x |
| Raw | 1024 | 1.0x |

SVD is 6x more compressive at equal quality for these low-rank images.

## Domain Finding

Tensor network compression (MPS/TT) is less efficient than SVD for 2D images
because the 1D flattening breaks spatial locality. For the formula mapping:

| Formula | SVD | MPS |
|---------|-----|-----|
| sigma | 7.9x compression | 1.3x compression |
| Df | k (2 for perfect) | chi (8 for perfect) |
| R | 120 dB | 120 dB (at sufficient chi) |

The formula correctly predicts that higher Df (chi/k) increases R. But the
Df required is higher for MPS because of the artificial 1D entanglement.

## Correction

The initial report claimed MPS achieved only 8 dB PSNR at all chi. This was
a shape-mismatch bug: `mps.to_dense()` returns a 2D array `(N, 1)`, and
comparing it to a 1D vector broadcasts the subtraction incorrectly, inflating
MSE by factor N. After fixing (`recon.flatten()[:N]`), results are as above.

## Files

- `phase3/seemps/mps_test.py` — MPS vs SVD comparison
- `phase3/seemps/REPORT.md` — this report
- `phase3/seemps/ERROR_REPORT.md` — SeeMPS Cython hang documentation
