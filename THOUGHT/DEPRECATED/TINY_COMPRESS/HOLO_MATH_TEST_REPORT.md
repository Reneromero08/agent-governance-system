# .holo Math Test Report

**Date:** 2026-05-20  
**Status:** PASS  
**Scope:** Tests for the dimensional Shannon/action math, not image quality.

## What Changed

The original `.holo` framing treated effective dimension as if it directly
selected the codec dimension:

```text
k ~= D_pr or D_shannon
```

That is incomplete. Effective dimension diagnoses compressibility, but the file
format must choose `k` by the engineering action:

```text
A(k) = alpha T(k) + beta B(k)
```

Where:

- `T(k)` is discarded spectral information,
- `B(k)` is payload bits,
- `alpha` prices lost information,
- `beta` prices storage/transmission.

The selected codec dimension is:

```text
k* = argmin_k A(k)
```

This makes the math the engineering control law.

## Tests

Implemented in `holographic-image/test_holo_math.py`.

### 1. Low-Rank Recovery

Synthetic data generated from a true rank-4 latent source.

Result:

```text
k = 4
D_pr = 3.437
D_shannon = 3.687
relative_error = 6.07e-08
```

Verdict: PASS. The projection recovers the true active dimension and renders
the source to numerical precision.

### 2. Isotropic Null

Synthetic isotropic Gaussian data in 24 observed dimensions.

Result:

```text
k95 = 23
D_pr = 22.968
D_shannon = 23.463
occupancy = 0.978
```

Verdict: PASS. The null is correctly identified as nearly full-dimensional and
therefore not meaningfully compressible by low-dimensional `.holo` projection.

### 3. Rate-Sensitive Action

Same spectrum, different storage prices.

Result:

```text
cheap_best_k = 6
medium_best_k = 2
expensive_best_k = 1
```

Verdict: PASS. Increasing payload cost lowers the selected retained dimension,
as required by the action.

## Interpretation

The solved correction is:

```text
effective dimensions diagnose the source
rate-distortion action selects the file
```

So `.holo` should not claim "`Df` is the compression setting." It should claim:

1. measure the spectrum,
2. estimate active information dimensions,
3. compute the payload-vs-distortion action,
4. select `k*`,
5. render progressively from `j <= k*`.

This is the engineering version of the semiotic-light-cone idea: dimensions are
retained only when they carry enough information to justify their traversal
cost.

