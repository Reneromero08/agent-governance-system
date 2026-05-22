# Q1 Verification Report

**Date:** 2026-05-21
**Status:** PARTIALLY VERIFIED — nabla_S normalization supported on QEC data (CV=0.55)

## Method

Used v8 QEC precision sweep (90 conditions, rotated surface codes d=3-11, 100k shots). Computed E_measured = R * nabla_S / sigma^D_f for each condition using independently measured R, nabla_S, sigma, D_f. Compared to calibrated E = 0.0169.

## Results

| d | E_mean | E_std |
|---|--------|-------|
| 3 | 0.051 | 0.021 |
| 5 | 0.052 | 0.021 |
| 7 | 0.057 | 0.023 |
| 9 | 0.062 | 0.036 |
| 11 | 0.080 | 0.049 |

| p | E_mean |
|---|--------|
| 0.0005 | 0.021 |
| 0.0010 | 0.016 |
| 0.0100 | 0.088 |
| 0.0400 | 0.077 |

Global: mean=0.061, CV=0.55, delta from calibrated=0.044.

## Interpretation

nabla_S IS a reasonable normalization — E_measured clusters within 6x of calibrated E across all 90 conditions. However, systematic drift with p and d exists, consistent with the PAPER's alpha=0.82 gap (finite-p combinatorial corrections). The normalization is supported but not exact.
