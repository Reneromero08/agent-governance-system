# QEC Precision Sweep v3 -- Direct Formula Test

Run id: `v3_meas`
Noise model: `meas`
Source data: `qec_v2_meas`
UTC time: `2026-05-13T10:21:52.282470+00:00`

## v3 Operational Definitions (all per-condition, measurable)

| Symbol | Definition | Source |
|--------|-----------|--------|
| `E` | `1.0` | normalized signal power |
| `grad_S` | syndrome density | fraction of detectors firing per shot |
| `sigma` | `1 - syndrome_density` | fraction of quiet detectors |
| `Df` | `d` | surface-code distance |
| `R_pred` | `(E / grad_S) * sigma^Df` | direct prediction, no fitting |

**Key difference from v2**: no p_th, no sqrt/threshold ratios, no learned α/β.
Every quantity is measured directly from the syndrome data per condition.

## 1. Direct Prediction (no learned coefficients)

- Direct MAE (test): `1.0724`
- Direct R2 (test): `0.7345`

### Zero-Fitting Diagnostic

If the formula is correct, R_actual = 1.0 * R_predicted + 0.0.

- Fitted alpha: `1.0092`  (ideal = 1.0)
- Fitted beta:  `-0.6503`  (ideal = 0.0)
- Residual std: `1.0793`
- alpha near 1: `True`
- beta near 0:  `False`

### Baseline Comparison (with learned coefficients, for context)

| Model | Test MAE | Test R2 |
|---|---:|---:|
| `standard_qec_scaling` | 1.0852 | 0.4992 |
| `p_only` | 1.5707 | 0.3107 |
| `distance_only` | 2.1183 | -0.0676 |

## 2. Slope Test: Does ln(R) grow as Df * ln(sigma)?

For each error rate and basis, the formula claims:
```
ln(R_actual) = ln(E/grad_S) + Df * ln(sigma)
```

Slope match fraction: `55.56%` (18 p/basis combos)

| p | basis | n_dists | empir slope | pred slope | slope error | match |
|---:|---:|---:|---:|---:|---:|---|
| 0.0005 | x | 4 | 0.2919 | -0.0066 | 0.2985 | True |
| 0.0005 | z | 4 | 0.2414 | -0.0067 | 0.2481 | True |
| 0.0010 | x | 4 | 0.5493 | -0.0134 | 0.5627 | False |
| 0.0010 | z | 4 | 0.5882 | -0.0134 | 0.6016 | False |
| 0.0020 | x | 4 | 0.7772 | -0.0265 | 0.8037 | False |
| 0.0020 | z | 4 | 0.7653 | -0.0266 | 0.7919 | False |
| 0.0040 | x | 4 | 0.9146 | -0.0529 | 0.9676 | False |
| 0.0040 | z | 4 | 0.6188 | -0.0524 | 0.6712 | False |
| 0.0060 | x | 4 | 0.5127 | -0.0780 | 0.5908 | False |
| 0.0060 | z | 4 | 0.4961 | -0.0780 | 0.5741 | False |
| 0.0080 | x | 4 | 0.3163 | -0.1032 | 0.4194 | True |
| 0.0080 | z | 4 | 0.3174 | -0.1031 | 0.4205 | True |
| 0.0100 | x | 4 | 0.2437 | -0.1279 | 0.3717 | True |
| 0.0100 | z | 4 | 0.2368 | -0.1270 | 0.3638 | True |
| 0.0200 | x | 4 | -0.0519 | -0.2408 | 0.1889 | True |
| 0.0200 | z | 4 | -0.0495 | -0.2407 | 0.1912 | True |
| 0.0400 | x | 4 | -0.0679 | -0.4229 | 0.3550 | True |
| 0.0400 | z | 4 | -0.0650 | -0.4214 | 0.3564 | True |

### Intercept Check

| p | basis | empir intercept | pred intercept | error |
|---:|---:|---:|---:|---:|
| 0.0005 | x | 0.7580 | 5.0191 | -4.2611 |
| 0.0005 | z | 1.1449 | 5.0139 | -3.8689 |
| 0.0010 | x | -0.7055 | 4.3197 | -5.0252 |
| 0.0010 | z | -1.0040 | 4.3221 | -5.3261 |
| 0.0020 | x | -1.9007 | 3.6424 | -5.5431 |
| 0.0020 | z | -1.8513 | 3.6408 | -5.4921 |
| 0.0040 | x | -3.6715 | 2.9656 | -6.6371 |
| 0.0040 | z | -2.3598 | 2.9758 | -5.3355 |
| 0.0060 | x | -2.5931 | 2.5895 | -5.1827 |
| 0.0060 | z | -2.5555 | 2.5898 | -5.1453 |
| 0.0080 | x | -2.1313 | 2.3228 | -4.4541 |
| 0.0080 | z | -2.1973 | 2.3234 | -4.5207 |
| 0.0100 | x | -2.2138 | 2.1197 | -4.3335 |
| 0.0100 | z | -2.1353 | 2.1263 | -4.2616 |
| 0.0200 | x | -1.7917 | 1.5420 | -3.3337 |
| 0.0200 | z | -1.7976 | 1.5423 | -3.3399 |
| 0.0400 | x | -1.9561 | 1.0646 | -3.0207 |
| 0.0400 | z | -1.9767 | 1.0677 | -3.0443 |

## 3. Verdict

**PARTIAL**: Alpha near 1, but beta = -0.6503 (not near 0). Formula captures relative scaling but has systematic offset.
- Slope structure weak: only 56% of conditions match.

## 4. Per-Point Errors (test distances only)

| basis | d | p | R_actual | R_predicted | error |
|---|---:|---:|---:|---:|---:|
| x | 7 | 0.0005 | 2.9958 | 4.9428 | 1.9470 |
| z | 7 | 0.0005 | 2.9958 | 4.9503 | 1.9545 |
| x | 9 | 0.0005 | 2.9958 | 4.9284 | 1.9326 |
| z | 9 | 0.0005 | 2.9958 | 4.9256 | 1.9299 |
| x | 7 | 0.0010 | 3.6889 | 4.2148 | 0.5259 |
| z | 7 | 0.0010 | 3.6889 | 4.2101 | 0.5212 |
| x | 9 | 0.0010 | 3.6889 | 4.1804 | 0.4914 |
| z | 9 | 0.0010 | 3.6889 | 4.1837 | 0.4947 |
| x | 7 | 0.0020 | 4.3821 | 3.4358 | 0.9463 |
| z | 7 | 0.0020 | 4.3821 | 3.4410 | 0.9411 |
| x | 9 | 0.0020 | 4.3821 | 3.3753 | 1.0067 |
| z | 9 | 0.0020 | 4.3821 | 3.3763 | 1.0058 |
| x | 7 | 0.0040 | 2.0307 | 2.5796 | 0.5489 |
| z | 7 | 0.0040 | 2.1308 | 2.5903 | 0.4595 |
| x | 9 | 0.0040 | 5.0752 | 2.4729 | 2.6023 |
| z | 9 | 0.0040 | 3.1293 | 2.4723 | 0.6570 |
| x | 7 | 0.0060 | 1.1369 | 2.0225 | 0.8856 |
| z | 7 | 0.0060 | 0.9268 | 2.0230 | 1.0962 |
| x | 9 | 0.0060 | 1.9842 | 1.8566 | 0.1275 |
| z | 9 | 0.0060 | 1.9253 | 1.8555 | 0.0698 |
| x | 7 | 0.0080 | 0.1663 | 1.5856 | 1.4194 |
| z | 7 | 0.0080 | -0.0517 | 1.5811 | 1.6328 |
| x | 9 | 0.0080 | 0.6746 | 1.3592 | 0.6846 |
| z | 9 | 0.0080 | 0.7121 | 1.3642 | 0.6521 |
| x | 7 | 0.0100 | -0.4590 | 1.2088 | 1.6677 |
| z | 7 | 0.0100 | -0.5262 | 1.2146 | 1.7407 |
| x | 9 | 0.0100 | -0.0606 | 0.9352 | 0.9958 |
| z | 9 | 0.0100 | 0.0331 | 0.9384 | 0.9054 |
| x | 7 | 0.0200 | -2.1340 | -0.1724 | 1.9616 |
| z | 7 | 0.0200 | -2.1533 | -0.1678 | 1.9855 |
| x | 9 | 0.0200 | -2.2704 | -0.6792 | 1.5912 |
| z | 9 | 0.0200 | -2.2419 | -0.6712 | 1.5707 |
| x | 7 | 0.0400 | -2.4729 | -1.9180 | 0.5548 |
| z | 7 | 0.0400 | -2.4702 | -1.9205 | 0.5497 |
| x | 9 | 0.0400 | -2.5194 | -2.7963 | 0.2769 |
| z | 9 | 0.0400 | -2.5182 | -2.7918 | 0.2736 |
