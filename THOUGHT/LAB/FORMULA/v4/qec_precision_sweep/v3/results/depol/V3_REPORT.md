# QEC Precision Sweep v3 -- Direct Formula Test

Run id: `v3_depol`
Noise model: `depol`
Source data: `qec_v2_depol`
UTC time: `2026-05-13T10:21:49.625710+00:00`

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

- Direct MAE (test): `1.9442`
- Direct R2 (test): `0.2391`

### Zero-Fitting Diagnostic

If the formula is correct, R_actual = 1.0 * R_predicted + 0.0.

- Fitted alpha: `0.9909`  (ideal = 1.0)
- Fitted beta:  `-1.7799`  (ideal = 0.0)
- Residual std: `1.2868`
- alpha near 1: `True`
- beta near 0:  `False`

### Baseline Comparison (with learned coefficients, for context)

| Model | Test MAE | Test R2 |
|---|---:|---:|
| `standard_qec_scaling` | 0.8422 | 0.8030 |
| `p_only` | 1.2076 | 0.6088 |
| `distance_only` | 2.2734 | 0.0016 |

## 2. Slope Test: Does ln(R) grow as Df * ln(sigma)?

For each error rate and basis, the formula claims:
```
ln(R_actual) = ln(E/grad_S) + Df * ln(sigma)
```

Slope match fraction: `83.33%` (18 p/basis combos)

| p | basis | n_dists | empir slope | pred slope | slope error | match |
|---:|---:|---:|---:|---:|---:|---|
| 0.0005 | x | 4 | 0.4652 | -0.0076 | 0.4728 | True |
| 0.0005 | z | 4 | 0.3847 | -0.0075 | 0.3923 | True |
| 0.0010 | x | 4 | 0.5784 | -0.0152 | 0.5936 | False |
| 0.0010 | z | 4 | 0.3845 | -0.0149 | 0.3994 | True |
| 0.0020 | x | 4 | 0.8125 | -0.0302 | 0.8427 | False |
| 0.0020 | z | 4 | 0.6521 | -0.0299 | 0.6819 | False |
| 0.0040 | x | 4 | 0.2567 | -0.0596 | 0.3163 | True |
| 0.0040 | z | 4 | 0.2700 | -0.0585 | 0.3285 | True |
| 0.0060 | x | 4 | 0.0535 | -0.0881 | 0.1416 | True |
| 0.0060 | z | 4 | 0.0791 | -0.0862 | 0.1653 | True |
| 0.0080 | x | 4 | -0.0659 | -0.1149 | 0.0490 | True |
| 0.0080 | z | 4 | -0.0416 | -0.1131 | 0.0715 | True |
| 0.0100 | x | 4 | -0.1427 | -0.1417 | -0.0010 | True |
| 0.0100 | z | 4 | -0.1132 | -0.1390 | 0.0258 | True |
| 0.0200 | x | 4 | -0.1500 | -0.2596 | 0.1096 | True |
| 0.0200 | z | 4 | -0.1651 | -0.2545 | 0.0894 | True |
| 0.0400 | x | 4 | -0.0416 | -0.4318 | 0.3902 | True |
| 0.0400 | z | 4 | -0.0432 | -0.4259 | 0.3827 | True |

### Intercept Check

| p | basis | empir intercept | pred intercept | error |
|---:|---:|---:|---:|---:|
| 0.0005 | x | -0.8391 | 4.8857 | -5.7248 |
| 0.0005 | z | 0.0461 | 4.8977 | -4.8516 |
| 0.0010 | x | -1.5393 | 4.2011 | -5.7403 |
| 0.0010 | z | 0.5579 | 4.2251 | -3.6672 |
| 0.0020 | x | -3.3016 | 3.5204 | -6.8220 |
| 0.0020 | z | -2.6630 | 3.5319 | -6.1949 |
| 0.0040 | x | -1.9475 | 2.8546 | -4.8022 |
| 0.0040 | z | -1.9155 | 2.8745 | -4.7899 |
| 0.0060 | x | -1.7234 | 2.4774 | -4.2008 |
| 0.0060 | z | -1.6907 | 2.4996 | -4.1903 |
| 0.0080 | x | -1.6000 | 2.2254 | -3.8254 |
| 0.0080 | z | -1.5635 | 2.2411 | -3.8046 |
| 0.0100 | x | -1.5009 | 2.0284 | -3.5293 |
| 0.0100 | z | -1.4870 | 2.0476 | -3.5346 |
| 0.0200 | x | -1.9271 | 1.4795 | -3.4066 |
| 0.0200 | z | -1.7636 | 1.4979 | -3.2615 |
| 0.0400 | x | -2.1965 | 1.0504 | -3.2469 |
| 0.0400 | z | -2.1893 | 1.0618 | -3.2511 |

## 3. Verdict

**PARTIAL**: Alpha near 1, but beta = -1.7799 (not near 0). Formula captures relative scaling but has systematic offset.
- Slope structure matches: 83% of p/basis conditions show ln(R) ∝ Df * ln(sigma).

## 4. Per-Point Errors (test distances only)

| basis | d | p | R_actual | R_predicted | error |
|---|---:|---:|---:|---:|---:|
| x | 7 | 0.0005 | 2.9958 | 4.7646 | 1.7688 |
| z | 7 | 0.0005 | 2.9958 | 4.7702 | 1.7745 |
| x | 9 | 0.0005 | 2.9958 | 4.7083 | 1.7125 |
| z | 9 | 0.0005 | 2.9958 | 4.7127 | 1.7169 |
| x | 7 | 0.0010 | 2.5903 | 4.0162 | 1.4259 |
| z | 7 | 0.0010 | 3.6889 | 4.0258 | 0.3369 |
| x | 9 | 0.0010 | 3.6889 | 3.9400 | 0.2510 |
| z | 9 | 0.0010 | 3.6889 | 3.9474 | 0.2585 |
| x | 7 | 0.0020 | 1.9842 | 3.2307 | 1.2465 |
| z | 7 | 0.0020 | 1.8171 | 3.2424 | 1.4253 |
| x | 9 | 0.0020 | 4.3821 | 3.1130 | 1.2691 |
| z | 9 | 0.0020 | 3.2835 | 3.1214 | 0.1620 |
| x | 7 | 0.0040 | -0.1233 | 2.3536 | 2.4769 |
| z | 7 | 0.0040 | -0.2575 | 2.3710 | 2.6285 |
| x | 9 | 0.0040 | 0.3839 | 2.1642 | 1.7803 |
| z | 9 | 0.0040 | 0.6564 | 2.1785 | 1.5221 |
| x | 7 | 0.0060 | -1.3684 | 1.7656 | 3.1340 |
| z | 7 | 0.0060 | -1.0916 | 1.7909 | 2.8825 |
| x | 9 | 0.0060 | -1.2151 | 1.5128 | 2.7279 |
| z | 9 | 0.0060 | -0.9886 | 1.5307 | 2.5193 |
| x | 7 | 0.0080 | -2.0922 | 1.3137 | 3.4059 |
| z | 7 | 0.0080 | -1.8761 | 1.3305 | 3.2066 |
| x | 9 | 0.0080 | -2.1543 | 0.9884 | 3.1427 |
| z | 9 | 0.0080 | -1.9137 | 1.0025 | 2.9161 |
| x | 7 | 0.0100 | -2.5682 | 0.9289 | 3.4970 |
| z | 7 | 0.0100 | -2.3211 | 0.9534 | 3.2745 |
| x | 9 | 0.0100 | -2.7247 | 0.5328 | 3.2575 |
| z | 9 | 0.0100 | -2.4494 | 0.5540 | 3.0035 |
| x | 7 | 0.0200 | -3.0615 | -0.4799 | 2.5816 |
| z | 7 | 0.0200 | -3.0009 | -0.4446 | 2.5562 |
| x | 9 | 0.0200 | -3.1747 | -1.1469 | 2.0278 |
| z | 9 | 0.0200 | -3.1537 | -1.1160 | 2.0377 |
| x | 7 | 0.0400 | -2.5167 | -2.1398 | 0.3769 |
| z | 7 | 0.0400 | -2.5204 | -2.0974 | 0.4230 |
| x | 9 | 0.0400 | -2.5177 | -3.1744 | 0.6567 |
| z | 9 | 0.0400 | -2.5252 | -3.1349 | 0.6097 |
