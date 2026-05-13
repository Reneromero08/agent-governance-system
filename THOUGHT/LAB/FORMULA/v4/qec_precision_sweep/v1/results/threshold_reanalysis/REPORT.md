# QEC Threshold Mapping Reanalysis

Run id: `threshold_reanalysis_v1`
Source run: `full_surface_code_v1`
UTC time: `2026-05-13T09:23:56.816397+00:00`

## Logic Fix

The first mapping used `sigma = 1 / sqrt(H2(p))`, which always stays above 1 for p < 0.5.
That means it always predicts larger code distance improves retention.
QEC has a threshold: distance helps below threshold and hurts above threshold.

This reanalysis uses a train-only threshold estimate and maps:

```text
grad_S = p / p_threshold
sigma = sqrt(p_threshold / p)
Df = surface-code distance
R = physical_error_rate / logical_error_rate_laplace
```

## Threshold Estimate

- Estimated threshold: `0.00707107`
- Method: geometric midpoint between last improving p and first degrading p
- Training distances used: `3` and `5`

| p | mean delta high-low | distance helped |
|---:|---:|---|
| 0.0010 | 2.4656 | True |
| 0.0020 | 0.9621 | True |
| 0.0050 | 0.2506 | True |
| 0.0100 | -0.3880 | False |
| 0.0200 | -0.5507 | False |
| 0.0400 | -0.2486 | False |

## Held-Out Model Comparison

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `threshold_formula_score` | 0.4745 | 0.8802 | 0.8180 | 0.8130 |
| `threshold_formula_components` | 0.4063 | 0.9654 | 0.8455 | 0.7102 |
| `standard_qec_scaling` | 0.4078 | 0.9832 | 0.8590 | 0.7609 |
| `original_formula_components` | 0.4807 | 1.0438 | 0.8128 | 0.7512 |
| `p_only` | 0.5841 | 1.2510 | 0.7511 | 0.6269 |
| `original_formula_score` | 0.5339 | 1.6255 | 0.7811 | 0.4585 |
| `distance_only` | 1.2253 | 2.3607 | 0.0198 | -0.0032 |

## Verdict

Best held-out model by MAE: `threshold_formula_score`.

The corrected threshold-relative Formula mapping won this held-out comparison.

This reanalysis reuses the recorded simulation data; it does not change the raw QEC results.
