# QEC Threshold Mapping Reanalysis

Run id: `independent_error_grid_fixed_threshold_v1`
Source run: `independent_error_grid_v1`
UTC time: `2026-05-13T09:31:28.882766+00:00`

## Logic Fix

The first mapping used `sigma = 1 / sqrt(H2(p))`, which always stays above 1 for p < 0.5.
That means it always predicts larger code distance improves retention.
QEC has a threshold: distance helps below threshold and hurts above threshold.

This reanalysis uses a fixed preregistered threshold and maps:

```text
grad_S = p / p_threshold
sigma = sqrt(p_threshold / p)
Df = surface-code distance
R = physical_error_rate / logical_error_rate_laplace
```

## Threshold Estimate

- Estimated threshold: `0.00707107`
- Method: fixed preregistered threshold supplied by --fixed-threshold
- Training distances used: `3` and `5`

| p | mean delta high-low | distance helped |
|---:|---:|---|
| 0.0008 | 1.6966 | True |
| 0.0015 | 1.7433 | True |
| 0.0030 | 0.5941 | True |
| 0.0060 | 0.0936 | True |
| 0.0120 | -0.4416 | False |
| 0.0240 | -0.5174 | False |
| 0.0480 | -0.1500 | False |

## Held-Out Model Comparison

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `standard_qec_scaling` | 0.3519 | 0.8740 | 0.8988 | 0.8052 |
| `threshold_formula_components` | 0.3561 | 0.9197 | 0.8846 | 0.7466 |
| `threshold_formula_score` | 0.4382 | 0.9628 | 0.8633 | 0.7854 |
| `original_formula_components` | 0.3998 | 0.9946 | 0.8652 | 0.7613 |
| `p_only` | 0.5449 | 1.3232 | 0.7960 | 0.5828 |
| `original_formula_score` | 0.4537 | 1.3555 | 0.8419 | 0.5696 |
| `distance_only` | 1.1900 | 2.3544 | 0.0235 | -0.0020 |

## Verdict

Best held-out model by MAE: `standard_qec_scaling`.

The corrected threshold-relative Formula mapping did not win this held-out comparison.

This reanalysis reuses the recorded simulation data; it does not change the raw QEC results.
