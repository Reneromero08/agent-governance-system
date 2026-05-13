# QEC Precision Sweep v2 -- Raw Sweep Report

Run id: `qec_v2_depol`
Noise model: `depol`
UTC time: `2026-05-13T09:57:50.294168+00:00`
Git revision: `bb04bbf7293cfe963da2b9d69311287498b49e31`

## Preregistered Mapping (v2, frozen p_th)

- `E`: `1 - p` (initial single-qubit survival probability)
- `grad_S`: `p / p_th` with `p_th = 0.007071067811865475`
- `sigma`: `sqrt(p_th / p)`
- `Df`: surface-code distance `d`
- `R`: `ln(physical_error_rate / logical_error_rate_laplace)`

Formula score:

```text
R_hat = (E / grad_S) * sigma ** Df
```

## Sweep

- Conditions: `72`
- Total shots: `1440000`
- Distances: `[3, 5, 7, 9]`
- Held-out distances: `[7, 9]`
- Physical error rates: `[0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04]`
- Bases: `['x', 'z']`

## Preliminary Model Comparison (built-in evaluation)

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `formula_score_v2` | 0.4170 | 0.8252 | 0.8736 | 0.8048 |
| `standard_qec_scaling` | 0.3661 | 0.8422 | 0.8951 | 0.8030 |
| `formula_components_v2` | 0.3662 | 0.8603 | 0.8936 | 0.7751 |
| `formula_components_v1` | 0.4274 | 0.9965 | 0.8685 | 0.7737 |
| `p_only` | 0.4929 | 1.2076 | 0.8299 | 0.6088 |
| `formula_score_v1` | 0.5267 | 1.6335 | 0.7973 | 0.4390 |
| `distance_only` | 1.2259 | 2.2734 | 0.0217 | 0.0016 |

## Preliminary Verdict (built-in evaluation only)

Best held-out model by MAE: `formula_score_v2`.

Note: The authoritative evaluation uses the separate evaluator script
(`evaluate_qec_v2.py`) which applies the full pass/fail criteria from
PREREGISTRATION_v2.md, including bootstrap confidence intervals.

## Files

- Raw condition CSV: `results\qec_v2_depol\conditions.csv`
- Full JSON payload: `results\qec_v2_depol\qec_precision_sweep_v2.json`
- This report: `REPORT.md`
