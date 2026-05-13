# QEC Precision Sweep v2 -- Raw Sweep Report

Run id: `qec_v2_meas`
Noise model: `meas`
UTC time: `2026-05-13T09:57:51.572647+00:00`
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
| `standard_qec_scaling` | 0.2201 | 1.0852 | 0.9678 | 0.4992 |
| `formula_components_v1` | 0.2795 | 1.2082 | 0.9496 | 0.5865 |
| `formula_score_v1` | 0.2961 | 1.3988 | 0.9471 | 0.4792 |
| `formula_score_v2` | 0.4701 | 1.4811 | 0.8773 | 0.4897 |
| `p_only` | 0.5933 | 1.5707 | 0.7962 | 0.3107 |
| `formula_components_v2` | 0.4122 | 1.8363 | 0.9103 | 0.2519 |
| `distance_only` | 1.2694 | 2.1183 | 0.1043 | -0.0676 |

## Preliminary Verdict (built-in evaluation only)

Best held-out model by MAE: `standard_qec_scaling`.

Note: The authoritative evaluation uses the separate evaluator script
(`evaluate_qec_v2.py`) which applies the full pass/fail criteria from
PREREGISTRATION_v2.md, including bootstrap confidence intervals.

## Files

- Raw condition CSV: `results\qec_v2_meas\conditions.csv`
- Full JSON payload: `results\qec_v2_meas\qec_precision_sweep_v2.json`
- This report: `REPORT.md`
