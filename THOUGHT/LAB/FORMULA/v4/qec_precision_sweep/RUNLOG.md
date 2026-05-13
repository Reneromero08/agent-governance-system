# QEC Precision Sweep Runlog

## Run: full_surface_code_v1

Date: 2026-05-13

Purpose: test whether the v4 Formula mapping predicts quantum error correction
logical-error suppression in a real surface-code simulator.

## Environment

- Python: 3.11.6
- Platform: Windows-10-10.0.26100-SP0
- `stim`: 1.15.0
- `pymatching`: 2.3.1
- `sinter`: 1.15.0
- `numpy`: 1.26.4
- `scipy`: 1.14.0
- `sklearn`: 1.8.0
- Git revision at run start: `bb04bbf7293cfe963da2b9d69311287498b49e31`

## Dependency Install

Command:

```powershell
py -3.11 -m pip install stim pymatching sinter
```

Install result: succeeded. Pip reported existing environment conflicts for
`convokit` expectations around TensorFlow, NumPy, and SciPy. Those conflicts
do not affect this QEC runner, which uses `stim`, `pymatching`, `sinter`,
`numpy`, `scipy`, and `sklearn`.

## Smoke Check

Command:

```powershell
py -3.11 THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\code\run_qec_precision_sweep.py --shots 100 --distances 3,5 --heldout-distances 5 --physical-error-rates 0.005,0.02 --bases x --run-id smoke_api_check
```

Result: passed and wrote smoke artifacts under `results/smoke_api_check/`.

## Full Command

```powershell
py -3.11 THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\code\run_qec_precision_sweep.py --shots 20000 --distances 3,5,7,9 --heldout-distances 7,9 --physical-error-rates 0.001,0.002,0.005,0.01,0.02,0.04 --bases x,z --seed 20260513 --run-id full_surface_code_v1 2>&1 | Tee-Object -FilePath THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\results\full_surface_code_v1_execution.log
```

Result: completed successfully.

## Preregistered Mapping

- `R`: logical-error suppression, `physical_error_rate / logical_error_rate_laplace`
- `E`: normalized initial logical-state integrity, fixed at `1.0`
- `grad_S`: binary entropy of the physical error rate, `H2(p)`
- `sigma`: entropy-to-correction efficiency proxy, `1 / sqrt(H2(p))`
- `Df`: surface-code distance `d`

Formula score:

```text
R_hat = (E / grad_S) * sigma ** Df
```

## Sweep Scope

- Conditions: 48
- Total shots: 960,000
- Codes: `surface_code:rotated_memory_x`, `surface_code:rotated_memory_z`
- Distances: 3, 5, 7, 9
- Training distances: 3, 5
- Held-out distances: 7, 9
- Physical error rates: 0.001, 0.002, 0.005, 0.01, 0.02, 0.04

## Result

Held-out model comparison:

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `standard_qec_scaling` | 0.4078 | 0.9832 | 0.8590 | 0.7609 |
| `formula_components` | 0.4807 | 1.0438 | 0.8128 | 0.7512 |
| `p_only` | 0.5841 | 1.2510 | 0.7511 | 0.6269 |
| `formula_score` | 0.5339 | 1.6255 | 0.7811 | 0.4585 |
| `distance_only` | 1.2253 | 2.3607 | 0.0198 | -0.0032 |

Verdict: the preregistered formula mapping did not beat the strongest held-out
baseline. The richer formula-components model was close to standard QEC scaling,
but the single formula score was materially weaker on held-out distances.

Interpretation: this does not prove or disprove the whole Formula. It is a
negative result for this first QEC domain mapping. The next useful move is to
freeze a better QEC-specific mapping before running a second sweep, especially
around whether `grad_S` should be physical entropy, syndrome entropy, or a
threshold-relative entropy pressure term.

## Artifacts

- `code/run_qec_precision_sweep.py`
- `results/smoke_api_check/`
- `results/full_surface_code_v1/conditions.csv`
- `results/full_surface_code_v1/qec_precision_sweep.json`
- `results/full_surface_code_v1/REPORT.md`
- `results/full_surface_code_v1_execution.log`

## SHA256

```text
CF6EACCA8ADED1830C8869207F3932D8D3DF1197A4173F42EBCC8F2D80F2B283  results/full_surface_code_v1/qec_precision_sweep.json
98E5ED346DEADF565D4F5D30FC96D52FC3FC74DAAE9E9654DBC25EF89D56F140  results/full_surface_code_v1/conditions.csv
9B7BB891E8E418A103B18E6C88A1374FFF623EA1FEE501EE83BD3756A8FA2CCC  results/full_surface_code_v1/REPORT.md
EEE5EE4ADB56482C953224ACDCCB2972820CEC7A147D3C0939D1D9BC0F9A4CA9  results/full_surface_code_v1_execution.log
```

## Logic Audit and Reanalysis: threshold_reanalysis_v1

Date: 2026-05-13

The first run exposed a mapping error, not a simulator error. The original
mapping used:

```text
sigma = 1 / sqrt(H2(p))
```

For every tested physical error rate, this keeps `sigma > 1`, so larger `Df`
always increases the predicted Formula score. That is not valid QEC logic. In
surface-code QEC, larger code distance helps below threshold and hurts above
threshold. The recorded raw data showed exactly that:

- distance helped at `p = 0.001, 0.002, 0.005`
- distance hurt at `p = 0.01, 0.02, 0.04`

I added `code/reanalyze_threshold_mapping.py`, which reuses the raw simulation
data and estimates a threshold from training distances only. It does not use
the held-out distances to choose the threshold.

Reanalysis command:

```powershell
py -3.11 THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\code\reanalyze_threshold_mapping.py --source-json THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\results\full_surface_code_v1\qec_precision_sweep.json --run-id threshold_reanalysis_v1 2>&1 | Tee-Object -FilePath THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\results\threshold_reanalysis_v1_execution.log
```

Corrected threshold-relative mapping:

```text
grad_S = p / p_threshold
sigma = sqrt(p_threshold / p)
Df = surface-code distance
R = physical_error_rate / logical_error_rate_laplace
```

Train-only threshold estimate:

```text
p_threshold = sqrt(0.005 * 0.01) = 0.007071067811865475
```

Held-out model comparison:

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `threshold_formula_score` | 0.4745 | 0.8802 | 0.8180 | 0.8130 |
| `threshold_formula_components` | 0.4063 | 0.9654 | 0.8455 | 0.7102 |
| `standard_qec_scaling` | 0.4078 | 0.9832 | 0.8590 | 0.7609 |
| `original_formula_components` | 0.4807 | 1.0438 | 0.8128 | 0.7512 |
| `p_only` | 0.5841 | 1.2510 | 0.7511 | 0.6269 |
| `original_formula_score` | 0.5339 | 1.6255 | 0.7811 | 0.4585 |
| `distance_only` | 1.2253 | 2.3607 | 0.0198 | -0.0032 |

Updated verdict: the corrected threshold-relative Formula mapping won this
held-out comparison. This is a real positive result for the QEC mapping, but it
is still not final proof. The threshold was inferred from training data, so the
next stronger test must freeze this mapping and threshold-estimation rule before
running a new independent sweep with different noise models and error rates.

Reanalysis SHA256:

```text
685BCA69DAAC73D25453F68BB99809DC7A628E21B28D0F79246D962A43110494  results/threshold_reanalysis_v1/threshold_reanalysis.json
668F87C0E2ACFB3ADE057E8D20E28C76472B7545D86242A0A31DEB148F7547A1  results/threshold_reanalysis_v1/REPORT.md
2CF376D959DEB0DE3044233F542B4459EB876CC75B8D1179FC4205F5CF2F0860  results/threshold_reanalysis_v1_execution.log
```

## Logic Audit 2 and Independent Error Grid: independent_error_grid_v1

Date: 2026-05-13

Additional logic checks:

1. The first corrected reanalysis was still partially adaptive because the
   threshold was estimated from the same run's training distances.
2. A stronger next step must freeze the threshold-estimation output before
   testing a new error grid.
3. The simulator data itself should not be rerun with the same physical error
   rates only; otherwise the held-out test remains too close to the first grid.

I added `--fixed-threshold` to `code/reanalyze_threshold_mapping.py` and froze
the prior threshold:

```text
p_threshold = 0.007071067811865475
```

Independent raw sweep command:

```powershell
py -3.11 THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\code\run_qec_precision_sweep.py --shots 20000 --distances 3,5,7,9 --heldout-distances 7,9 --physical-error-rates 0.00075,0.0015,0.003,0.006,0.012,0.024,0.048 --bases x,z --seed 20260514 --run-id independent_error_grid_v1 2>&1 | Tee-Object -FilePath THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\results\independent_error_grid_v1_execution.log
```

Scope:

- Conditions: 56
- Total shots: 1,120,000
- Codes: `surface_code:rotated_memory_x`, `surface_code:rotated_memory_z`
- Distances: 3, 5, 7, 9
- Training distances: 3, 5
- Held-out distances: 7, 9
- Physical error rates: 0.00075, 0.0015, 0.003, 0.006, 0.012, 0.024, 0.048

Fixed-threshold reanalysis command:

```powershell
py -3.11 THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\code\reanalyze_threshold_mapping.py --source-json THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\results\independent_error_grid_v1\qec_precision_sweep.json --run-id independent_error_grid_fixed_threshold_v1 --fixed-threshold 0.007071067811865475 2>&1 | Tee-Object -FilePath THOUGHT\LAB\FORMULA\v4\qec_precision_sweep\results\independent_error_grid_fixed_threshold_v1_execution.log
```

Held-out model comparison:

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `standard_qec_scaling` | 0.3519 | 0.8740 | 0.8988 | 0.8052 |
| `threshold_formula_components` | 0.3561 | 0.9197 | 0.8846 | 0.7466 |
| `threshold_formula_score` | 0.4382 | 0.9628 | 0.8633 | 0.7854 |
| `original_formula_components` | 0.3998 | 0.9946 | 0.8652 | 0.7613 |
| `p_only` | 0.5449 | 1.3232 | 0.7960 | 0.5828 |
| `original_formula_score` | 0.4537 | 1.3555 | 0.8419 | 0.5696 |
| `distance_only` | 1.1900 | 2.3544 | 0.0235 | -0.0020 |

Updated verdict: the corrected threshold-relative Formula mapping remains much
better than the original mapping, and it captures the QEC threshold sign flip.
However, on the independent error grid with the threshold frozen, it did not
beat the stronger standard QEC scaling baseline. This weakens the first positive
reanalysis. The current evidence is mixed: the domain mapping is meaningful, but
not yet proven superior to established QEC scaling.

Independent-run SHA256:

```text
193AEC4874508015D23B5BA471F5EED3BBBE6EA08343DC23BC38498090E8C5E9  results/independent_error_grid_v1/qec_precision_sweep.json
BFD74AD84E5F106AF33E9F4BC5921CC1E76BFC3D2FB01725AE962494FC8824C8  results/independent_error_grid_v1/conditions.csv
34EABF102CDC00E708C8C04DD5DCFA81CBD07BE49393772E94F3C0D42033F1B0  results/independent_error_grid_v1_execution.log
0CF39316614E8D0F629B4AF86C7C93E4E93DECCD6A7F97A28EF46221CDFC78B3  results/independent_error_grid_fixed_threshold_v1/threshold_reanalysis.json
82EE20E7293552F80D7986DCC14626538C7DEF59A8D4ACB6F271989C9F8A59B9  results/independent_error_grid_fixed_threshold_v1/REPORT.md
770ABDCC0BC8B6C951F325B5B32702F4DD93BEAD2ABEAF9F732B1DCC5C8B99B9  results/independent_error_grid_fixed_threshold_v1_execution.log
```
