# QEC Precision Sweep v2 -- Frozen Preregistration

Date locked: 2026-05-13
Status: FROZEN. No mapping change permitted after sweep execution.

## 1. Hypothesis

The functional form

```
R = (E / grad_S) * sigma^Df
```

predicts logical-error suppression in surface-code QEC better than standard
QEC scaling baselines, across at least two distinct noise models, when all
parameters are locked before sweeps and no post-hoc remapping occurs.

## 2. Operational Definitions (FROZEN)

Every symbol is defined computationally. No free interpretation remains.

### 2.1 E -- Initial Logical-Qubit Survival

`E` is the probability that a bare single physical qubit survives one error
channel application without an error. This provides a non-trivial E that varies
with the physical error rate.

```
E = 1 - p
```

where `p` is the physical error rate for the condition.

Rationale: v1 used E=1.0, which removed one variable from the formula. Making
E depend on p tests whether the formula's E/grad_S ratio carries information
beyond grad_S alone. This is the simplest non-trivial E that does not require
extra simulation or post-hoc estimation.

### 2.2 grad_S -- Threshold-Relative Entropy Pressure

`grad_S` measures how far the physical error rate is from the code threshold.
It is the dimensionless ratio of physical error rate to the frozen threshold.

```
grad_S = p / p_th
```

where `p_th = 0.007071067811865475` (FROZEN, see Section 2.6).

Rationale: v1 showed that threshold-relative pressure captures the sign flip
(above/below threshold). The raw H2(p) always predicts improvement with
distance, which is wrong for QEC. This mapping is retained from the successful
threshold_reanalysis_v1 and is the strongest known candidate.

### 2.3 sigma -- Threshold-Relative Correction Efficiency

`sigma` measures the error correction efficiency at the given physical error
rate relative to threshold. Below threshold, sigma > 1 and each additional
unit of Df multiplies retention. Above threshold, sigma < 1 and each additional
unit of Df divides retention (correction fails).

```
sigma = sqrt(p_th / p)
```

Rationale: This mapping correctly produces sigma > 1 below threshold and
sigma < 1 above threshold. It is the simplest threshold-sign-aware definition.
The square root reflects the fact that logical error suppression scales as
p^(d/2) in the low-p regime for surface codes, making sigma the per-distance
multiplicative factor.

### 2.4 Df -- Redundancy Depth

`Df` is the code distance: the number of physical qubits along one dimension
of the surface-code lattice, which determines the number of independent
error-correction layers.

```
Df = d
```

where `d` is the surface-code distance (3, 5, 7, or 9).

### 2.5 R -- Log Logical-Error Suppression

`R` is the natural logarithm of the ratio of physical error rate to
Laplace-smoothed logical error rate. This is the observable the formula
aims to predict.

```
R = ln(p / p_L_laplace)
```

where:
- `p_L_laplace = (logical_error_count + 0.5) / (shots + 1.0)`
- `logical_error_count` is the number of shots where the decoder prediction
  mismatches the true observable.

This target is identical to v1 for comparability.

### 2.6 p_th -- Frozen Threshold (CRITICAL)

`p_th = 0.007071067811865475`

This value was estimated from the v1 `full_surface_code_v1` run using ONLY
training distances (3, 5) as the geometric midpoint between the last
distance-improving p (0.005) and the first distance-hurting p (0.01).

**This value is FROZEN and MUST NOT be re-estimated from v2 data.** Using
v2 data to adjust p_th constitutes post-hoc remapping and invalidates the
preregistration.

### 2.7 Formula Score (for regression)

```
log_R_hat = ln(E / grad_S) + Df * ln(sigma)
          = ln((1-p) / (p/p_th)) + d * ln(sqrt(p_th/p))
```

This is the single-feature model `formula_score`.

The decomposition `formula_components` separates:
- Feature 1: `ln(E / grad_S)` = `ln((1-p) * p_th / p)`
- Feature 2: `Df * ln(sigma)` = `d * ln(sqrt(p_th/p))`
- Feature 3: `basis_z` indicator (1.0 for Z-basis, 0.0 for X-basis)

## 3. Noise Models (at least two)

### 3.1 Noise Model A: Depolarizing-Dominant (DEPOL)

Standard circuit-level noise with all error sources set to the same physical
error rate `p`. This is identical to the v1 noise model and serves as the
continuity baseline.

```
after_clifford_depolarization = p
after_reset_flip_probability   = p
before_measure_flip_probability = p
before_round_data_depolarization = p
```

### 3.2 Noise Model B: Measurement-Heavy (MEAS)

Circuit-level noise biased toward measurement operations. Gate errors are
reduced; measurement and reset errors are amplified. This tests whether the
formula generalizes when the error channel is not uniform.

```
after_clifford_depolarization  = p * 0.2
after_reset_flip_probability   = p * 2.0
before_measure_flip_probability = p * 3.0
before_round_data_depolarization = p * 0.5
```

The total error budget per round is approximately equal to DEPOL at the same
p, but concentrated in measurement steps. This changes the effective threshold
and syndrome pattern without changing the formula mapping.

### 3.3 Extended (optional): Phenomenological Noise (PHENOM)

If resources permit, a third noise model with only data and measurement noise
(no gate errors) can be added. The mapping and criteria remain identical.

```
after_clifford_depolarization  = 0.0
after_reset_flip_probability   = p
before_measure_flip_probability = p
before_round_data_depolarization = p
```

This is NOT required for the primary pass/fail verdict. Include only if the
two primary models produce an ambiguous result.

## 4. Code Family

Rotated surface code memory experiment (`surface_code:rotated_memory_x` and
`surface_code:rotated_memory_z`) via Stim. This matches v1 for comparability.

Decoding: PyMatching (minimum-weight perfect matching).

## 5. Physical Error Rates

Nine rates providing dense coverage around the frozen threshold:

```
p ∈ {0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04}
```

This grid places four points below threshold (~0.0071), one near threshold,
and three above. The v1 grid had only three points below threshold, limiting
resolution in the regime where the formula should show the strongest signal.

## 6. Distances and Holdout

```
Training distances: d ∈ {3, 5}
Held-out distances: d ∈ {7, 9}
```

Bases: X and Z (both included per distance/p combination).

Total conditions per noise model: 2 bases * 4 distances * 9 rates = 72

## 7. Shot Budget

```
shots = 20,000 per condition (matching v1)
```

Total per noise model: 72 * 20,000 = 1,440,000 shots
Total for two noise models: 2,880,000 shots

At the lowest p (0.0005), expected logical errors at d=3 are approximately
10 (0.0005 * 20000), giving reasonable statistics.

## 8. Baselines

All models predict `log_suppression = ln(p / p_L_laplace)` and are evaluated
on held-out distances only.

### 8.1 standard_qec_scaling (strongest baseline)

Features: [ln(p), d, d * ln(p), basis_z]

Captures the standard p^d suppression law. This was the strongest model in
both independent v1 tests. It is the primary benchmark the formula must
match or beat.

### 8.2 p_only (weak baseline)

Features: [ln(p), basis_z]

Physical error rate alone. The formula must beat this to show Df carries
predictive value.

### 8.3 distance_only (null baseline)

Features: [d, basis_z]

Code distance alone. Demonstrates that distance without error rate is
insufficient, confirming the threshold structure.

### 8.4 formula_score (primary test)

Features: [log_R_hat] (single feature from Section 2.7)

The formula as a single scalar predictor. If this beats standard_qec_scaling,
the formula's functional form captures QEC structure that the p^d law misses.

### 8.5 formula_components (diagnostic test)

Features: [ln(E/grad_S), Df*ln(sigma), basis_z]

The additive decomposition. If formula_components beats formula_score, the
linear model is learning weights that differ from the formula's 1:1 ratio,
suggesting a systematic miscalibration.

## 9. Evaluation Protocol

All models are trained on distances {3, 5} and evaluated on distances {7, 9}.

Models are scikit-learn pipelines: `StandardScaler -> LinearRegression`.

Metrics (computed on held-out test set only):
- **MAE** (mean absolute error on log_suppression): primary metric
- **R2** (coefficient of determination): secondary metric
- Per-point absolute errors: diagnostic (recorded, not gating)

Training uses all training conditions; test uses all held-out conditions.
No cross-validation, no hyperparameter tuning, no threshold re-estimation.

If the model cannot be fit (e.g., singular matrix), the condition is recorded
as a failure and counts against the formula.

## 10. Pass/Fail Criteria

### 10.1 Primary Verdict (per noise model)

The formula PASSES a noise model if:

1. `formula_score` test MAE <= `standard_qec_scaling` test MAE * 1.05
   (within 5% of the strongest baseline)
2. AND `formula_score` test MAE < `p_only` test MAE
   (strictly better than physical error rate alone)
3. AND `formula_components` test R2 >= 0.0
   (the decomposition is not anti-predictive)

The formula FAILS a noise model if ANY of:

1. `standard_qec_scaling` test MAE < `formula_score` test MAE * 0.90
   (standard QEC is more than 10% better)
2. OR `formula_score` test MAE >= `p_only` test MAE
   (no better than ignoring code distance)
3. OR `distance_only` test MAE <= `formula_score` test MAE
   (distance alone is as good or better; indicates no threshold structure)

**Ambiguous** if formula_score is close (within 10%) but doesn't reach the
5% threshold of standard QEC, or if the two noise models disagree.

### 10.2 Cross-Model Verdict

| DEPOL | MEAS | Overall Verdict |
|-------|------|-----------------|
| PASS | PASS | **CONFIRMED** -- formula robust across noise models |
| PASS | FAIL | **MIXED** -- formula noise-model dependent |
| FAIL | PASS | **MIXED** -- formula noise-model dependent |
| FAIL | FAIL | **NEGATIVE** -- formula does not beat baselines |
| AMBIGUOUS | * | **INCONCLUSIVE** -- more data needed |

### 10.3 What Falsifies the QEC Mapping

The QEC domain mapping is **FALSIFIED** if:

1. `formula_score` test MAE > 1.5 * `standard_qec_scaling` test MAE on BOTH noise models
   (formula is substantially worse than standard QEC)
2. AND `formula_components` test R2 < 0.0 on BOTH noise models
   (decomposed formula components are anti-predictive)
3. AND `p_only` test MAE < `formula_score` test MAE on BOTH noise models
   (physical error rate alone is a better predictor than the full formula)

This is a high bar for falsification. Meeting it means the formula's QEC
mapping has no detectable predictive value beyond the physical error rate.

**Important**: Falsification of this mapping does NOT falsify the entire
Formula. It falsifies this QEC mapping. A different QEC mapping (different
operational definitions of E, grad_S, sigma, Df) could still succeed.

### 10.4 What Confirms the QEC Mapping

The QEC domain mapping is **CONFIRMED** if:

1. PASS on both DEPOL and MEAS noise models
2. AND `formula_score` test R2 >= `standard_qec_scaling` test R2 - 0.03 on both models
3. AND `formula_components` test MAE < `standard_qec_scaling` test MAE on at least one model
   (the decomposed formula strictly beats standard QEC on at least one noise model)

## 11. Post-Registration Discipline

### 11.1 Hard Prohibitions

- **NO** re-estimation of p_th from v2 data
- **NO** change to any operational definition after sweep begins
- **NO** selective reporting (all sweeps, noise models, and distances are reported)
- **NO** dropping of conditions with zero logical errors (Laplace smoothing handles this)
- **NO** model selection by looking at held-out results before choosing mapping

### 11.2 Allowed Actions

- Fixing bugs in simulation code (document all changes)
- Adding the optional PHENOM noise model (does not retroactively change verdict)
- Running additional shots at low p if Laplace smoothing is insufficient
  (must use identical seed offset; report original and augmented results)
- Computing confidence intervals via bootstrapping (diagnostic only, not gating)

### 11.3 Report Requirements

The final report MUST include:

- Exact command-line invocation with all arguments
- Git revision at run time
- Python environment (pip freeze or equivalent)
- Full condition table (CSV) with per-condition results
- Held-out model comparison table
- Per-point prediction vs actual for held-out distances
- Confidence intervals on test MAE (bootstrap, 1000 resamples)
- Verdict table per Section 10.2
- SHA256 hashes of all output files
- Any deviations from this preregistration, with justification

## 12. Anticipated Outcomes

### 12.1 Why the formula might win

The threshold-relative mapping already demonstrated competitive performance
(threshold_reanalysis_v1: test MAE 0.880 vs 0.983 for standard QEC). The v2
changes (non-trivial E, denser p-grid, two noise models) address the specific
weaknesses identified in GAP_REPORT.md.

### 12.2 Why the formula might lose

The independent frozen-threshold test (independent_error_grid_fixed_threshold_v1)
showed standard QEC winning (MAE 0.874 vs 0.920). The v2 mapping may simply be
restating known QEC scaling in formula notation. If formula_score cannot beat
standard_qec_scaling on DEPOL (which is similar to v1 conditions), the mapping
is likely capturing the same underlying p^d structure without adding predictive
power.

### 12.3 Why the result might be inconclusive

The measurement-heavy noise model (MEAS) shifts the effective threshold. The
frozen p_th = 0.0071 was estimated under DEPOL conditions. If the MEAS
threshold differs substantially, the formula may perform poorly on MEAS even
if it succeeds on DEPOL. This would suggest the mapping is noise-model
dependent but not necessarily wrong.

## 13. Execution Plan

### Step 1: DEPOL noise model sweep

```
python THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/code/run_qec_precision_sweep_v2.py \
  --shots 20000 --distances 3,5,7,9 --heldout-distances 7,9 \
  --physical-error-rates 0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.04 \
  --bases x,z --seed 20260514 --noise-model depol --run-id qec_v2_depol
```

### Step 2: MEAS noise model sweep

```
python THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/code/run_qec_precision_sweep_v2.py \
  --shots 20000 --distances 3,5,7,9 --heldout-distances 7,9 \
  --physical-error-rates 0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.04 \
  --bases x,z --seed 20260515 --noise-model meas --run-id qec_v2_meas
```

### Step 3: Evaluate with frozen p_th

```
python THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/code/evaluate_qec_v2.py \
  --source-dir results/qec_v2_depol --run-id qec_v2_depol_eval \
  --frozen-threshold 0.007071067811865475

python THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/code/evaluate_qec_v2.py \
  --source-dir results/qec_v2_meas --run-id qec_v2_meas_eval \
  --frozen-threshold 0.007071067811865475
```

### Step 4: Cross-model report

Generate unified report comparing DEPOL and MEAS verdicts per Section 10.2.

## 14. Relationship to v1

This preregistration supersedes the v1 mapping. The v1 results remain part of
the evidence record. The v2 mapping differs from v1 in:

| Aspect | v1 | v2 |
|--------|----|----|
| E | Fixed at 1.0 | `1 - p` (non-trivial) |
| grad_S | H2(p) then p/p_th | `p/p_th` (locked from start) |
| sigma | 1/sqrt(H2(p)) then sqrt(p_th/p) | `sqrt(p_th/p)` (locked from start) |
| p_th | Estimated adaptively in reanalysis | Frozen from v1 training data |
| Noise models | 1 (uniform circuit-level) | 2 (DEPOL + MEAS) |
| p-grid | 6 rates | 9 rates (denser near threshold) |
| pass/fail | Post-hoc interpretation | Preregistered criteria |
