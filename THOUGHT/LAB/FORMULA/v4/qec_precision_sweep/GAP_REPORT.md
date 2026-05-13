# QEC Precision Sweep Gap Report

Date: 2026-05-13

This report summarizes what is wrong or missing after the first QEC precision
sweep, the threshold reanalysis, and the independent frozen-threshold follow-up.

## Current State

The QEC test is meaningful, but not yet a proof of the Formula.

The first mapping failed because it used:

```text
sigma = 1 / sqrt(H2(p))
```

That was logically wrong for QEC because it always predicts that increasing code
distance improves retention. QEC has a threshold: distance helps below threshold
and hurts above threshold.

The corrected threshold-relative mapping was:

```text
grad_S = p / p_threshold
sigma = sqrt(p_threshold / p)
Df = surface-code distance
R = physical_error_rate / logical_error_rate_laplace
```

That mapping captured the threshold sign flip and beat the standard QEC scaling
baseline in the first reanalysis. But when the threshold was frozen and tested
on a new independent error grid, standard QEC scaling won:

```text
standard_qec_scaling          test MAE 0.8740
threshold_formula_components  test MAE 0.9197
threshold_formula_score       test MAE 0.9628
original_formula_components   test MAE 0.9946
original_formula_score        test MAE 1.3555
```

So the current verdict is:

```text
The Formula has a real QEC connection, but the current QEC mapping is not yet
confirmed as superior to established QEC scaling.
```

## What Is Wrong Or Missing

### 1. The Mapping Is Still Underdefined

The lab needs a frozen QEC mapping before more serious runs.

Current open variables:

```text
E       = ?
grad_S  = ?
sigma   = ?
Df      = ?
R       = ?
```

The mapping must be locked before looking at new results. Otherwise each
positive result risks being post hoc.

### 2. `sigma` Is Doing Too Much

The corrected mapping used:

```text
sigma = sqrt(p_threshold / p)
```

This captures threshold behavior, but it may simply restate known QEC scaling in
Formula notation.

Open question:

```text
Does sigma represent correction efficiency, redundancy efficiency, syndrome
information gain, or below-threshold survival multiplier?
```

Until this is resolved, the mapping is useful but not conceptually clean.

### 3. `E` Is Currently Trivial

The first runs used:

```text
E = 1.0
```

That removes one of the Formula's major variables from the test. A stronger QEC
test needs a real operational meaning for `E`.

Candidate meanings:

- initial logical-state preparation fidelity
- circuit integrity
- correction resource budget
- available redundancy/work
- initial coherent information

### 4. `grad_S` May Be The Wrong Entropy Pressure

The corrected run used:

```text
grad_S = p / p_threshold
```

That is plausible but incomplete. QEC may require an entropy pressure term based
on the actual syndrome stream or noise channel instead.

Candidate alternatives:

- syndrome entropy
- detector event density
- logical entropy gradient
- noise-channel entropy
- threshold-relative entropy pressure
- round-to-round syndrome growth

### 5. The Standard Baseline Still Wins Independently

The strongest independent result favored:

```text
standard_qec_scaling
```

This means the Formula mapping is close, but has not yet shown added predictive
power beyond established QEC math.

### 6. Only One Code Family Was Tested

The current tests used Stim rotated surface-code memory circuits.

Needed:

- repetition code as sanity check only
- rotated surface code as main track
- at least one additional stabilizer/code family if practical

The Formula should not be judged from one code family alone.

### 7. Only One Broad Noise Style Was Tested

The current runner used Stim's generated circuit-level noise knobs:

- after-clifford depolarization
- reset flip
- measurement flip
- before-round data depolarization

Needed independent noise models:

- depolarizing-heavy noise
- measurement-heavy noise
- biased noise
- phenomenological noise
- circuit-level mixed noise

### 8. The Target `R` May Be Incomplete

The current target was:

```text
R = physical_error_rate / logical_error_rate_laplace
```

This is defensible, but other targets may better match the Formula:

- logical survival, `1 - p_L`
- log logical suppression
- threshold-normalized suppression
- corrected fidelity
- decay rate across rounds

The next test should explicitly choose one primary `R` and reserve the others
as secondary metrics.

### 9. Low-Error Finite-Shot Artifacts Exist

At low physical error rates, some conditions had zero logical failures.
Laplace smoothing prevented infinities, but the low-error region is still
measurement-limited.

Needed:

- more shots at low `p`
- adaptive stopping by target logical-error count
- confidence intervals on logical failure rates

### 10. No Frozen Next Test Exists Yet

The next run should not be another tweak-after-looking pass.

Before running again, the lab needs a short preregistration with:

- exact mapping
- exact code families
- exact noise models
- exact baselines
- exact train/test split
- exact pass/fail criteria

## Best Next Test

The best next test is a non-adaptive QEC v2 mapping test.

Proposed direction:

```text
E       = initial logical integrity or correction resource budget
grad_S  = syndrome entropy or threshold-relative syndrome pressure
sigma   = correction efficiency per redundancy depth
Df      = code distance / redundancy depth
R       = log logical-error suppression or logical survival
```

Run requirements:

- freeze the mapping before execution
- use at least two noise models
- keep distances `7` and `9` held out
- compare against standard QEC scaling
- report confidence intervals
- treat failure to beat baseline as a real negative result

## Evidence Level

Current evidence level:

```text
Promising but not confirmed.
```

The Formula appears to map naturally onto QEC threshold structure, but the
current implementation has not yet shown robust predictive advantage over
standard QEC scaling.

