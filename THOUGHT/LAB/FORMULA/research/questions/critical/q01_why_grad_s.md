# Question 1: Why grad_S? (R: 1800)

**STATUS: ANSWERED**

## Question
What is the deeper principle behind local dispersion as truth indicator?

## Answer (core principle)
`grad_S` is the local **scale / uncertainty** parameter. Dividing by it converts a "truth compatibility" term into an **evidence density** that is comparable across scales (the normalization constant of location-scale likelihoods).

---

## TESTS
`experiments/open_questions/q1/`
- `q1_why_grad_s_test.py` - alternatives comparison
- `q1_deep_grad_s_test.py` - correlated vs independent
- `q1_adversarial_test.py` - attack vectors
- `q1_essence_is_truth_test.py` - E must be grounded in reality
- `q1_derivation_test.py` - Free Energy / likelihood derivation (exact)
- `q1_definitive_test.py` - axiom-based uniqueness proof

---

## WHAT WE PROVED (AIRTIGHT)

### 1. Division by dispersion is forced (location-scale normalization)
Any location-scale family has the form:

```
p(x | mu, s) = (1/s) * f((x - mu)/s)
```

Let `truth` be the target value and define the dimensionless error:

```
z = |mu - truth| / s
```

If we call the (bounded) shape term `E(z) := f(z)`, then evaluating at the truth gives:

```
p(truth | mu, s) = const * E(z) / s
```

So `E/grad_S` is not a design choice: it is the **likelihood normalization constant** for scale families. `grad_S` is (an estimator of) `s`.

### 2. Exact Free Energy equivalence (Gaussian case)
For Gaussian beliefs:

```
F = z^2/2 + log(s) + const
```

Choose:

```
E(z) = exp(-z^2/2)
R = E(z) / s
```

Then:

```
exp(-F) = const * R
log(R) = -F + const
```

Verified in `experiments/open_questions/q1/q1_derivation_test.py` (log(R) vs -F correlation = 1.0; offset matches 0.5*log(2*pi)).

### 3. Why `std` (and not variance) in 1D
Under unit scaling `x -> kx`:
- `error -> k*error`
- `std -> k*std`
- `z = error/std` is invariant, so `E(z)` is invariant

Therefore:
- `E/std` scales as `1/k` (linear)
- `E/std^2` scales as `1/k^2` (quadratic)

Linear scaling preserves comparability across unit systems and matches 1D likelihood normalization (`1/std`).

### 4. Why `std` vs `MAD` is a modeling choice (not a tie to be "won")
The denominator is always the **scale parameter**, but which scale you use depends on the assumed noise family:
- Gaussian (L2 / quadratic free energy) -> scale = `std`
- Laplace (L1 / absolute free energy) -> scale = `MAD`-like `b`

`q1_derivation_test.py` shows both identities are exact (drop constants) and that mismatching the scale breaks the constant-offset property.

### 5. Precision / SNR interpretation
Since `precision = 1/std^2`:

```
R = E/std = E * sqrt(precision)
```

So `R` is **sqrt-precision-weighted evidence**: confidence helps only when compatibility `E` is high.

---

## KEY CLARIFICATION: dispersion is not truth
Dispersion (`grad_S`) is **confidence**. Truth requires both:
- `E`: compatibility with reality (cannot be computed from agreement alone)
- `grad_S`: local uncertainty (scale)

Dispersion becomes "truth-indicative" only through the normalized evidence density `E/grad_S`.

---

## NOTE ON `sigma^Df`
`sigma^Df` is a separate multiplicative scaling term (fractal depth / domain scaling). It does not change why `grad_S` must appear in the denominator; it changes how resonance is modulated across scale/complexity.

---

## ORIGINAL ANSWER (preserved)

grad_S works because it measures **POTENTIAL SURPRISE**.

```
R = E / grad_S = truth / uncertainty = 1 / (surprise rate)
```

- grad_S = local uncertainty = potential surprise
- High grad_S = unpredictable outcomes = high free energy = don't act
- Low grad_S = predictable outcomes = low free energy = act efficiently

The formula implements the **Free Energy Principle**: minimize surprise.
The formula implements **Least Action**: minimize wasted effort.
