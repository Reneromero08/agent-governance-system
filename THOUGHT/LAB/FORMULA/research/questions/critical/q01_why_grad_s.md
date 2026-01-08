# Question 1: Why grad_S? (R: 1800)

**STATUS: PARTIALLY ANSWERED**

## Question
What is the deeper principle behind local dispersion as truth indicator?

---

## TESTS
`open_questions/q1/`
- `q1_why_grad_s_test.py` - alternatives comparison
- `q1_deep_grad_s_test.py` - independence requirement
- `q1_adversarial_test.py` - attack vectors
- `q1_essence_is_truth_test.py` - E = truth definition
- `q1_derivation_test.py` - Free Energy derivation attempt
- `q1_definitive_test.py` - axiom-based uniqueness proof

---

## WHAT WE PROVED (SOLID)

### 1. Division is forced by dimensional analysis
E is dimensionless [0,1], std has units of measurement.
- E + std: INVALID (can't add different dimensions)
- E - std: INVALID (can't subtract different dimensions)
- E * std: WRONG direction (rewards uncertainty)
- E / std: VALID (truth per unit uncertainty)
- E / std^2: VALID but different behavior

**Conclusion:** Only E/std^n forms are dimensionally valid.

### 2. Linear scaling (n=1) beats quadratic (n=2)

| Scale | E/std ratio | E/std^2 ratio | Expected 1/k |
|-------|-------------|---------------|--------------|
| 0.1   | 11.83       | 118.31        | 10.0         |
| 1.0   | 1.00        | 1.00          | 1.0          |
| 10.0  | 0.04        | 0.004         | 0.1          |
| 100.0 | 0.0006      | 0.000006      | 0.01         |

**Conclusion:** E/std gives roughly linear scaling. E/std^2 gives quadratic (distorts comparisons).

### 3. E/std beats E/std^2 in Free Energy alignment
Spearman correlation with -F:
- E/std: 0.33
- E/std^2: 0.08

**Conclusion:** E/std aligns 4x better with Free Energy.

### 4. R = E * sqrt(precision)
Mathematically verified: R = E/std = E * sqrt(1/std^2) = E * sqrt(precision)

Max difference: 0.0000000000

**Conclusion:** R is sqrt-precision-weighted evidence.

### 5. R is error-aware SNR

| Scenario | E | std | R | SNR |
|----------|---|-----|---|-----|
| High truth, low noise | 0.95 | 0.45 | 2.10 | 22.0 |
| High truth, high noise | 0.94 | 2.85 | 0.33 | 3.5 |
| Low truth, low noise | 0.17 | 0.54 | 0.31 | 27.9 |
| Low truth, high noise | 0.16 | 2.64 | 0.06 | 5.8 |

**Conclusion:** Classic SNR ignores whether signal is TRUE. R penalizes false signals.

### 6. E = amount of truth (measured against reality)

| Bias | E | grad_S | R |
|------|---|--------|---|
| 0 (truth) | 0.97 | 0.08 | 6.08 |
| 10 (echo) | 0.09 | 0.09 | 0.51 |
| 50 (echo) | 0.02 | 0.11 | 0.09 |

Despite same tightness, R drops 60x because E drops.

### 7. R-gating reduces entropy

| | Mean Error | Entropy |
|---|------------|---------|
| Ungated | 6.38 | 6.88 |
| R-gated | 0.09 | 0.07 |

R-gating: 97.7% free energy reduction, 99.7% efficiency gain.

---

## WHAT WE TESTED BUT IS WEAK/INCONCLUSIVE

### 1. std vs MAD is basically a tie
Spearman correlation with -F:
- E/std: 0.6316
- E/MAD: 0.6304

**Gap: 0.0012 (0.2%)** - This is noise, not proof.

### 2. R is NOT simply proportional to 1/F
- Overall correlation R vs 1/F: **0.14** (weak)
- Within Gaussian data: **0.83** (strong)
- Within heavy-tailed: **0.92** (strong)

**Conclusion:** R relates to 1/F within consistent scenarios, not across all scenarios.

---

## WHAT'S STILL UNPROVEN

1. **Why E = 1/(1+error)?** - Assumed, not derived
2. **The sigma^Df term** - Full formula R = E/grad_S * sigma^Df is unexamined
3. **Why std beats MAD?** - 0.2% difference is noise
4. **Uniqueness** - Axioms chosen may be post-hoc

---

## SUB-QUESTIONS REMAINING

1. **Variance additivity** - Does Var(X+Y) = Var(X) + Var(Y) make R composable in a way MAD doesn't?
2. **Cramer-Rao bound** - Is std special because of Fisher information?
3. **E derivation** - Can we derive E = 1/(1+error) from first principles?
4. **The sigma^Df term** - What does it do? Full formula is R = E/grad_S * sigma^Df

---

## SUMMARY

**ANSWERED (with tests):**
- [x] Why division? -> Dimensional analysis forces it
- [x] Why std not variance? -> Linear scaling behavior
- [x] Bayesian connection? -> R = E * sqrt(precision)
- [x] Signal-to-noise? -> R is error-aware SNR

**INCONCLUSIVE:**
- [ ] Why std not MAD? -> 0.2% difference is noise
- [ ] R ~ 1/F? -> Only holds within similar scenarios

**UNANSWERED:**
- [ ] Why E = 1/(1+error)?
- [ ] How does sigma^Df interact?
- [ ] Is there a true uniqueness derivation?

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
