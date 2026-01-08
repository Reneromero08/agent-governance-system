# Question 9: Free Energy Principle (R: 1580)

**STATUS: PARTIALLY ANSWERED**

## Question
Friston's FEP minimizes surprise. Does R track prediction error or model confidence? Are they measuring the same thing differently?

---

## TESTS
`open_questions/q6/`
- `q6_free_energy_test.py` - R vs F correlation, gating efficiency

---

## FINDINGS

### 1. R is Free Energy (via log), not 1/F:
   - Empirically: R vs F correlation: -0.23 (negative as expected)
   - Empirically: R-gating reduces free energy by **97.7%**
   - Analytically (Gaussian family): `log(R) = -F + const` and `R ∝ exp(-F)`

### 2. Least Action confirmed:
   - R-gating is **99.7%** more efficient
   - Ungated action cost: 6.19
   - R-gated action cost: 0.02

### 3. Power law relationship:
   - log(R) vs log(F) correlation: -0.47
   - Suggests R ~ 1/F^0.47

---

## ANSWER

**PARTIALLY:** in a specified likelihood family (Gaussian), `R` is directly equivalent to `exp(-F)` up to a constant; the universal mapping for the full formula across families is not finished.

```
log(R) = -F + const      (Gaussian family)
R ∝ exp(-F)
```

- High R = low free energy = confident prediction = ACT
- Low R = high free energy = surprise expected = DON'T ACT
- R-gating = variational free energy minimization
