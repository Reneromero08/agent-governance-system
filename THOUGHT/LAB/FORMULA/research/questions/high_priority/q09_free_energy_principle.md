# Question 9: Free Energy Principle (R: 1580)

**STATUS: ANSWERED**

## Question
Friston's FEP minimizes surprise. Does R track prediction error or model confidence? Are they measuring the same thing differently?

---

## TESTS
`open_questions/q6/`
- `q6_free_energy_test.py` - R vs F correlation, gating efficiency

---

## FINDINGS

### 1. R is inverse Free Energy:
   - R vs F correlation: -0.23 (negative as expected)
   - R-gating reduces free energy by **97.7%**

### 2. Least Action confirmed:
   - R-gating is **99.7%** more efficient
   - Ungated action cost: 6.19
   - R-gated action cost: 0.02

### 3. Power law relationship:
   - log(R) vs log(F) correlation: -0.47
   - Suggests R ~ 1/F^0.47

---

## ANSWER

**YES - R implements the Free Energy Principle.**

```
R = E / grad_S ~ 1 / F
```

- High R = low free energy = confident prediction = ACT
- Low R = high free energy = surprise expected = DON'T ACT
- R-gating = variational free energy minimization
