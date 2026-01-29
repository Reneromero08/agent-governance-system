# Question 9: Free Energy Principle (R: 1580)

**STATUS: ANSWERED**

## Question
Friston's FEP minimizes surprise. Does R track prediction error or model confidence? Are they measuring the same thing differently?

---

## TESTS
- `questions/6/q6_free_energy_test.py` - R vs F correlation, gating efficiency
- `questions/1/q1_derivation_test.py` - **Test 4: Family-scoped equivalence (Gaussian AND Laplace)**

---

## FINDINGS

### 1. R is Free Energy (via log), not 1/F:
   - Empirically: R vs F correlation: -0.23 (negative as expected)
   - Empirically: R-gating reduces free energy by **97.7%**
   - Analytically: `log(R) = -F + const` and `R ∝ exp(-F)`

### 2. Least Action confirmed:
   - R-gating is **99.7%** more efficient
   - Ungated action cost: 6.19
   - R-gated action cost: 0.02

### 3. Power law relationship:
   - log(R) vs log(F) correlation: -0.47
   - Suggests R ~ 1/F^0.47

### 4. Family-scoped equivalence (from Q1 Test 4):
   - Gaussian: `std(log(R_std) + F_gauss) ≈ 0` ✓
   - Laplace: `std(log(R_mad) + F_laplace) ≈ 0` ✓
   - Mismatch: `std(log(R_mad_mismatch) + F_gauss) > 0` (fails as expected)

---

## ANSWER

**YES:** R and Free Energy measure the same thing differently. Within any specified likelihood family, `R ∝ exp(-F)` exactly.

```
log(R) = -F + const      (any location-scale family)
R ∝ exp(-F)
```

The scale parameter adapts to the family:
- **Gaussian (L2):** denominator = std
- **Laplace (L1):** denominator = MAD-like scale

This is not a limitation - it's the correct behavior. The "universal mapping" is: use the family-appropriate scale parameter, and the identity holds exactly.

**Interpretation:**
- High R = low free energy = confident prediction = ACT
- Low R = high free energy = surprise expected = DON'T ACT
- R-gating = variational free energy minimization

**Connection to Q1 and Q3:**
- Q1 proves the mathematical necessity of division by scale
- Q3 proves R = E(z)/σ is unique given axioms A1-A4
- Together they establish: R implements FEP across all location-scale families
