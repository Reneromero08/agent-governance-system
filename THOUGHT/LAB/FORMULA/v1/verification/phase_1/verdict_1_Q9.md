# Verdict: Q9 - Free Energy Principle (R=1580)

**Reviewer:** Adversarial skeptic (Phase 1)
**Date:** 2026-02-05
**Files reviewed:**
- `THOUGHT/LAB/FORMULA/questions/high_q09_1580/q09_free_energy_principle.md`
- `THOUGHT/LAB/FORMULA/questions/critical_q01_1800/q01_why_grad_s.md`
- `THOUGHT/LAB/FORMULA/questions/critical_q01_1800/tests/q1_derivation_test.py`
- `THOUGHT/LAB/FORMULA/questions/critical_q06_1650/tests/q6_free_energy_test.py`
- `THOUGHT/LAB/FORMULA/SEMIOTIC_AXIOMS.md`
- `THOUGHT/LAB/FORMULA/GLOSSARY.md`
- `THOUGHT/LAB/FORMULA/SPECIFICATION.md`

---

## Verification Rubric

```
Q09: Free Energy Principle (R=1580)
- Claimed status: ANSWERED
- Proof type: mixed (analytic identity under specific construction + numerical demonstration)
- Logical soundness: GAPS
- Claims match evidence: OVERCLAIMED
- Dependencies satisfied: MISSING [Q1 derivation assumes specific E(z) form not in original formula; Q3 uniqueness claim not verified here]
- Circular reasoning: DETECTED [see Section 2 below]
- Post-hoc fitting: DETECTED [see Section 5 below]
- Recommended status: PARTIAL (identity valid for Gaussian/Laplace under specific E(z) construction; general claim unwarranted)
- Confidence: MEDIUM
- Issues: see detailed analysis below
```

---

## Detailed Analysis

### 1. The log(R) = -F + const Derivation: Line-by-Line Audit

The derivation presented in `q1_derivation_test.py` (Test 1) and `q01_why_grad_s.md` proceeds as follows:

**Step 1.** Define z = |mu - truth| / std (dimensionless error). VALID.

**Step 2.** Write Gaussian free energy: F = z^2/2 + log(std) + const. VALID -- this is the negative log of the Gaussian density (up to additive constant 0.5*log(2*pi)).

**Step 3.** Choose E(z) = exp(-z^2/2). This is the KEY move. See Section 2 below.

**Step 4.** Define R = E(z) / std.

**Step 5.** Then exp(-F) = (1/sqrt(2*pi)) * exp(-z^2/2) / std = (1/sqrt(2*pi)) * R.

**Step 6.** Therefore log(R) = -F + 0.5*log(2*pi). QED.

Steps 4-6 are algebraically correct GIVEN Step 3. The algebra from Step 3 onward is trivially valid -- it is just rewriting the Gaussian density with different labels.

**GAP: Step 3 is not derived; it is imposed.** The original formula defines R = (E / grad_S) * sigma^Df where E is "mean pairwise cosine similarity" (semantic domain), "mutual information" (quantum domain), or other domain-specific quantities. None of these are exp(-z^2/2). The derivation silently replaces the original E with a new E(z) that is chosen specifically to make the identity work. This is not a derivation of the original formula; it is a construction of a new formula designed to match free energy.

### 2. Circularity: E(z) Is Reverse-Engineered

This is the central problem. The argument proceeds:

1. Start with the Gaussian density: p(truth | mu, std) = (1/(std*sqrt(2*pi))) * exp(-z^2/2)
2. Re-label exp(-z^2/2) as "E(z)" and 1/std as "1/grad_S"
3. Declare R = E(z)/grad_S
4. Observe log(R) = -F + const
5. Conclude: "R is free energy"

This is circular. You defined R to be the Gaussian density (up to constant), then showed R equals the Gaussian density (up to constant). Of course it does -- you defined it that way.

The honest statement would be: "IF we define E as the exponential of the negative squared standardized error, THEN R is proportional to the Gaussian likelihood." That is a tautology: E/std is the Gaussian pdf if E is the Gaussian kernel. The real question -- does the ORIGINAL formula's E (cosine similarity, mutual information, etc.) relate to free energy? -- is never addressed.

The GLOSSARY.md defines E in the semantic domain as "mean pairwise cosine similarity of embedding cluster." This is NOT exp(-z^2/2). The derivation introduces a different quantity with the same symbol name and claims the identity transfers.

### 3. Beyond Gaussian: Acknowledged but Overclaimed

The proof does acknowledge the Gaussian assumption. Test 4 in `q1_derivation_test.py` extends to the Laplace family, showing that with the Laplace kernel E(z) = exp(-|z|) and the Laplace scale parameter (MAD), the identity log(R) = -F + const also holds. This is correct and is a legitimate generalization to location-scale families.

However, the claim "any location-scale family" is stated without full proof. The generalization argument is: for any location-scale family, p(x|mu,s) = (1/s)*f((x-mu)/s), so if you define E(z) = f(z) and R = E/s, then R is the density evaluated at truth. This is correct BUT:

- It only works when E(z) is defined AS the kernel of the family, which brings back the circularity (Section 2).
- It does not work for the original semantic/quantum definitions of E.
- The claim "R implements FEP across all location-scale families" (Q9, line 74) conflates the re-labeled formula with the original formula.

### 4. FEP Connection: Structural or Notational?

**NOTATIONAL.** The connection to Friston's Free Energy Principle is a notational relabeling, not a structural mathematical identity. Here is why:

Friston's FEP defines variational free energy as:

```
F = E_q[log q(z) - log p(z, x)]
  = KL[q(z) || p(z)] - E_q[log p(x|z)]
```

where q(z) is a recognition density (an approximate posterior) and p(z,x) is a generative model. This quantity involves:
- A recognition density q(z) that is optimized
- A generative model p(z,x) with specified priors
- Expectations taken over q

The R formula involves:
- A fixed set of observations
- A point-estimate comparison (mean vs truth)
- No recognition density, no generative model, no variational optimization

The SPECIFICATION.md (Proposition 3.2) acknowledges this gap partially by stating the identity requires "the identification E = exp(-E_q[log p(x|z)]) and grad_S = exp(H[q(z)])". But these identifications are not justified anywhere. They are ad-hoc mappings that make the notation match. You can make ANY ratio log(A/B) look like negative free energy by identifying A with the right exponential and B with another.

Specifically:
- E_q[log p(x|z)] is an expectation over a recognition density; it is not a point evaluation of a kernel.
- H[q(z)] is the entropy of a distribution; it is not the standard deviation of observations.

These are fundamentally different mathematical objects.

### 5. Post-Hoc Fitting in the Empirical Tests

The empirical tests in `q6_free_energy_test.py` exhibit clear post-hoc fitting:

**Problem A: compute_R uses a different formula than the one being theorized about.** In the Q6 test, `compute_R` defines E = 1/(1 + |mean - truth|), a hand-crafted heuristic. This is NEITHER the original cosine-similarity E NOR the E(z) = exp(-z^2/2) used in the analytical derivation. Three different definitions of E are used across the project without reconciliation.

**Problem B: The empirical correlation is weak.** The Q9 document reports "R vs F correlation: -0.23". A correlation of -0.23 is very weak. It is then claimed as "negative as expected" rather than acknowledged as evidence of a poor fit. If R truly equaled exp(-F), the correlation between R and F would be strongly negative (near -1 for small F ranges). A correlation of -0.23 is consistent with two loosely related quantities, not an identity.

**Problem C: R-gating "reduces free energy by 97.7%" is misleading.** The test generates 50% "good" observations (near truth) and 50% "bad" observations (far from truth, biased 5-15 units). R-gating filters out bad observations (which have high F). This proves only that a filter correlated with truth-proximity filters out high-surprise observations -- which is trivially true for any reasonable quality measure. It does not demonstrate that R IS free energy.

**Problem D: Least action "99.7% more efficient" is similarly trivial.** Any threshold filter that removes bad data will show this improvement. The result says nothing specific about R vs. any other quality filter (e.g., raw error, SNR, confidence interval width).

### 6. Semantic Free Energy: Metaphor, Not Mathematics

The Q9 document states "R-gating = variational free energy minimization." This is a metaphor. Variational free energy minimization is a specific optimization procedure: you adjust the parameters of q(z) to minimize F with respect to q, holding the generative model fixed. R-gating does not optimize any variational distribution. It applies a threshold to a pre-computed scalar.

Calling R-gating "free energy minimization" is like calling any quality-filtering process "free energy minimization." The term loses all mathematical precision.

The phrase "semantic free energy" is never formally defined with the axiomatic properties that make free energy meaningful in statistical mechanics or variational inference (convexity, connection to partition functions, bound on log-evidence, etc.). It is a label applied to a quantity that superficially resembles free energy.

### 7. Surprise Minimization: Trivially True or Not Proven

The claim "R-maximization = surprise minimization" is either trivially true or unproven, depending on interpretation:

**Trivial interpretation:** Define "surprise" as -log(R). Then R-maximization is surprise-minimization by definition. But this is circular -- you have not shown that YOUR definition of surprise corresponds to Friston's definition (the negative log-evidence, -log p(o)).

**Non-trivial interpretation:** Show that maximizing R is equivalent to minimizing -log p(o) under a specified generative model. This is only demonstrated for the case where R is DEFINED to be the Gaussian density, in which case it is again circular (Section 2).

For the original formula's E (cosine similarity), no one has shown that maximizing E/grad_S minimizes -log p(o) for any well-specified generative model.

### 8. Active Inference Connection

The Q9 document makes passing claims about active inference ("ACT" vs "DON'T ACT"). Active inference in Friston's framework involves expected free energy (a lookahead quantity), policy selection, and precision-weighted prediction errors driving action. The R formula has none of this machinery. The "act/don't act" threshold is a simple binary gate, not an active inference policy. This is SPECULATIVE and OVERCLAIMED.

### 9. Dependencies on Q1

Q9 explicitly states "Q1 proves the mathematical necessity of division by scale" and "Together [Q1, Q3] establish: R implements FEP across all location-scale families."

Q1's "proof" (as analyzed above) uses a re-defined E(z) that is not the original formula's E. Therefore:
- Q1's conclusion is valid only for the re-defined formula, not the original.
- Q9 inherits this limitation.
- The claim that the original R formula "implements FEP" is not established.

Q3's uniqueness proof is referenced but not reviewed here. If Q3 proves R = E/grad_S is unique given axioms A1-A4, that only constrains the functional form, not the interpretation as free energy. Uniqueness of form does not imply identity with free energy unless the axioms themselves encode free energy structure.

### 10. The Power Law Finding Is Unexplained

Q9 reports "log(R) vs log(F) correlation: -0.47, suggests R ~ 1/F^0.47". If R truly equaled exp(-F), the log-log relationship would not be a power law (it would be log(R) = -F, not log(R) = -0.47*log(F)). The fact that the empirical data shows a power law rather than the claimed exponential relationship CONTRADICTS the identity log(R) = -F + const for the formula as implemented in the Q6 test. This inconsistency is reported but not explained or reconciled.

---

## Summary of Issues

| # | Issue | Severity |
|---|-------|----------|
| 1 | E(z) = exp(-z^2/2) is reverse-engineered, not derived from original formula | CRITICAL |
| 2 | Three incompatible definitions of E used across files (cosine similarity, 1/(1+error), exp(-z^2/2)) | CRITICAL |
| 3 | Circularity: identity holds because R was defined to match the Gaussian pdf | HIGH |
| 4 | FEP connection is notational, not structural (no recognition density, no generative model) | HIGH |
| 5 | "Semantic free energy" is undefined as a mathematical object with FEP properties | HIGH |
| 6 | Empirical R-F correlation is -0.23, inconsistent with claimed identity | MEDIUM |
| 7 | Power law finding (-0.47) contradicts the exponential identity | MEDIUM |
| 8 | R-gating efficiency results are trivially true for any quality filter | MEDIUM |
| 9 | Active inference claims are speculative with no formal grounding | LOW |
| 10 | "Any location-scale family" generalization is valid only with re-defined E | MEDIUM |

---

## What Would Fix This

1. **Acknowledge the re-definition.** State clearly: "The original formula's E (cosine similarity) is NOT the same as E(z) = exp(-z^2/2). The free energy identity holds for a specific instantiation of R, not for the general formula."

2. **Show the bridge.** If cosine-similarity E is approximately exp(-z^2/2) under some distributional assumption, prove it. This would be a genuine and interesting result.

3. **Downgrade claims.** Replace "R IS free energy" with "R can be instantiated as a form proportional to the Gaussian likelihood, in which case it relates to free energy." Replace "R-gating = variational free energy minimization" with "R-gating resembles free energy minimization under Gaussian assumptions."

4. **Reconcile the three E definitions.** The Q6 test, the analytical derivation, and the GLOSSARY all use different E. Pick one and be consistent, or explicitly map between them.

5. **Drop or qualify active inference claims.** There is no active inference machinery here.

---

## Final Verdict

The analytical identity log(R) = -F + const is algebraically valid but circular: it holds because R was defined to be the Gaussian density. The connection to Friston's FEP is notational, not structural. The empirical evidence is weak (r = -0.23) and the "efficiency" results are trivially achievable by any quality filter. The claim that the original Living Formula "implements the Free Energy Principle" is an overclaim that conflates a re-labeled formula with the original.

**Recommended status: PARTIAL** -- the identity is valid within a specific Gaussian/Laplace construction but does not extend to the original formula's semantic definitions of E without an unproven bridge.
