# Q09: R Connects to the Free Energy Principle

## Hypothesis

R is formally connected to Friston's Free Energy Principle (FEP). Specifically: log(R) = -F + const, where F is the variational free energy. R-maximization is therefore equivalent to surprise minimization. This is not merely a notational resemblance but a structural mathematical identity: R implements the FEP across all location-scale likelihood families, and R-gating constitutes a form of variational free energy minimization.

## v1 Evidence Summary

Tests from Q1 and Q6 were used:

1. **q1_derivation_test.py (Test 1)** -- Gaussian case: defined E(z) = exp(-z^2/2), R = E/std, F = z^2/2 + log(std) + const. Verified log(R) vs -F correlation = 1.0 with offset matching 0.5*log(2*pi).

2. **q1_derivation_test.py (Test 4)** -- Family-scoped equivalence:
   - Gaussian: std(log(R_std) + F_gauss) approximately 0 (identity holds).
   - Laplace: std(log(R_mad) + F_laplace) approximately 0 (identity holds).
   - Mismatch: std(log(R_mad_mismatch) + F_gauss) > 0 (fails as expected).

3. **q6_free_energy_test.py** -- Empirical test with yet another E definition (E = 1/(1+|mean-truth|)):
   - R vs F correlation: -0.23 (weak negative).
   - R-gating reduces free energy by 97.7%.
   - R-gating is 99.7% more efficient (ungated cost 6.19, gated cost 0.02).
   - log(R) vs log(F) correlation: -0.47, suggesting R ~ 1/F^0.47.

## v1 Methodology Problems

The verification identified the following issues:

1. **E(z) is reverse-engineered, not derived (CRITICAL).** The derivation defines E(z) = exp(-z^2/2) -- which IS the Gaussian kernel -- then shows R = E/std equals the Gaussian likelihood. This is algebraically trivial: log(Gaussian likelihood) = -(Gaussian negative log-likelihood). The "identity" was guaranteed by how E was defined.

2. **Three incompatible E definitions used (CRITICAL).** Analytical derivation: E = exp(-z^2/2). Empirical Q6 test: E = 1/(1+|mean-truth|). GLOSSARY: E = mean pairwise cosine similarity. Results from one E do not apply to the other. The real question -- does the operational E (cosine similarity) relate to free energy? -- is never tested.

3. **FEP connection is notational, not structural (HIGH).** Friston's FEP involves a recognition density q(z), a generative model p(z,x), variational optimization, and expectations over q. The R formula involves fixed observations, a point-estimate comparison, no recognition density, no generative model, and no variational optimization. Sharing the form "log(something) = -F" does not make them the same theory.

4. **Empirical R-F correlation is weak (MEDIUM).** The Q6 test shows R vs F correlation = -0.23. If R truly equaled exp(-F), this correlation should be strongly negative. The weak correlation is consistent with loosely related quantities, not an identity.

5. **Power law contradicts the identity (MEDIUM).** log(R) vs log(F) correlation = -0.47 suggests R ~ 1/F^0.47. But if log(R) = -F + const, the log-log relationship would NOT be a power law. This inconsistency is reported but never reconciled.

6. **R-gating efficiency is trivially true (MEDIUM).** Any threshold filter correlated with truth-proximity will reduce free energy when it filters out bad observations. The 97.7% reduction is not specific to R.

7. **Active inference claims are speculative (LOW).** Active inference involves expected free energy, policy selection, and precision-weighted prediction errors. R has none of this machinery.

## v2 Test Plan

### Test 1: Test the Connection with Operational E (Cosine Similarity)
- Compute R using the actual operational E (mean pairwise cosine similarity) on real embedding clusters.
- Compute variational free energy F under a Gaussian generative model for the same clusters.
- Measure the correlation between log(R_operational) and -F.
- If the correlation is weak (< 0.5), the FEP connection does not hold for the actual formula.

### Test 2: Structural FEP Requirements
Test whether R satisfies the structural requirements of a free energy functional:
- **Upper bound on surprise:** F >= -log p(o). Does -log(R) upper-bound the negative log-evidence? Measure on real data.
- **Tightness:** Does minimizing -log(R) approach minimizing surprise? Compare against actual variational inference on the same data.
- **Recognition density:** Can R be interpreted as arising from an implicit recognition density? Derive what q(z) would have to be and test whether it is a valid probability distribution.

### Test 3: R vs. Proper Variational Inference
- On a Bayesian inference task with known ground truth (e.g., Bayesian linear regression on UCI datasets):
  - Compute R for the data.
  - Run actual variational inference (mean-field or full-rank Gaussian VI).
  - Compare R-gating decisions against VI-based decisions.
  - If R-gating matches or approximates VI decisions, the connection is meaningful.
  - If R-gating is uncorrelated with VI decisions, the connection is notational only.

### Test 4: Beyond Location-Scale
- Test the R-F connection on non-location-scale families:
  - Poisson data (count data, no scale parameter).
  - Multinomial data (categorical, no scale parameter).
  - Beta-distributed data (bounded, non-symmetric in general).
- For each, compute R with domain-appropriate E and compare against the family's free energy.
- The hypothesis predicts the connection should generalize; failure here limits the scope.

### Test 5: Comparison Against Other Quality Filters
- On the R-gating efficiency test, compare R against:
  - Oracle filter (knows ground truth).
  - Raw error filter (|mean - truth| < threshold -- requires truth but sets the ceiling).
  - Variance filter (1/var > threshold -- does not require truth).
  - Confidence interval filter (CI width < threshold).
  - Random filter (baseline).
- Measure free energy reduction for each. If R does not substantially outperform the variance filter, the FEP connection adds no practical value.

## Required Data

- **STS-B / MTEB** -- for computing operational E on real embedding clusters
- **UCI regression datasets** (Boston Housing, Concrete, Energy Efficiency) -- for Bayesian inference comparison
- **Sentence-transformers models** (all-MiniLM-L6-v2, all-mpnet-base-v2) -- for embedding generation
- **Count data datasets** (e.g., bike sharing demand, insurance claims) -- for Poisson domain testing
- **Categorical data datasets** (e.g., UCI mushroom, adult census) -- for multinomial testing

## Pre-Registered Criteria

- **Success (confirm):** Correlation between log(R_operational) and -F exceeds 0.7 on real embedding data. AND -log(R) is a valid upper bound on surprise for at least 80% of test clusters. AND R-gating decisions match variational inference decisions with Cohen's kappa > 0.6. AND R-gating outperforms the simple variance filter (1/var) by at least 10% in free energy reduction.
- **Failure (falsify):** Correlation between log(R_operational) and -F is below 0.3 on real data, OR -log(R) fails as an upper bound on surprise for more than 50% of clusters, OR R-gating is no better than a variance filter (within 5% free energy reduction), OR R-gating decisions are uncorrelated with VI decisions (kappa < 0.2).
- **Inconclusive:** Moderate correlation (0.3-0.7) between log(R) and -F; upper bound property holds sometimes; R-gating slightly outperforms variance filter but not dramatically.

## Baseline Comparisons

R's FEP connection must be demonstrated to provide value beyond:
- Simple inverse variance (1/sigma^2) as a precision measure
- Negative log-likelihood evaluated at the mean
- Evidence Lower BOund (ELBO) from actual variational inference
- Bayesian Information Criterion (BIC)
- Akaike Information Criterion (AIC)
- Minimum Description Length (MDL)

## Salvageable from v1

- **q1_derivation_test.py (Test 4)** -- The family-scoped equivalence test (Gaussian and Laplace) is algebraically valid within stated scope. The code structure for testing multiple likelihood families is reusable. Path: `v1/questions/critical_q01_1800/tests/q1_derivation_test.py`
- **The family-scoping insight** -- The observation that the identity requires matching the scale parameter to the likelihood family (std for Gaussian, MAD for Laplace) is a genuine contribution. Path: `v1/questions/high_q09_1580/q09_free_energy_principle.md`
- **q6_free_energy_test.py** -- The test framework for comparing R-gating against ungated operation is reusable, though E must be replaced with operational E and baselines must be added. Path: `v1/questions/critical_q06_1650/tests/q6_free_energy_test.py`
