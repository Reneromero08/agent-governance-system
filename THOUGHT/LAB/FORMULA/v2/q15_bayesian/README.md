# Q15: R Has a Bayesian Interpretation

## Hypothesis

R has a formal connection to Bayesian inference. Specifically: R is connected to likelihood precision (an intensive quantity, like temperature) rather than posterior concentration (an extensive quantity, like heat). R correlates perfectly with sqrt(likelihood precision), making it an "evidence density" that measures signal quality independent of sample size. This means R functions as a gate (accept/reject a source) rather than a confidence interval (accumulate evidence), preventing "false confidence via volume."

## v1 Evidence Summary

Q15 had a turbulent history with three distinct phases:

1. **Original claim:** R has Bayesian connections (untested).

2. **GLM4.7 correction (Q15_CORRECTED_RESULTS.md):** Rigorous falsification test using a neural network with trained weights. Tested R against actual Bayesian quantities:
   - R vs. Hessian-based posterior precision: confidence interval included zero.
   - R vs. KL divergence: no significant correlation.
   - R vs. Fisher information: no significant correlation.
   - Result: 0/3 predictions validated. Status: FALSIFIED.
   - Used 5 independent trials with different seeds.

3. **Re-reversal (Q15_PROPER_TEST_RESULTS.md):** New test with simplified formula (E hardcoded to 1.0, sigma^Df dropped). Computed R = 1/std on Gaussian data. Found:
   - R vs sqrt(likelihood precision = 1/std^2): correlation = 1.0.
   - R vs posterior precision: correlation = -0.0937.
   - Conclusion: R measures "signal quality" (intensive), not data volume (extensive).
   - Used a single seed (seed=42).

## v1 Methodology Problems

The verification identified the following issues:

1. **Core "discovery" is a tautology (CRITICAL).** The "proper test" hardcodes E = 1.0 and drops sigma^Df, making R = 1/std. It then reports that R correlates perfectly with sqrt(1/std^2) = 1/std. This is verifying that 1/x = 1/x. The correlation of 1.0 is an algebraic identity, not a discovery.

2. **Actual formula never tested (CRITICAL).** The full formula R = (E/grad_S) * sigma^Df was never tested against Bayesian quantities. Only the gutted skeleton R = 1/std was tested. The Bayesian connection holds for 1/std (trivially known) but is not demonstrated for R as actually defined.

3. **Post-hoc rescue after more rigorous falsification (HIGH).** The earlier falsification (CORRECTED_RESULTS) used a real learning system, actual Bayesian quantities, 5 independent seeds, and found no connection. The "rescue" used a tautological test on trivial data with a single seed. The stronger test said "no connection"; the weaker test said "perfect connection." The weaker test was accepted.

4. **Intensive/extensive framing is post-hoc rhetoric (HIGH).** The intensive/extensive distinction was introduced AFTER the falsification to explain away negative results. No rigorous definition of "intensive" in the physics sense is provided. R is called "intensive" because it does not depend on N -- but any quantity that depends only on sigma has this property, including 1/sigma itself.

5. **"Temperature of the truth" metaphor has no formal content (MEDIUM).** Evocative language does not constitute a mathematical connection.

6. **All evidence synthetic (MEDIUM).** Gaussian random data only. No real-world data.

## v2 Test Plan

### Test 1: Full-Formula Bayesian Correlation
- Compute the FULL formula R = (E/grad_S) * sigma^Df on real data (not the E=1 simplification).
- Use multiple embedding models on STS-B clusters.
- Compute actual Bayesian quantities for the same clusters:
  - Posterior precision (from Bayesian linear regression or Gaussian process)
  - Marginal likelihood (model evidence)
  - Fisher information matrix (trace or determinant)
  - ELBO from variational inference
- Measure correlation between full R and each Bayesian quantity.
- If full R correlates with sqrt(likelihood precision), the connection holds. If only R=1/std correlates, the connection is trivial and does not extend to the formula.

### Test 2: Intensive Property Verification
- Formally test whether R is intensive (independent of system size):
  - For the same data source, compute R on subsets of size N = 10, 20, 50, 100, 200, 500, 1000.
  - Measure whether R stabilizes (coefficient of variation < 0.1 across N values).
  - Compare against posterior precision (which should grow with N) and likelihood precision (which should be constant).
- Do this for full R, not just 1/std.
- Do this on real data from at least 3 different domains.

### Test 3: Gating vs. Confidence Interval Comparison
- Set up a decision task: decide whether to trust a data source.
- Compare:
  - R-gating: trust if R > threshold.
  - Bayesian confidence interval: trust if CI width < threshold.
  - Posterior probability: trust if P(correct) > threshold.
- Measure decision accuracy on held-out test data.
- The hypothesis predicts R-gating should be superior because it prevents "false confidence via volume" (trusting bad sources with lots of data). Test this explicitly:
  - Create a noisy source with N=10000 observations (low quality, high volume).
  - Create a clean source with N=10 observations (high quality, low volume).
  - R-gating should prefer the clean source; Bayesian confidence should prefer the noisy source.

### Test 4: Reinstating the Falsification
- Reproduce the CORRECTED_RESULTS.md falsification test on the full formula.
- Use a neural network or other non-trivial model where Bayesian inference has standard meaning.
- Compute R (with proper E, not E=1) alongside Hessian-based precision, KL divergence, and Fisher information.
- Run with 10+ independent seeds.
- If the falsification reproduces, acknowledge that the full formula does not have a Bayesian connection.

### Test 5: Comparison Against Known Intensive Bayesian Quantities
- Compare R against established intensive Bayesian quantities:
  - Likelihood ratio (point-wise, not aggregated)
  - Bayes factor for model comparison
  - Predictive density at observed values
- Determine whether R captures the same or different information.
- If R is indistinguishable from 1/sigma (which is trivially intensive), the Bayesian interpretation adds nothing beyond "R includes 1/sigma in its definition."

## Required Data

- **STS-B / MTEB** -- for computing full R with operational E
- **UCI regression datasets** -- for Bayesian inference comparison with known posteriors
- **Sentence-transformers models** -- for embedding generation
- **Neural network weight datasets** -- for Hessian-based posterior precision computation
- **Gaussian process benchmark datasets** (e.g., Mauna Loa CO2, airline passengers) -- for clean Bayesian inference benchmarks

## Pre-Registered Criteria

- **Success (confirm):** Full-formula R (with operational E, not E=1) correlates with sqrt(likelihood precision) at rho > 0.7 on real data from at least 2 domains. AND R is intensive (CV < 0.15 across sample sizes N=10 to N=1000) for the full formula. AND R-gating outperforms confidence-interval-based gating on the volume-vs-quality discrimination task. AND the CORRECTED_RESULTS falsification does NOT reproduce on the full formula with 10+ seeds (i.e., the full formula actually does correlate with Bayesian quantities).
- **Failure (falsify):** Full-formula R does NOT correlate with any standard Bayesian quantity (all rho < 0.2 with 10+ seeds), OR R is NOT intensive for the full formula (CV > 0.3 across sample sizes), OR R-gating performs no better than confidence-interval gating on the discrimination task, OR the CORRECTED_RESULTS falsification reproduces on the full formula.
- **Inconclusive:** Moderate correlations (0.2-0.7); R is approximately intensive but with some N-dependence; gating performance is similar across methods.

## Baseline Comparisons

R's Bayesian interpretation must provide value beyond:
- 1/sigma (the trivial intensive component already known to be in R)
- sqrt(Fisher information) (the standard frequentist precision)
- Bayesian posterior precision (the standard Bayesian quantity)
- ELBO (the standard variational quantity)
- Predictive log-likelihood
- Cross-validation error as a quality signal

## Salvageable from v1

- **q15_bayesian_validated.py (the falsification test)** -- Ironically, this is MORE valuable than the "proper" test. Its methodology (real neural network, Hessian computation, 5 seeds, multiple Bayesian quantities) is rigorous and should be the starting point for v2. Path: `v1/questions/medium_q15_1460/tests/q15_bayesian_validated.py`
- **Q15_CORRECTED_RESULTS.md** -- The falsification report is methodologically sound and should be taken seriously rather than overturned by a tautological test. Path: `v1/questions/medium_q15_1460/reports/Q15_CORRECTED_RESULTS.md`
- **Q15_INTENSIVE_EXTENSIVE_DISCOVERY.md** -- The intensive/extensive framing, while post-hoc, raises a genuinely interesting question that should be tested properly. Path: `v1/questions/medium_q15_1460/reports/Q15_INTENSIVE_EXTENSIVE_DISCOVERY.md`
- **The gating-vs-confidence-interval insight** -- The conceptual distinction between R as a gate (accept/reject source quality) vs. Bayesian confidence as an accumulator (more data = more confidence) is a testable and interesting claim. Worth preserving as a specific test case.
