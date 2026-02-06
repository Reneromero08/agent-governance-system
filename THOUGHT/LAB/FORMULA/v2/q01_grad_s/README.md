# Q01: grad_S Is the Correct Normalization

## Hypothesis

Dividing by grad_S (the local dispersion / scale parameter) is the mathematically necessary normalization for the resonance formula R = E / grad_S. Specifically: for any location-scale probability family, the likelihood density evaluated at the truth point has the form (1/s) * f((x - mu)/s), making division by the scale parameter structurally forced -- not a design choice. Furthermore, R = E(z)/s is the unique form satisfying locality, normalized deviation, monotonicity, and scale normalization axioms.

## v1 Evidence Summary

Six test files were produced in v1:

1. **q1_why_grad_s_test.py** -- Compared E/std against alternatives; found E/std correlates best with "correctness" metric 1/(1+|mean-truth|).
2. **q1_deep_grad_s_test.py** -- Tested correlated vs independent observations.
3. **q1_adversarial_test.py** -- Ran attack vectors against the formula; noted "Circular E (v1) is USELESS" when E depends on truth.
4. **q1_essence_is_truth_test.py** -- Tested that E must be grounded in reality.
5. **q1_derivation_test.py** -- Verified log(R) = -F + const for Gaussian case (correlation = 1.0, offset matches 0.5*log(2*pi)). Extended to Laplace family showing log(R_mad) = -F_laplace + const.
6. **q1_definitive_test.py** -- Attempted "uniqueness proof" via four axioms: dimensional consistency, monotonicity, linear scale behavior, free energy alignment. Claimed R = E/std is uniquely determined.

All tests used synthetic data (numpy random generation). E was defined as exp(-z^2/2) where z = |mean - truth| / std, which is the Gaussian kernel -- not the operational E (cosine similarity) from the GLOSSARY.

## v1 Methodology Problems

The verification identified the following issues with the v1 tests:

1. **Free Energy identity is tautological by construction (CRITICAL).** R was defined to be the Gaussian likelihood (E = exp(-z^2/2), R = E/std), then the test verified log(Gaussian likelihood) = -(Gaussian negative log-likelihood). This is verifying that log(exp(x)) = x.

2. **Uniqueness proof is circular (CRITICAL).** Axiom 3 explicitly requires "linear scale behavior," which forces n=1 in E/std^n. The axioms were chosen to produce the pre-existing formula. Any exponent n could be "uniquely determined" by an axiom requiring "n-th power scale behavior."

3. **grad_S dimensionality contradiction (HIGH).** GLOSSARY defines grad_S as "dimensionless scalar" but the uniqueness proof claims std has physical units (length, time). These cannot both be true.

4. **E definition gap (HIGH).** Tests use E = exp(-z^2/2) but the GLOSSARY defines E as "mean pairwise cosine similarity" (semantic domain) or "mutual information" (quantum domain). The proof applies to Gaussian likelihood E, not operational E.

5. **All evidence is synthetic (MEDIUM).** No real-world data in any test. No real embedding data, no real semantic tasks.

6. **Post-hoc derivation (MEDIUM).** The formula existed before the location-scale derivation was constructed. The axioms were written to describe the pre-existing formula, not the other way around.

7. **Hidden premise (MEDIUM).** The derivation assumes the quantity of interest IS the likelihood density at the truth point. This equates "resonance" with "likelihood" without justification.

## v2 Test Plan

### Test 1: Location-Scale Derivation on Real Data
- Take real-world measurement datasets where ground truth is known (e.g., NIST Standard Reference Data, UCI regression benchmarks).
- Compute R = E/std using the Gaussian kernel E and compare against ground-truth accuracy.
- Compare R against alternative normalizations: E/std^2 (precision-weighted), E/MAD, E/IQR, E*exp(-std), E/(std + epsilon).
- Measure which normalization best predicts proximity to truth, using held-out test data.

### Test 2: Bridge Between Theoretical E and Operational E
- For text embedding data, compute both E_gaussian = exp(-z^2/2) and E_cosine = mean pairwise cosine similarity for the same observation clusters.
- Measure the correlation between E_gaussian and E_cosine across at least 500 clusters.
- Determine under what conditions (if any) cosine-similarity E approximates the Gaussian kernel E.

### Test 3: Non-Location-Scale Domains
- Apply R = E/grad_S to domains where location-scale assumptions fail: count data (Poisson), compositional data (Dirichlet), directional data (von Mises).
- Compare R against domain-appropriate alternatives (e.g., for Poisson: E/sqrt(lambda)).
- Determine whether the 1/s normalization remains optimal outside location-scale families.

### Test 4: Uniqueness from Independent Axioms
- Formulate axioms that do NOT individually force the answer.
- Start from information-theoretic axioms (e.g., Fisher sufficiency, data processing inequality) rather than scale-behavior axioms.
- Attempt a genuine derivation where the form E/s emerges as a consequence rather than a premise.

### Test 5: Real-World Semantic Validation
- Use STS-B (Semantic Textual Similarity Benchmark) with human similarity judgments as ground truth.
- Compute R for clusters of sentence embeddings grouped by topic/meaning.
- Measure whether R correlates with human-judged semantic coherence.

## Required Data

- **NIST Standard Reference Data** (physics/chemistry measurements with certified values)
- **UCI Machine Learning Repository** regression datasets (Boston Housing, Concrete Strength, Energy Efficiency)
- **STS-B** (Semantic Textual Similarity Benchmark, part of GLUE/SuperGLUE)
- **MTEB** (Massive Text Embedding Benchmark) clustering tasks
- **LibriSpeech** (for audio domain testing)

## Pre-Registered Criteria

- **Success (confirm):** On real-world data with known ground truth, E/std outperforms at least 4 of 5 alternative normalizations (E/std^2, E/MAD, E/IQR, E*exp(-std), raw E) in predicting truth-proximity, with a statistically significant advantage (p < 0.01, two-tailed paired t-test across datasets). AND the bridge test shows correlation > 0.8 between E_gaussian and E_cosine.
- **Failure (falsify):** E/std does NOT outperform the majority of alternatives on real data (loses to 3+ alternatives), OR bridge test shows correlation < 0.3 between E_gaussian and E_cosine, indicating the theoretical E and operational E are unrelated.
- **Inconclusive:** E/std outperforms some but not all alternatives without clear statistical significance, or bridge correlation falls between 0.3 and 0.8.

## Baseline Comparisons

R = E/std must outperform:
- Raw E (no normalization) -- to show normalization adds value
- E/std^2 (precision-weighted) -- the standard Bayesian quantity
- E/MAD (robust alternative) -- to show std is specifically preferred
- E/IQR (another robust alternative)
- Simple SNR = mean/std -- to show E contributes beyond basic signal-to-noise
- Random baseline -- R computed on shuffled data

## Salvageable from v1

- **q1_derivation_test.py** -- The Gaussian and Laplace family-scoped identity tests are algebraically correct within their stated scope. The code structure for computing R across families is reusable, but must be supplemented with real data and the operational E definition. Path: `v1/questions/critical_q01_1800/tests/q1_derivation_test.py`
- **q1_adversarial_test.py** -- The insight that "Circular E is useless" is valuable self-criticism. The attack vector framework can be reused. Path: `v1/questions/critical_q01_1800/tests/q1_adversarial_test.py`
- **reports/Q1_GRAD_S_SOLVED_MEANING.md** -- More honest than the primary document about scope limitations ("if your resonance is built from a location-scale uncertainty model"). Path: `v1/questions/critical_q01_1800/reports/Q1_GRAD_S_SOLVED_MEANING.md`
