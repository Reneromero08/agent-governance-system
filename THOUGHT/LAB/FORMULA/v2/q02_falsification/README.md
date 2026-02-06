# Q02: The Formula Has Meaningful Falsification Criteria

## Hypothesis

The Living Formula R = (E / grad_S) * sigma^Df has concrete, testable falsification criteria -- specific conditions under which we would say the formula is wrong, not just "needs more context." These criteria have modus tollens structure: if the formula is correct, then P; if NOT P is observed, the formula is incorrect.

## v1 Evidence Summary

Two test files were produced:

1. **q2_falsification_test.py** -- Tested four attack vectors: echo chamber (correlated biased), adversarial (identical wrong values), systematic bias (constant offset), and bimodal trap (two clusters). Used a toy E definition: E = 1/(1 + std(observations)).

2. **q2_echo_chamber_deep_test.py** -- Deeper echo chamber analysis. Key findings:
   - Independent observations: mean R = 0.15, mean error = 0.26 (R predicts accuracy).
   - Echo chamber: mean R = 3.10, mean error = 2.44 (R does NOT predict accuracy; 20x inflation).
   - R > 95th percentile: 0% independent, 10% echo chambers.
   - Bootstrap defense: adding 1 fresh observation drops echo R from 2.47 to 0.18; adding 20 drops to 0.05.
   - Echo R drops 93% vs real drops 75% when fresh data added.

Three stated criteria: (1) R measures local agreement (correct claim), (2) R fails when observations are correlated, (3) Defense: add fresh data; if R crashes, it was an echo chamber.

## v1 Methodology Problems

The verification identified the following issues:

1. **Circular escape hatch (CRITICAL).** The falsification criterion for R's main failure mode (echo chambers) is: "If R fails, independence was probably violated." This uses the failure to retroactively explain the failure, without any independent test of the independence assumption. R contains no term that measures or penalizes correlation.

2. **Test code uses wrong formula (CRITICAL).** The test defines E = 1/(1 + std(observations)), making R = sigma^Df / (std * (1+std)). This is a toy proxy, NOT the actual formula from the GLOSSARY (E = mean pairwise cosine similarity). Conclusions about the toy formula do not transfer.

3. **No modus tollens structure (HIGH).** The criteria protect the formula from falsification rather than exposing it. The correct structure would be: "For independent observations, high R implies high accuracy. If independent observations show high R with low accuracy, the formula is falsified." But "high R" and "high accuracy" are never defined numerically.

4. **No numerical thresholds (HIGH).** What R value constitutes "high"? What drop percentage constitutes "crash"? The 30% drop threshold in the test code was chosen ad hoc and tuned for good F1 on specific simulation parameters.

5. **Missing attack vectors (HIGH).** Not covered: confounded variables (independent but sharing systematic bias), dimensionality collapse (1D projection hiding high-dimensional disagreement), scale dependence (what if sigma is wrong?), grad_S degeneracy (R diverges as grad_S approaches 0), adversarial embedding attacks.

6. **All criteria reduce to one test (MEDIUM).** Every criterion boils down to "Is the independence assumption satisfied?" No separate criteria for E being miscalculated, grad_S being a poor variability measure, sigma being wrong, Df being incorrectly estimated, or the functional form being wrong.

7. **Bootstrap defense is post-hoc (MEDIUM).** The defense was designed after the attack was discovered, not pre-registered.

8. **All evidence synthetic (HIGH).** No real embedding data, no real semantic tasks, no published benchmarks.

## v2 Test Plan

### Test 1: Pre-Registered Modus Tollens Criteria
Before running any test, state falsification criteria with the form:
- "If R is correct, then for N >= 30 independent observations from the same ground-truth cluster, R > T implies mean error < E_max."
- Define T and E_max from a calibration dataset (not the test dataset).
- Run the test on a held-out dataset. If the conditional is violated, the formula is falsified under the stated conditions.

### Test 2: Component-Level Falsification
Separate criteria for each formula component:
- **E falsification:** On STS-B, if cosine-similarity E does not correlate (r > 0.5) with human similarity judgments, E is a poor alignment measure.
- **grad_S falsification:** If grad_S does not correlate with empirical uncertainty (measured by bootstrap variance of cluster centroids), grad_S is a poor dispersion measure.
- **sigma falsification:** If the optimal sigma varies by more than 50% across domains, the universality claim for sigma fails.
- **Df falsification:** If Df measured from eigenvalue decay does not predict cross-domain transfer performance, Df has no explanatory value.
- **Functional form falsification:** If an alternative form (E - grad_S, E * exp(-grad_S), log(E)/grad_S) outperforms R on truth-prediction tasks, the specific ratio form is not privileged.

### Test 3: Real-Data Echo Chamber Detection
- Use known echo chamber datasets (e.g., Reddit political subreddits with cross-validated ground truth, Twitter bot detection datasets).
- Compute R on real correlated sources vs. diverse sources.
- Measure whether R or the bootstrap defense reliably distinguishes echo chambers WITHOUT oracle access to truth.

### Test 4: Confounded Variables Attack
- Generate independent observations that share systematic bias (e.g., embeddings from models trained on the same biased corpus).
- Measure whether R is high and wrong -- this would be a failure mode not covered by the independence criterion.
- Determine whether any modification to R can detect this failure.

### Test 5: Adversarial Robustness on Real Embeddings
- Apply adversarial perturbations to sentence embeddings that preserve cosine similarity but change semantic content.
- Measure whether R remains high despite the meaning change.
- Compare against adversarial robustness of alternative measures.

## Required Data

- **STS-B** (Semantic Textual Similarity Benchmark) for ground-truth similarity
- **MTEB** clustering tasks for diverse embedding evaluation
- **Reddit political datasets** (e.g., r/politics vs. r/conservative with fact-check labels)
- **Twitter bot detection datasets** (e.g., Cresci 2017, TwiBot-22)
- **TextAttack adversarial examples** for adversarial robustness testing
- **MultiNLI** for contradiction/entailment ground truth

## Pre-Registered Criteria

- **Success (confirm):** At least 4 of 5 component-level falsification tests pass (each component demonstrates the property it claims), AND the modus tollens test on held-out real data holds (high R implies accuracy within the pre-specified threshold), AND the formula detects at least 80% of echo chambers on real data without oracle access.
- **Failure (falsify):** The modus tollens test fails on real data (high R does NOT imply accuracy for independent observations), OR 3+ component-level tests fail, OR the confounded-variables attack produces high R on systematically wrong answers with statistically independent observations.
- **Inconclusive:** Some but not all components pass; echo chamber detection works on synthetic but not real data; confounded-variables results are ambiguous.

## Baseline Comparisons

The falsification framework must be compared against:
- Standard statistical hypothesis testing (p-values, confidence intervals)
- Bayesian model comparison (Bayes factors)
- Cross-validation error as a quality signal
- Simple agreement metrics (raw inter-rater agreement, Fleiss' kappa)
- Anomaly detection methods for echo chamber identification (isolation forest, LOF)

## Salvageable from v1

- **q2_echo_chamber_deep_test.py** -- The echo chamber simulation framework is useful for generating synthetic test cases, though the E definition must be replaced. The specific numbers (20x R inflation, 93% drop with fresh data) provide a baseline for synthetic calibration. Path: `v1/questions/critical_q02_1750/tests/q2_echo_chamber_deep_test.py`
- **Bootstrap defense concept** -- The idea of testing R's robustness to fresh data is sound, even though the specific implementation was post-hoc. Should be formalized with pre-registered thresholds.
- **Attack vector list** -- The four attack types (echo chamber, adversarial, systematic bias, bimodal) are a starting point, though five additional vectors need to be added.
