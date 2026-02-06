# Q03: R Generalizes Across Domains

## Hypothesis

The formula R = E(z)/sigma generalizes across fundamentally different domains -- not merely different statistical distributions, but genuinely different conceptual domains (text semantics, quantum mechanics, financial markets, biological systems). This generalization reflects a deep isomorphism: R captures the universal structure of evidence under noise, and any domain with distributed observations, scale-dependent measurements, agreement-indicating truth, and signal-quality requirements must exhibit R = E(z)/sigma as its evidence measure.

## v1 Evidence Summary

Six test files and multiple reports were produced:

1. **test_phase1_uniqueness.py** -- Attempted axiomatic uniqueness proof via A1-A4. Axiom checks all returned True unconditionally. "Uniqueness" tested by comparing R = E/sigma against E/sigma^2, E^2/sigma, E-sigma using a CV threshold of 0.3.

2. **test_phase1_unified_formula.py** -- Applied the same R computation (E = mean(exp(-z^2/2)), R = E/std) to Gaussian, Bernoulli, and Quantum observation vectors.

3. **test_phase2_falsification.py / test_phase2_pareto.py** -- Phase 2 originally FAILED: some alternatives dominate R on (information transfer, noise sensitivity) frontier. Metrics were then revised to "correct metrics" where R excels (likelihood precision correlation, intensive property, cross-domain transfer).

4. **test_phase3_adversarial.py** -- Tested 5 adversarial domains (Cauchy, Poisson sparse, bimodal GMM, AR(1), random walk). Pass criterion: R > 0 and error >= 0. Result: 5/5 passed. Random walk "truth" was set to np.mean(observations).

5. **test_phase3_quantum.py** -- Quantum Darwinism test. sigma_quantum = sqrt(N_fragments) and Df = 1/purity defined by fiat. Entanglement test always returns True regardless of results.

Cross-domain transfer: threshold learned on Gaussian transferred to Uniform with Gaussian high-R error = 0.23, Uniform high-R error = 0.18.

## v1 Methodology Problems

The verification identified the following issues:

1. **All "domains" reduce to the same computation (CRITICAL).** Gaussian, Bernoulli, and Quantum tests all compute E = mean(exp(-z^2/2)) and R = E/std on arrays of numbers. The "quantum" test generates +/-1 outcomes from QuTiP-computed probabilities but the R computation is purely classical. This is the same arithmetic on different inputs, not genuine cross-domain generalization.

2. **Necessity proof is circular (CRITICAL).** Axiom A4 ("R must scale as 1/sigma") IS the conclusion. Demanding 1/sigma scaling is logically equivalent to demanding sigma in the denominator. The axioms were extracted from properties of the pre-existing formula.

3. **Near-vacuous adversarial pass criteria (HIGH).** The test passes if R > 0 and error >= 0, which only checks that the code does not crash. Whether R is meaningful, predictive, or calibrated is not tested.

4. **Post-hoc Pareto metric selection (HIGH).** Phase 2 originally failed. The response was to declare the original metrics "wrong" and select metrics where R excels (likelihood precision correlation, intensive property). This is textbook post-hoc fitting.

5. **Semantic E never tested (CRITICAL).** GLOSSARY E (cosine similarity) is never used. All tests use E = exp(-z^2/2). The generalization claim requires bridging these definitions, which was never done.

6. **Quantum tests have no quantum content (HIGH).** sigma = sqrt(N) and Df = 1/purity are assigned by fiat. R_full = R_base * sqrt(N)^(1/purity) trivially scales as designed.

7. **Project's own roadmap says 1/7 criteria met.** The research roadmap lists 7 criteria for ANSWERED status and self-assesses 1/7 complete -- yet Q3 was marked ANSWERED.

## v2 Test Plan

### Test 1: Genuinely Different Domains with Domain-Appropriate E
Apply R to at least 5 fundamentally different domains, using the domain-appropriate definition of E (NOT the same Gaussian kernel for all):
- **Text semantics:** E = mean pairwise cosine similarity of sentence embeddings. Dataset: STS-B.
- **Protein structure:** E = TM-score or GDT-TS of structure predictions vs. experimental structures. Dataset: CASP14/CASP15.
- **Financial time series:** E = correlation of return predictions with actual returns. Dataset: S&P 500 daily returns.
- **Climate science:** E = skill score of ensemble weather forecasts. Dataset: ECMWF reanalysis (ERA5).
- **Medical diagnosis:** E = inter-rater agreement (Fleiss' kappa) among independent diagnoses. Dataset: MIMIC-IV or published diagnostic agreement studies.

For each domain, define grad_S from domain-appropriate dispersion measures.

### Test 2: Cross-Domain Transfer
- Calibrate R thresholds (what R value separates "reliable" from "unreliable" evidence) on one domain.
- Apply those thresholds to a different domain WITHOUT recalibration.
- Measure whether the threshold transfers: does high-R evidence in the new domain actually correlate with accuracy?
- Compare at least 3 cross-domain pairs.

### Test 3: Comparison Against Domain-Specific Alternatives
For each domain, compare R against the established quality metric:
- Text: R vs. BERTScore, BLEU, human judgment
- Protein: R vs. TM-score alone, LDDT
- Finance: R vs. Sharpe ratio, information ratio
- Climate: R vs. RMSE, Brier skill score
- Medical: R vs. Cohen's kappa, diagnostic accuracy

R must add value beyond what these domain-specific metrics already provide.

### Test 4: Axiom Validation from Independent Principles
- Rather than testing whether R satisfies chosen axioms, test whether the axioms themselves are independently motivated.
- For each domain, verify: (A1) can evidence be computed locally? (A2) does normalization by dispersion improve prediction? (A3) does evidence decrease with deviation? (A4) is the quality measure independent of sample size?
- If axioms hold naturally in 4+ domains, this supports generalization. If axioms must be forced or reinterpreted per domain, generalization is surface-level.

### Test 5: Adversarial Domains with Meaningful Pass Criteria
- Rerun adversarial tests with actual performance criteria: R must predict ground-truth accuracy with correlation > 0.5 (not just R > 0).
- Include domains where R is expected to fail (e.g., purely chaotic systems) and verify it does fail.
- A test that cannot fail is not a test.

## Required Data

- **STS-B / MTEB** -- text semantic similarity with human judgments
- **CASP14/CASP15** -- protein structure prediction with experimental ground truth
- **S&P 500 daily returns** (via Yahoo Finance API or CRSP)
- **ERA5 reanalysis** (ECMWF climate data, freely available)
- **MIMIC-IV** (medical records, requires credentialed access) or published diagnostic agreement datasets
- **Lorenz attractor / logistic map** data for chaotic systems (negative control)

## Pre-Registered Criteria

- **Success (confirm):** R computed with domain-appropriate E and grad_S correlates with ground-truth accuracy (Spearman rho > 0.5) in at least 4 of 5 domains. AND cross-domain threshold transfer works (accuracy in new domain is within 20% of calibration domain) for at least 2 of 3 pairs. AND R outperforms at least 2 of 3 domain-specific alternatives in at least 3 domains.
- **Failure (falsify):** R correlates with accuracy in fewer than 2 of 5 domains (rho > 0.5), OR cross-domain transfer fails entirely (transferred thresholds perform no better than random), OR R is beaten by all domain-specific alternatives in all domains tested.
- **Inconclusive:** R works in 2-3 domains but fails in others; cross-domain transfer is mixed; R matches but does not outperform domain-specific alternatives.

## Baseline Comparisons

R must outperform or match:
- Domain-specific quality metrics (listed above per domain)
- Simple signal-to-noise ratio (mean/std)
- Raw E without normalization
- Z-score alone (without the E function)
- Ensemble variance as an uncertainty measure
- Random forest confidence scores (for prediction tasks)

## Salvageable from v1

- **test_phase1_unified_formula.py** -- The multi-distribution computation framework is reusable for generating synthetic calibration data, though it must be supplemented with real data and domain-specific E definitions. Path: `v1/questions/critical_q03_1720/tests/test_phase1_unified_formula.py`
- **q03_research_roadmap.md** -- The honest self-assessment (1/7 criteria met) and the 7 success criteria are a valuable starting point for v2 planning. Path: `v1/questions/critical_q03_1720/q3_why_generalize/q03_research_roadmap.md`
- **q03_interface_theory_note.md** -- The more modest framing ("R is a natural structure for adaptive interfaces") is more defensible than "universal necessity" and could inform v2 scope. Path: `v1/questions/critical_q03_1720/q3_why_generalize/q03_interface_theory_note.md`
- **Cross-domain transfer numbers** -- Gaussian-to-Uniform transfer (error 0.23 vs 0.18) provides a synthetic baseline, though both are location-scale families (a very easy case).
