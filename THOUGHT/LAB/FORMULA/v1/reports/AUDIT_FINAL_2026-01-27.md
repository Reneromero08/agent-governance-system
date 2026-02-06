# FINAL AUDIT: Research Questions Modified 2026-01-27

**Auditor:** Claude Opus 4.5
**Date:** 2026-01-27
**Audit Criteria:** Pre-registration, Independent Ground Truth, Real Data, Honest Reporting, P-hacking, Tests Executed

---

## EXECUTIVE SUMMARY

| Metric | Count |
|--------|-------|
| Questions Audited | 13 |
| PASS | 5 |
| PARTIAL | 4 |
| FAIL | 2 |
| ENGINEERING (Not Science) | 2 |

**Overall Assessment:** Mixed. Several questions show good scientific practice (pre-registration, negative controls, honest falsification). However, significant issues remain with synthetic data usage, circular validation, and some spin on partial results.

---

## AUDIT TABLE

| Q# | Pre-reg | Independent GT | Real Data | Honest | Ran Tests | VERDICT |
|----|---------|----------------|-----------|--------|-----------|---------|
| Q16 | YES | YES | YES (SNLI/ANLI) | YES | YES | **PASS** |
| Q19 | YES | YES | YES (OASST/SHP/HH-RLHF) | YES | YES | **PASS** |
| Q20 | YES | PARTIAL | YES (Audio/Image) | YES (after correction) | YES | **PARTIAL** |
| Q22 | NO | NO | NO (synthetic) | PARTIAL | PARTIAL | **FAIL** |
| Q23 | YES | YES | PARTIAL | YES (falsified) | YES | **PASS** |
| Q24 | PARTIAL | NO | YES (SPY market) | YES | YES | **PARTIAL** |
| Q25 | YES | N/A | NO (synthetic) | PARTIAL | YES | **PARTIAL** |
| Q26 | YES | N/A | PARTIAL (embeddings) | YES (corrected) | YES | **PASS** |
| Q28 | YES | PARTIAL | YES (SPY market) | YES | YES | **PASS** |
| Q29 | N/A | N/A | N/A | N/A | YES | **ENGINEERING** |
| Q30 | N/A | N/A | NO (synthetic) | YES | YES | **ENGINEERING** |
| Q52 | YES | YES | NO (Logistic map) | YES (falsified) | YES | **PARTIAL** |
| Q53 | YES | PARTIAL | YES (embeddings) | YES (corrected) | YES | **PARTIAL** |

---

## DETAILED FINDINGS

### Q16: Domain Boundaries - **PASS**

**Strengths:**
- Pre-registered hypothesis: "R < 0.5 correlation in adversarial/NLI domains"
- Used REAL external datasets: SNLI (Stanford NLI), ANLI R3 (Facebook adversarial)
- Ground truth (entailment/neutral/contradiction labels) independent of R
- Included positive control (topical consistency)
- Honestly reported SNLI result that FAILED pre-registered threshold (r=0.71 > 0.5)
- Test actually executed with 500+300+200 samples

**Weaknesses:**
- None significant

**Evidence of Test Execution:**
- Results file: `q16_results.json` with timestamps, sample counts, p-values
- Reports unexpected SNLI result honestly


### Q19: Value Learning - **PASS**

**Strengths:**
- Pre-registered: "r > 0.5 between R and inter-annotator agreement"
- Used 3 REAL datasets: OASST, SHP, HH-RLHF (n=900 total)
- Ground truth (annotator agreement) independent of R
- Honestly reports within-source divergence (avg r = 0.051)
- Flags Simpson's paradox risk
- Notes SHP and HH-RLHF show NEGATIVE correlations

**Weaknesses:**
- The "PASS" verdict (r=0.52 > 0.5) barely clears threshold
- Cross-source confounding creates spurious correlation
- Status "CONDITIONALLY CONFIRMED" is honest but the condition is severe

**Red Flags:**
- The overall r=0.52 is driven almost entirely by OASST (r=0.60)
- Other sources show r=-0.14 (SHP) and r=-0.31 (HH-RLHF)
- A stricter auditor might call this FAIL due to Simpson's paradox


### Q20: Tautology Risk - **PARTIAL**

**Strengths:**
- Pre-registered predictions for code embeddings, random control, Riemann alpha
- CRITICAL: Conducted novel domain test (audio, image, graph)
- Honestly reports 8e FAILS on novel domains (38-100% error)
- Updates verdict from "EXPLANATORY" to "CIRCULAR VALIDATION CONFIRMED"
- Self-correcting science

**Weaknesses:**
- Original test used code snippets with text embedding models (still text-adjacent)
- Novel domain test was added AFTER audit concerns raised
- The Riemann alpha = 0.5 connection is still cited as evidence despite domain failure

**Red Flags:**
- Status changed from PASS to CIRCULAR - good catch but late
- Still claims "8e holds for text embeddings" which is circular by definition


### Q22: Threshold Calibration - **FAIL**

**Strengths:**
- Tests multiple thresholds and percentiles
- Honest about fixed constants not working

**Weaknesses:**
- NO pre-registration
- Ground truth (F1 score) depends on R via the classification task
- Uses Q23's synthetic word clusters, not real labeled data
- "PARTIALLY ANSWERED" status hides that core testing is incomplete
- No external validation dataset

**Red Flags:**
- References Q23 test data but Q23 used synthetic word clusters
- "Multi-domain validation missing" - acknowledged but not addressed
- This is hypothesis-generating, not hypothesis-testing


### Q23: sqrt(3) Geometry - **PASS**

**Strengths:**
- Pre-registered multiple hypotheses (hexagonal packing, Berry phase, distinguishability)
- Honestly FALSIFIED hexagonal geometry hypothesis
- Honestly reports sqrt(3) is NOT universally optimal (2/5 models)
- Negative controls: random vs trained embeddings
- Multi-model validation (5 models tested)
- Clear "CLOSED - EMPIRICAL NOT GEOMETRIC" conclusion

**Weaknesses:**
- Original data was synthetic word clusters (not external benchmark)
- Angle analysis used 2D PCA projection (not true high-dimensional geometry)

**Red Flags:**
- None - this is good falsificationist science


### Q24: Failure Modes - **PARTIAL**

**Strengths:**
- Used REAL data: SPY market data from yfinance
- Tests 4 strategies with quantitative outcomes
- Honest that WAIT strategy FAILS (-34% R improvement)

**Weaknesses:**
- PARTIAL pre-registration: hypothesis "waiting improves R by >20%" was stated
- Ground truth for "success rate" is self-defined (R improvement, acceptable outcome)
- Only 17 low-R periods tested - small sample

**Red Flags:**
- "CHANGE_FEATURES +80% improvement" based on n=17 periods
- Success metrics are internal to R, not external outcomes


### Q25: What Determines Sigma - **PARTIAL**

**Strengths:**
- Pre-registered: "R^2 > 0.7 for sigma predictability"
- Cross-validated R^2 = 0.86 clearly exceeds threshold
- Tests 22 datasets across 7 domains
- Regression formula with interpretable coefficients

**Weaknesses:**
- ALL 22 datasets are SYNTHETIC
- Ground truth (optimal sigma) found by grid search ON the same formula
- This is parameter fitting, not independent validation
- "Domain-agnostic" claim not validated on real data

**Red Flags:**
- Circular: sigma is optimized for R, then R^2 is computed on sigma
- No external validation on actual NLP/image/market benchmarks
- "Remaining Questions" section acknowledges synthetic limitation


### Q26: Minimum Data Requirements - **PASS**

**Strengths:**
- Pre-registered: "N_min scales with log(dimensionality)"
- Honestly reports hypothesis FALSIFIED (N_min is constant, not scaling)
- Rigorous retest after original test flagged as UNDERPOWERED
- Multi-model testing (7 models, multiple dimensions)
- Semantic structure test (4 corpus types)
- CORRECTIONS LOG documents what went wrong

**Weaknesses:**
- Original test was indeed underpowered (1 model at 1 dimension)
- Embeddings are real but test task is synthetic (R stability measurement)

**Red Flags:**
- Original spin "N=5-10 is enough" was based on single test - corrected


### Q28: Attractors - **PASS**

**Strengths:**
- Pre-registered: "R convergent, not chaotic" with falsification "Lyapunov > 0.05"
- Used REAL data: SPY market regimes from yfinance
- 7 distinct market regimes tested
- Lyapunov exponent computed (mean 0.036 < 0.05 threshold)
- 82% pass rate with clear breakdown by test type

**Weaknesses:**
- Ground truth (market regimes) defined by researcher
- Ornstein-Uhlenbeck fitting is common but assumptions not validated

**Red Flags:**
- "R can be trusted as decision input" is a strong claim from limited validation


### Q29: Numerical Stability - **ENGINEERING**

This is NOT a scientific question. It documents a standard numerical stability fix (epsilon floor for division by zero).

- No hypothesis to test
- No ground truth needed
- Implementation guidance only

**Verdict:** Correctly labeled as "SOLVED engineering, not open science"


### Q30: Approximations - **ENGINEERING**

This is engineering optimization, not scientific hypothesis testing.

**Strengths:**
- Clear speedup measurements (100-300x)
- 100% gate accuracy on test cases
- Pareto frontier analysis

**Weaknesses:**
- Test data is synthetic (clustered/random embeddings)
- "Gate accuracy" depends on R threshold calibration

**Verdict:** Valid engineering but not science


### Q52: Chaos Theory - **PARTIAL**

**Strengths:**
- Pre-registered: "R inversely correlates with Lyapunov (r < -0.5)"
- Falsification criterion: "|r| < 0.3"
- Honestly reports hypothesis FALSIFIED (actual r = +0.545, opposite direction)
- Clear explanation of why original intuition was wrong
- Negative control (random noise) included

**Weaknesses:**
- Used synthetic data (Logistic map, Henon attractor)
- No real-world chaotic systems tested
- Lorenz attractor mentioned but not rigorously tested

**Red Flags:**
- Pivots from "R detects chaos" to "R measures attractor dimension" - valid but post-hoc


### Q53: Pentagonal Phi Geometry - **PARTIAL**

**Strengths:**
- Pre-registered pentagonal/phi hypotheses
- CONFIRMATION BIAS AUDIT explicitly conducted
- Honestly reports 4/5 tests FAIL
- Changes verdict from "SUPPORTED" to "PARTIAL"
- Clear distinction between what IS vs IS NOT supported

**Weaknesses:**
- Only 1/5 tests pass (72-degree clustering)
- Original "SUPPORTED" verdict was overstated
- Phi connection completely unsupported

**Red Flags:**
- The 72-degree finding may be coincidental to semantic clustering
- "Mean angle 75 deg" is called "72-degree clustering" - slight spin


---

## ANTI-PATTERNS DETECTED

### 1. Synthetic Data Overuse
**Questions affected:** Q22, Q25, Q30, Q52

Using synthetic data is appropriate for initial exploration but claiming "validation" on self-generated data is circular. Q25 is particularly egregious: sigma is optimized on R, then R^2 is computed on sigma predictions.

### 2. Post-Hoc Hypothesis Modification
**Questions affected:** Q52, Q53

When results contradict hypotheses, new interpretations are proposed. This is scientifically valid (hypothesis refinement) but should be clearly labeled as exploratory, not confirmatory.

### 3. Internal Ground Truth
**Questions affected:** Q22, Q24, Q25

When the "ground truth" is derived from or dependent on the R formula itself, validation is circular. Q22's F1 scores depend on R thresholds. Q24's "success rate" is R improvement.

### 4. Simpson's Paradox Masked
**Questions affected:** Q19

The overall r=0.52 correlation hides that 2/3 data sources show NEGATIVE correlations. The status "PASS" is technically correct but misleading.

### 5. Small Sample Pivoting
**Questions affected:** Q24

Drawing strong conclusions ("CHANGE_FEATURES works!") from n=17 low-R periods is premature.


---

## COMMENDATIONS

### Good Scientific Practice Observed:

1. **Q16, Q23:** Honest falsification when hypotheses fail
2. **Q20, Q53:** Self-auditing for confirmation bias
3. **Q26:** Corrections log documenting underpowered original test
4. **Q52:** Reversing direction of correlation reported honestly
5. **Q19:** Simpson's paradox risk flagged in document
6. **Q28:** Multiple Lyapunov exponent tests with clear threshold


---

## RECOMMENDATIONS

1. **External Benchmarks Required:** Q22, Q25 need validation on real labeled datasets (not synthetic)

2. **Simpson's Paradox Resolution:** Q19 should either:
   - Change status to PARTIAL
   - OR conduct within-source validation with proper ground truth

3. **Circular Validation Cleanup:** Q25's sigma optimization needs external validation

4. **Sample Size Requirements:** Q24 needs larger sample before claiming strategy effectiveness

5. **Status Clarity:** "CONDITIONALLY CONFIRMED" (Q19) and "PARTIAL" (Q53) should use consistent terminology


---

## CONCLUSION

The research shows a mix of rigorous and problematic practices. The best questions (Q16, Q23, Q26, Q28, Q52) demonstrate proper falsificationist science with pre-registration, real data, and honest reporting of failures. The weaker questions (Q22, Q25) suffer from circular validation on synthetic data.

**Key Improvement:** All "RESOLVED" or "CONFIRMED" questions should pass the test: "Can this be replicated on external data that R/the researchers never touched?"

Currently, only Q16 and Q19 clearly pass this test. The others require either:
- External benchmark validation
- Downgrade to PARTIAL/EXPLORATORY status
- Explicit acknowledgment of circular validation

---

*Audit completed: 2026-01-27*
*Auditor: Claude Opus 4.5*
