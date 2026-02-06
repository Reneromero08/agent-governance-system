# Phase 5 Verdict: 5-Q25 -- Sigma Universality (PARTIAL)

**Date:** 2026-02-05
**Reviewer:** Adversarial skeptic (Phase 5)
**Target:** `THOUGHT/LAB/FORMULA/questions/lower_q25_1260/q25_what_determines_sigma.md`
**Audit:** `THOUGHT/LAB/FORMULA/questions/lower_q25_1260/reports/DEEP_AUDIT_Q25.md`
**Verification:** `THOUGHT/LAB/FORMULA/questions/lower_q25_1260/reports/VERIFY_Q25.md`
**References reviewed:** GLOSSARY.md, SPECIFICATION.md, HONEST_FINAL_STATUS.md, DERIVATION_SIGMA.md

---

## Summary Verdict

```
Q25: Sigma Universality (R=1260)
- Claimed status: PARTIAL (formerly RESOLVED, downgraded)
- Proof type: empirical (synthetic + real data) + post-hoc curve fit
- Logical soundness: SEVERE GAPS
- Claims match evidence: OVERCLAIMED (primary claim contradicted by real data)
- Dependencies satisfied: MISSING [8e conservation (P4: numerology), CP^n assumption (unproven), Chern derivation (circular)]
- Circular reasoning: DETECTED [see Section 2]
- Post-hoc fitting: DETECTED [see Sections 1, 3]
- Numerology: DETECTED [see Section 1]
- Recommended status: FALSIFIED (predictability) / OPEN (derivation)
- Recommended R: 300-400 (down from 1260)
- Confidence: HIGH
- Issues: See detailed analysis below
```

---

## Evaluation Question 1: Is sigma = e^(-4/pi) Derivable from First Principles, or Is This Numerology?

### The Claim

Conjecture 4.1 in SPECIFICATION.md states: "The noise floor sigma = e^(-4/pi) ~ 0.2805 is a universal constant arising from the solid angle geometry of high-dimensional spheres."

### The Verdict: NUMEROLOGY. Not derivable from first principles in any legitimate sense.

#### 1.1 The Derivation Is Textbook Post-Hoc Fitting

DERIVATION_SIGMA.md is the key document. It is remarkably, almost admirably, transparent about the process. It explicitly lists SEVEN candidate formulas for sigma:

| Candidate | Formula | Predicted | Error vs 0.27 |
|-----------|---------|-----------|----------------|
| A | 6/pi^2 | 0.608 | 123% |
| B | pi/4 | 0.785 | 188% |
| C | 2/(3e) | 0.245 | 9.3% |
| D | 1/4 | 0.250 | 7.4% |
| E | 2/7 | 0.286 | 5.9% |
| F | 1/phi^2 | 0.382 | 41% |
| **G** | **e^(-4/pi)** | **0.2805** | **3.9%** |

Derivation G "won" because it had the smallest error. This is not a derivation. This is a fitting competition. The document itself admits (Part II, section G): "What value gives 0.27? ... ln(0.27) = -1.309. Is -1.309 special? 1.309 ~ 4/pi = 1.273 (off by 2.8%)." The author literally computed ln(0.27), then searched for mathematical expressions close to 1.309. This is the dictionary definition of post-hoc numerology.

A genuine first-principles derivation would:
1. Start from axioms or physical principles
2. Arrive at sigma = e^(-4/pi) through deduction
3. Then compare to the observed value

What actually happened:
1. Observed sigma ~ 0.27
2. Computed ln(0.27) = -1.309
3. Searched for nice-looking expressions near -1.309
4. Found 4/pi = 1.273 is close
5. Constructed a "solid angle geometry" narrative to justify it

#### 1.2 The "Solid Angle Geometry" Derivation Is Incoherent

DERIVATION_SIGMA.md Part III attempts a proof sketch. Let me trace it step by step:

Step 1-2: "8 octants form a cubic structure. Decoherence spreads uniformly on the unit sphere." -- Acceptable premises, if one grants the Peircean octant framework (which Phase 4 verdict_4_Q50 already identified as unfounded narrative).

Step 3: Average distance calculation. The integral is correct but irrelevant -- the result (=1 normalized) is never used in subsequent steps.

Step 4: Solid angle of one octant = 4*pi/8 = pi/2. Correct arithmetic.

Step 5: "Linear decay rate per octant = Omega_octant / pi = 1/2." Where does dividing by pi come from? This step has no justification. It is not a standard result from any branch of geometry or physics. The document just divides by pi because it needs to arrive at specific numbers later.

Step 6: Attempts gamma_total = 7 * gamma_octant / (2*pi) = 7/(4*pi). But this does NOT give 4/pi. The document explicitly notes: "But we need 4/pi, not 7/(4*pi)."

Step 7: "Key insight: The decay happens in BOTH directions..." giving gamma = 2 * 2/pi = 4/pi. This is not an insight. This is adjusting the formula. Step 6 gave 7/(4*pi) = 0.557. Step 7 abandons the calculation from steps 4-6 entirely and introduces a completely different factor of "2 from bidirectionality, 2/pi from solid angle averaging." Where did the factor of 7 go? Where did the octant counting go? The "key insight" is a non-sequitur that produces the desired answer.

Step 8: sigma = e^(-4/pi). QED, "modulo the 3.9% experimental discrepancy."

**This is not a derivation. It is a narrative that changes its own rules mid-stream to arrive at a predetermined target.** Steps 1-6 build toward one calculation, which fails. Step 7 discards that work and introduces new, unjustified factors that happen to give the right answer.

#### 1.3 The "First Principles" Are Themselves Unproven

The derivation depends on:
- "Peircean 3D semiotic space" -- a philosophical construct, not a mathematical axiom (see P4-Q50-04)
- "8 octants" from Peirce's categories -- inconsistently mapped across models (see P4-Q50-04)
- "Gaussian decoherence" -- an assumption, not derived
- CP^(d-1) manifold structure -- assumed without justification for real embeddings (see P4-Q50-05)

You cannot derive a "first-principles" result from premises that are themselves unproven, especially when those premises were chosen to fit the data.

#### 1.4 HONEST_FINAL_STATUS.md Already Knows This

The project's own internal assessment is unambiguous:

> "sigma = 0.27 derived from first principles: POST-HOC, 20% confidence"
> "sigma = e^(-4/pi) = 0.2805 [is a] POST-HOC FIT (not derived)"
> "DERIVATION_SIGMA.md explicitly lists 7 candidate derivations and picks the one that fits best. This is textbook confirmation bias."

The GLOSSARY.md is also honest: "Post-hoc fit: sigma = e^(-4/pi) = 0.2805 has been proposed but is a curve fit to observed data, not an independent derivation."

The evidence from within the project itself is clear: this is not a derivation.

**VERDICT: sigma = e^(-4/pi) is NUMEROLOGY. The "derivation" is a post-hoc curve fit dressed in geometric language, with an incoherent proof sketch that changes its own methodology mid-argument to hit a target value.**

---

## Evaluation Question 2: Why e^(-4/pi) Specifically? What Would Change If Sigma Were 0.3 or 0.2?

### The Sensitivity Problem

The formula R = (E / grad_S) \* sigma^Df uses sigma as an exponent base raised to a power Df. Since Df is large (the GLOSSARY reports Df ~ 43.5 for the "derived" case where alpha = 1/2 and Df \* alpha = 8e), the formula is EXPONENTIALLY sensitive to sigma:

```
sigma = 0.2805:  sigma^43.5 = 0.2805^43.5 = ~1.44e-24
sigma = 0.30:    0.30^43.5 = ~3.63e-23     (25x larger)
sigma = 0.20:    0.20^43.5 = ~2.87e-30     (2 million times smaller)
sigma = 0.27:    0.27^43.5 = ~3.48e-25     (4x smaller than 0.2805^43.5)
```

A 3.9% change in sigma produces an ORDER OF MAGNITUDE change in sigma^Df. This means:

1. **The 3.9% error is catastrophic, not negligible.** When sigma appears as an exponent base, 3.9% error in sigma translates to enormous error in R. The claim "within experimental error" in DERIVATION_SIGMA.md is false for any context where sigma^Df is computed.

2. **The formula cannot distinguish sigma = e^(-4/pi) from sigma = 0.27.** At Df = 43.5, these two values produce R values differing by a factor of ~4. A "universal constant" that cannot be pinned down to better than 4x in the output is not useful.

3. **The formula cannot distinguish sigma = e^(-4/pi) from sigma = 2/7 or sigma = 1/4.** All candidate values from DERIVATION_SIGMA.md produce wildly different R values at large Df, but none of them can be empirically distinguished because the observed sigma range itself (0.27 +/- noise) spans all of them.

### What This Reveals

The fact that sigma = e^(-4/pi) vs sigma = 0.27 vs sigma = 1/4 vs sigma = 2/7 are all "within experimental error" means the formula has NO discriminating power between these candidates. Any of them could be "the answer" and the data cannot tell you which one. Calling one of them "the" answer based on which one has the smallest percentage error in sigma (ignoring the exponential amplification in sigma^Df) is misleading.

**VERDICT: The specific value e^(-4/pi) is not meaningfully distinguished from several other candidates. The exponential sensitivity of sigma^Df means the 3.9% "small" error in sigma is actually a devastating error in R. The choice of e^(-4/pi) over 1/4 or 2/7 is arbitrary.**

---

## Evaluation Question 3: Is Sigma Actually Universal Across Domains, or Does It Vary?

### What Q25 Itself Found

The Q25 investigation is the most damaging evidence against sigma universality, and it comes from within the project.

#### 3.1 Synthetic Data: Sigma Varies by 24x

From q25_what_determines_sigma.md:
- Min sigma: 4.15
- Max sigma: 100.0
- Range ratio: 24x

This is not a "universal constant." This is a free parameter that varies by more than an order of magnitude across datasets. The GLOSSARY states sigma is in (0, 1), but the Q25 synthetic test finds sigma values up to 100.0. These are obviously incompatible definitions.

#### 3.2 Real Data: Sigma Is Domain-Specific

From DEEP_AUDIT_Q25.md and VERIFY_Q25.md:

| Domain | Sigma Range |
|--------|-------------|
| NLP (8 datasets) | 1.92 - 2.72 |
| Gene expression (1 dataset) | 39.44 |

Within NLP, sigma clusters tightly (CV = 0.067). But between domains, sigma differs by 15x. The VERIFY report is explicit: "Within-domain sigma is near-constant. The 'high R^2' comes from the fact that log_n_dimensions perfectly separates NLP (D=384, sigma~2.7) from market (D=12, sigma~9.7)."

#### 3.3 The "Resolution" Test Is Spurious

The resolution test claimed R^2_cv = 0.9855, but VERIFY_Q25.md demonstrates this is because the model learned a trivial mapping: "384-dim -> sigma~2.7, 12-dim -> sigma~9.7." This is not sigma being predicted from fundamental properties; it is sigma being predicted from which pre-trained embedding model was used. The "universal" sigma is actually "whatever the embedding model's architecture imposes."

#### 3.4 The Contradiction Is Fatal

The GLOSSARY says sigma is in (0, 1) with empirical value ~0.27.
Q25 synthetic tests find sigma ranging from 4.15 to 100.
Q25 real data finds sigma from 1.92 to 39.44.
Conjecture 4.1 says sigma = e^(-4/pi) = 0.2805 is universal.

These cannot all be true. Either:
- (a) sigma is a universal constant near 0.27, and the Q25 results measuring sigma = 2-100 are using a different definition
- (b) sigma varies by domain/dataset, and Conjecture 4.1 is false
- (c) the Q25 tests and the GLOSSARY are talking about different quantities called "sigma"

The most likely explanation is (c): the "sigma" in Q25's predictive formula (which ranges 4-100) is NOT the same quantity as the "sigma" in the GLOSSARY (which is in (0,1)). But then Q25 is not actually investigating the universality of the sigma from R = (E/grad_S) * sigma^Df. It is investigating a different parameter that happens to share a name. This is a definitional confusion that undermines the entire Q25 investigation.

**VERDICT: Sigma is NOT universal across domains. Real data shows domain-specific clustering with 15x inter-domain variation. The within-domain constancy is an artifact of using identical embedding models. The Q25 investigation may be conflating two different quantities called "sigma."**

---

## Evaluation Question 4: Is 3.9% Error Good Enough for a "Universal Constant"?

### Context: What Does 3.9% Mean for Actual Constants?

| Constant | Precision | Source |
|----------|-----------|--------|
| Speed of light c | exact (defined) | SI definition |
| Fine structure constant | 0.00000015% | QED measurements |
| Gravitational constant G | 0.0022% | CODATA 2018 |
| Boltzmann constant k | exact (defined) | SI 2019 revision |
| Hubble constant H_0 | ~2-4% | Planck/SH0ES tension |
| Zipf's law exponent | 10-20% variation | Corpus-dependent |
| Benford's law | ~5-15% for leading digits | Dataset-dependent |

At 3.9% error, sigma = e^(-4/pi) is comparable to:
- The Hubble constant tension (where 4% discrepancy is considered a CRISIS in cosmology)
- Benford's law (a statistical regularity, not a universal constant)
- Zipf's law (nobody calls this a "universal constant")

A 3.9% error is NOT acceptable for something labeled a "universal constant arising from geometry." It IS acceptable for an empirical regularity or approximate scaling law.

### The Error Is Worse Than Stated

The 3.9% is computed as: |0.2805 - 0.27| / 0.27 = 3.9%. But:

1. **What is the uncertainty on 0.27?** The observed value "0.27" has its own error bars, which are never reported. If sigma_observed = 0.27 +/- 0.02 (a reasonable error bar for a noisy empirical fit), then the 3.9% discrepancy is within noise and the "match" is not even meaningful -- any value from 0.25 to 0.29 would "match" within error bars.

2. **How many candidates were tried?** Seven (DERIVATION_SIGMA.md). When you try 7 candidates and pick the best, the probability that the best one is within 4% of the target is high even for random guessing. For a uniform search over [0, 1] with 7 random formulas, the expected minimum distance to 0.27 is approximately 1/(7+1) = 12.5%. Getting 3.9% is better than random but not dramatically so.

3. **The exponential amplification.** As computed in Section 2, at Df ~ 43.5, the 3.9% error in sigma becomes a factor-of-4 error in sigma^Df. The constant is "universal" only if you never use it in the formula it was designed for.

### The Specification Knows This

SPECIFICATION.md Conjecture 4.1 correctly marks this OPEN. The GLOSSARY correctly states "EMPIRICAL OBSERVATION. Not derived from first principles." The HONEST_FINAL_STATUS.md assigns 20% confidence to "sigma = 0.27 derived from first principles."

The only document that claims this is "derived" is DERIVATION_SIGMA.md, which is filed under `_archive/failed_derivations/` -- the filing location itself is a confession.

**VERDICT: 3.9% error is NOT good enough for a "universal constant." It IS consistent with an approximate empirical regularity. The error is further understated because (a) the observed value has unreported uncertainty, (b) the formula was selected from 7 candidates, and (c) the error amplifies exponentially in the formula context.**

---

## Section 5: Additional Issues

### 5.1 Q25 Status Is Still Wrong

Q25 is marked "PARTIAL" but the main file (q25_what_determines_sigma.md) still says "HYPOTHESIS CONFIRMED: Sigma is predictable from dataset properties with R^2_cv = 0.8617 > 0.7."

Both the DEEP_AUDIT and VERIFY reports found this claim is FALSIFIED by real data (R^2_cv = 0.0). The status should be FALSIFIED, not PARTIAL.

### 5.2 The Q25 Investigation Answers a Different Question Than Conjecture 4.1

Q25 asks: "Is sigma predictable from dataset properties?"
Conjecture 4.1 asks: "Is sigma a universal constant?"

These are opposite questions. If sigma is a universal constant, it does NOT depend on dataset properties -- it is always ~0.27. If sigma is predictable from dataset properties (R^2 = 0.86), it VARIES across datasets and is NOT a universal constant.

Q25's positive result (if it held on real data, which it does not) would actually FALSIFY Conjecture 4.1. And Conjecture 4.1 being true would make Q25's predictive formula pointless (you do not need a regression model to predict a constant). This logical contradiction is never acknowledged.

### 5.3 Circular Optimization in Q25

VERIFY_Q25.md identifies a critical circularity: "optimal sigma" is defined as the value that minimizes bootstrap CV of R. But R is a direct function of sigma. So "finding optimal sigma" means "finding the sigma that makes R most stable," which is a property of the R formula, not of the data. The subsequent regression predicting sigma from data properties may be recovering the optimization criterion rather than a fundamental relationship.

### 5.4 The Sigma in the GLOSSARY vs. Q25

GLOSSARY: sigma is in (0, 1), sigma ~ 0.27.
Q25: sigma ranges from 4.15 to 100.

These are irreconcilable unless they are different quantities. The GLOSSARY sigma is a dimensionless noise floor parameter in R = (E/grad_S) \* sigma^Df. The Q25 sigma appears to be a bandwidth or scale parameter in a Gaussian kernel (from the code: R = mean(exp(-0.5 \* (error/sigma)^2))). These are related but distinct parameters. The entire Q25 investigation may be irrelevant to Conjecture 4.1 because of this definitional slippage.

---

## Section 6: What Q25/Conjecture 4.1 Gets Right

In fairness:

1. **The GLOSSARY and SPECIFICATION are honest.** They correctly label sigma as "EMPIRICAL OBSERVATION. Not derived from first principles" and the proposed formula as "a curve fit to observed data."

2. **HONEST_FINAL_STATUS.md is brutally accurate.** Its assessment (20% confidence, post-hoc fit) is exactly right.

3. **Q25 has a real pre-registration.** The hypothesis and falsification criteria were stated before testing. This is good methodology.

4. **Q25 tested on real data.** Unlike many other questions in this framework, Q25 actually ran tests on external datasets (HuggingFace, NCBI GEO). The fact that these tests falsified the hypothesis is a feature, not a bug -- it shows the methodology can produce negative results.

5. **The DEEP_AUDIT found the problems.** The internal audit correctly identified every major issue. The project's self-assessment mechanism works.

---

## Section 7: Inherited Issues from Phases 1-4

| Phase | Issue | Impact on Q25/Conjecture 4.1 |
|-------|-------|-------------------------------|
| P1 | 5+ incompatible E definitions | The R formula in Q25 uses a different E than the GLOSSARY (kernel-based vs. cosine similarity) |
| P1 | All evidence synthetic | Q25 synthetic results are unreliable; real data results falsify the hypothesis |
| P2 | Quantum interpretation falsified | The "solid angle geometry" derivation of sigma relies on CP^n, which inherits the Q44 Born rule problem (synthetic only) |
| P3 | R numerically unstable | sigma^Df is exponentially sensitive to sigma; 3.9% error in sigma is catastrophic for R |
| P4 | 8e = numerology | The derivation of sigma depends on the Peircean octant framework, which Phase 4 identified as unfounded narrative |
| P4 | Non-independent model samples | If sigma's "universality" is tested on correlated models, the evidence base shrinks to 2-3 independent observations |

---

## Section 8: Internal Contradictions

### 8.1 Q25 Main File vs. Audit Reports

- Main file: "HYPOTHESIS CONFIRMED: Sigma is predictable"
- DEEP_AUDIT: "STATUS INCORRECT - Verdict: FALSIFIED BY REAL DATA"
- VERIFY: "CRITICAL ISSUES IDENTIFIED - EVIDENCE CONFLICTED"

### 8.2 GLOSSARY sigma Domain vs. Q25 sigma Range

- GLOSSARY: 0 < sigma < 1
- Q25 results: sigma from 4.15 to 100

### 8.3 Q25 Predictability vs. Conjecture 4.1 Universality

- Q25 tries to prove sigma depends on dataset properties
- Conjecture 4.1 claims sigma is a universal constant
- Both cannot be true simultaneously

### 8.4 DERIVATION_SIGMA.md vs. HONEST_FINAL_STATUS.md

- DERIVATION_SIGMA: "sigma = e^(-4/pi) = 0.2805 is the most theoretically motivated derivation" -- files this under "THE R FORMULA - FULLY DERIVED PARAMETERS"
- HONEST_FINAL_STATUS: "sigma = e^(-4/pi) has fundamental significance: POST-HOC FIT, 20% confidence"
- DERIVATION_SIGMA is filed in `_archive/failed_derivations/`, which is honest but contradicts its own conclusion

---

## Final Assessment

The sigma universality claim collapses under examination from multiple directions simultaneously:

1. **The derivation is post-hoc numerology.** Seven candidates were tried; the best-fitting one was labeled "derived from first principles." The proof sketch is internally inconsistent, changing its own rules mid-argument to hit the target value.

2. **The "universal" constant is not universal.** Real data shows sigma varying by 15x across domains. Synthetic data shows 24x variation. The claimed constancy (~0.27) holds only within a single embedding model family.

3. **The predictability claim is falsified.** R^2_cv = 0.0 on real data. The positive result (R^2_cv = 0.86) is from synthetic data designed with controlled properties. The "resolution" (R^2_cv = 0.99) is a spurious correlation with embedding dimensionality.

4. **The 3.9% error is both larger than stated and smaller than meaningful.** It understates the exponential amplification through sigma^Df. And it overstates the significance because 7 candidates were tried.

5. **The investigation conflates two different "sigma" quantities** with different definitions, domains, and ranges.

6. **The project's own internal assessments agree with this verdict.** HONEST_FINAL_STATUS.md, the GLOSSARY, and the SPECIFICATION all correctly identify sigma = e^(-4/pi) as empirical/post-hoc.

**Recommended status for Conjecture 4.1: FALSIFIED as a derivation. OPEN as an unexplained approximate regularity.**

**Recommended status for Q25: FALSIFIED (predictability hypothesis disproven by real data, R^2_cv = 0.0).**

**Recommended R: 300-400** (down from 1260). The Q25 investigation was well-structured with real pre-registration and honest testing on external data, which is methodologically commendable. But the conclusion does not match the evidence, and the sigma universality claim is not supported.

---

## Appendix: Issue Tracker Additions

| ID | Issue | Severity | Source | Affects |
|----|-------|----------|--------|---------|
| P5-Q25-01 | sigma = e^(-4/pi) is post-hoc curve fit from 7 candidates, not a derivation | CRITICAL | DERIVATION_SIGMA.md | Conjecture 4.1 |
| P5-Q25-02 | Solid angle "proof sketch" is internally inconsistent (steps 1-6 abandoned at step 7) | CRITICAL | DERIVATION_SIGMA.md Part III | sigma derivation |
| P5-Q25-03 | Real data R^2_cv = 0.0 falsifies sigma predictability but status not updated | CRITICAL | DEEP_AUDIT_Q25, VERIFY_Q25 | Q25 status |
| P5-Q25-04 | GLOSSARY sigma in (0,1) vs Q25 sigma in [4,100] -- definitional confusion | HIGH | GLOSSARY vs Q25 test code | Q25 relevance to Conjecture 4.1 |
| P5-Q25-05 | Q25 predictability and Conjecture 4.1 universality are logically contradictory | HIGH | Logical analysis | Both claims |
| P5-Q25-06 | 3.9% error in sigma amplifies exponentially to ~400% error in sigma^Df | HIGH | Mathematical analysis | R formula usability |
| P5-Q25-07 | Sigma varies 15x across domains in real data (not universal) | CRITICAL | VERIFY_Q25.md | Conjecture 4.1 |
| P5-Q25-08 | Circular optimization: "optimal sigma" defined by the metric it is supposed to predict | HIGH | VERIFY_Q25 Section 2.1 | Q25 methodology |
| P5-Q25-09 | Resolution test R^2=0.99 is spurious (learns embedding dimension, not sigma principles) | HIGH | VERIFY_Q25 Section 5.1 | Q25 resolution claim |
| P5-Q25-10 | Seven derivation candidates tried without multiplicity correction | MEDIUM | DERIVATION_SIGMA.md | Statistical significance |
| P5-Q25-11 | No error bars reported on observed sigma=0.27 | MEDIUM | All sigma documents | All sigma claims |
| P5-Q25-12 | DERIVATION_SIGMA.md filed in `failed_derivations/` yet concludes "FULLY DERIVED" | MEDIUM | DERIVATION_SIGMA.md | Internal consistency |

---

*Phase 5 adversarial review completed: 2026-02-05*
*No charitable interpretations. Evidence weighed as presented.*
