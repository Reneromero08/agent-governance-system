# Phase 6D Verdict: Engineering / Utility Batch (5 Questions)

**Date:** 2026-02-05
**Reviewer:** Adversarial skeptic (Phase 6D)
**Batch:** Q26, Q29, Q30, Q31, Q46

---

## Inherited Issues (Phases 1-5)

These inherited defects apply to all questions in this batch:

| Phase | Issue | Relevance to Batch 6D |
|-------|-------|-----------------------|
| P1 | 5+ incompatible E definitions | Q29, Q30 use E in R = E/sigma without specifying which E |
| P1 | Quantum interpretation falsified | Q31 still references "Fubini-Study metric" and "geodesic flow" from Q43 (labeled OVERCLAIMED in P3) |
| P3 | sigma^Df = 10^47 overflow | Q29 addresses sigma=0 but NOT the sigma^Df overflow, which is the actual severe instability |
| P3 | R numerically unstable | Q29's epsilon fix is necessary but insufficient (see verdict below) |
| P4 | 8e = numerology | Q31 references Df=22 which feeds into the 8e numerology chain |

---

# Q26: Minimum Data Requirements (R=1240)

**Target:** `THOUGHT/LAB/FORMULA/questions/lower_q26_1240/q26_minimum_data_requirements.md`
**Audit:** `THOUGHT/LAB/FORMULA/questions/lower_q26_1240/reports/DEEP_AUDIT_Q26.md`
**Verification:** `THOUGHT/LAB/FORMULA/questions/lower_q26_1240/reports/VERIFY_Q26.md`

```
Q26: Minimum Data Requirements (R=1240)
- Claimed status: RESOLVED
- Proof type: Empirical (multi-model bootstrap, real embeddings)
- Logical soundness: ADEQUATE (within narrow scope)
- Claims match evidence: MOSTLY MATCHED (one overclaim, see below)
- Dependencies satisfied: NONE NEEDED (self-contained empirical test)
- Circular reasoning: MINOR (see Section 1.3)
- Post-hoc fitting: DETECTED (semantic structure finding is post-hoc)
- Numerology: NONE
- Recommended status: RESOLVED (with caveats)
- Recommended R: 1100 (minor downgrade)
- Confidence: HIGH
- Issues: See detailed analysis below
```

## 1. Model Selection Methodology

### 1.1 What Was Done Well

Q26 is genuinely one of the better-executed questions in this project. The self-correction from an underpowered single-model test to a rigorous multi-model retest demonstrates intellectual honesty. Key strengths:

- Pre-registration of hypotheses with explicit falsification criteria
- Real embedding models (sentence-transformers), not synthetic data
- Multi-model validation (7 configurations)
- Honest admission that the original test was "SPIN and UNDERPOWERED"
- Bootstrap methodology with 50 trials (adequate for CV estimation)

### 1.2 Are Minimum Data Requirements Derived or Arbitrary?

**EMPIRICAL, NOT DERIVED. The threshold is arbitrary.**

The N_min determination uses a CV < 0.10 threshold (10% coefficient of variation). This threshold is:
- Not derived from any theoretical bound
- Not justified by application requirements
- A round number chosen by convention

If the threshold were 0.05 instead of 0.10, N_min would increase. If 0.20, N_min would be even smaller (possibly N=2). The reported N_min = 3 is an artifact of the threshold choice, not a fundamental property of the embedding space.

A genuine sample complexity bound would look like: "For accuracy delta with probability 1-epsilon, N >= f(d, delta, epsilon)." No such bound is provided. The document does not even attempt one. It substitutes empirical bootstrap for theoretical analysis.

This is acceptable for engineering guidance but should not be called a "sample complexity bound" (the original question asks for one).

### 1.3 Subtle Circularity in the "No Scaling with D" Claim

The claim "N_min is constant at 3 regardless of D" has a methodological issue. The PCA projections (D=50, 100, 200, 400) are all derived from the same base model (all-mpnet-base-v2, D=768). PCA does not create independent data at different dimensionalities -- it projects the same data to lower dimensions. The intrinsic structure is preserved, so of course N_min does not change.

To genuinely test whether N_min scales with D, you would need independently trained models at genuinely different dimensionalities. The test has 3 independent models (MiniLM-L6-v2, mpnet-base-v2, paraphrase-MiniLM-L3-v2), all of which happen to produce D=384 or D=768. This is 2 distinct dimensionalities from 3 independent models -- far too few data points to establish "no scaling law." The correct conclusion is: "No scaling observed between D=384 and D=768 for sentence-transformer models." The claim "N_min does not scale with D" is overgeneralized.

### 1.4 The Semantic Structure Finding Is Post-Hoc

The discovery that coherent content needs N=5 while diverse content needs N=3 was NOT pre-registered. It emerged from the retest and is presented as a "NEW FINDING." While interesting, this is:
- A post-hoc observation (not pre-registered)
- Tested on a single model (MiniLM-L6-v2 only, per the semantic structure test)
- Not validated on held-out data or independent corpora

The VERIFY report correctly flags this: "Semantic structure effect discovered post-hoc (should be pre-registered)." But this limitation is not prominently stated in the main document.

### 1.5 What the Evidence Actually Supports

| Claim | Verdict | Evidence |
|-------|---------|----------|
| N_min does not scale with D | OVERGENERALIZED | Only 2 distinct D values (384, 768) from independent models |
| N_min = 3 for diverse content | SUPPORTED | 7 configurations confirm |
| N_min = 5 for coherent content | WEAKLY SUPPORTED | Single model, post-hoc |
| N = 5-10 sufficient in practice | SUPPORTED | Conservative and reasonable |
| No scaling law of any kind | SUPPORTED | R^2 = 0.0 for all fits (but on too-narrow D range) |

### 1.6 Missing from Q26

1. **No theoretical sample complexity bound** -- the original question asks for one
2. **No analysis of embedding model dependence** -- what happens with OpenAI embeddings, Cohere, etc.?
3. **No analysis of the CV=0.10 threshold choice** -- sensitivity analysis absent
4. **No connection to the R formula** -- N_min for what? For stable R computation? The document never specifies what "reliable gating" means quantitatively

**VERDICT: RESOLVED is acceptable but overstated. The practical guidance (use N >= 5) is sound. The theoretical claims (no scaling law) are underpowered. The semantic structure finding needs pre-registered replication.**

---

# Q29: Numerical Stability (R=1180)

**Target:** `THOUGHT/LAB/FORMULA/questions/engineering_q29_1180/q29_numerical_stability.md`
**Audit:** `THOUGHT/LAB/FORMULA/questions/engineering_q29_1180/reports/DEEP_AUDIT_Q29.md`
**Verification:** `THOUGHT/LAB/FORMULA/questions/engineering_q29_1180/reports/VERIFY_Q29.md`

```
Q29: Numerical Stability (R=1180)
- Claimed status: SOLVED
- Proof type: Engineering (standard epsilon floor)
- Logical soundness: ADEQUATE for what it addresses; CRITICALLY INCOMPLETE for the actual problem
- Claims match evidence: MATCHED (within scope); SCOPE IS WRONG
- Dependencies satisfied: N/A (standalone engineering)
- Circular reasoning: NONE
- Post-hoc fitting: NONE
- Numerology: NONE
- Recommended status: SOLVED (for div/0 only); the REAL stability problem is UNADDRESSED
- Recommended R: 400 (down from 1180)
- Confidence: HIGH
- Issues: See detailed analysis below
```

## 1. What Q29 Actually Solves

Q29 addresses division by zero when sigma = 0 in R = E / sigma. The solution -- `R = E / max(sigma, epsilon)` with epsilon = 1e-6 -- is correct, standard, and well-tested. Eight edge cases pass. The epsilon value is appropriate. The code is clean. No complaints about the engineering itself.

## 2. What Q29 Does NOT Solve -- And This Is the Critical Gap

**Q29 solves the TRIVIAL numerical stability problem and ignores the CATASTROPHIC one.**

Phase 3 identified that the full R formula includes sigma^Df, where Df can be large (Df = 22 for BERT, Df = 43.5 in other contexts). The sigma^Df term is where the real numerical disaster lives:

```
sigma = 0.27:   sigma^22 = 0.27^22 = 2.7e-13
sigma = 0.28:   sigma^22 = 0.28^22 = 1.2e-12    (4.4x larger)
sigma = 0.30:   sigma^22 = 0.30^22 = 1.3e-11    (48x larger)

sigma = 0.27:   sigma^43.5 = 3.5e-25
sigma = 0.30:   sigma^43.5 = 3.6e-23    (100x larger)
```

A 3.7% change in sigma (0.27 to 0.28) produces a 4.4x change in sigma^22. An 11% change (0.27 to 0.30) produces a 48x change. At Df = 43.5, these become 25x and 1000x respectively.

**The epsilon floor does nothing for this.** When sigma is 0.27 vs 0.28, both are far above epsilon = 1e-6, so the floor never activates. But the output changes by orders of magnitude. This is the REAL numerical stability problem, and Q29 completely ignores it.

### 2.1 The Validation Table Exposes the Problem

From Q29's own edge case table:

| Scenario | E | sigma | R_eps |
|----------|---|-------|-------|
| Identical embeddings | 1.0 | 0.0 | 1,000,000 |
| High E, low sigma | 0.9997 | 0.0001 | 9,596 |

These are marked "PASS" because they produce finite numbers. But R = 1,000,000 and R = 9,596 are wildly different from typical R values (which are order 1-10 per other documents). A "stable" computation that returns values spanning 6 orders of magnitude depending on edge cases is not practically stable -- it is just not literally infinity.

### 2.2 What a Complete Stability Solution Would Look Like

1. **Sigma^Df overflow/underflow handling**: Log-space computation of sigma^Df (i.e., compute Df * log(sigma) and check for overflow before exponentiating)
2. **Condition number analysis**: How sensitive is R to perturbations in sigma as a function of Df?
3. **Clamping or normalization**: Bound R to a meaningful range (what R values are actually used in gating decisions?)
4. **Df-dependent epsilon**: If Df is large, the effective epsilon for sigma^Df is much larger than 1e-6
5. **Sensitivity to Df itself**: Df = 22 vs Df = 43.5 produces wildly different sigma^Df; which Df is canonical?

None of these are addressed.

### 2.3 The Mock Embedder Limitation

All Q29 tests use a mock embedder (hash-based deterministic), not real embeddings. The DEEP_AUDIT notes this is "appropriate for testing numerical stability." This is correct for the div/0 case but means the tests never encounter the ACTUAL numerical challenges that arise with real high-dimensional embeddings (e.g., near-degenerate covariance matrices, numerical rank deficiency, floating point accumulation errors in large similarity matrices).

**VERDICT: Q29 is SOLVED for the trivial problem (div/0) and UNADDRESSED for the serious problem (sigma^Df overflow and exponential sensitivity). The R-score of 1180 dramatically overstates the contribution. The answer is correct but the question was scoped too narrowly to matter.**

---

# Q30: Approximations (R=1160)

**Target:** `THOUGHT/LAB/FORMULA/questions/engineering_q30_1160/q30_approximations.md`
**Audit:** `THOUGHT/LAB/FORMULA/questions/engineering_q30_1160/reports/DEEP_AUDIT_Q30.md`
**Verification:** `THOUGHT/LAB/FORMULA/questions/engineering_q30_1160/reports/VERIFY_Q30.md`

```
Q30: Approximations (R=1160)
- Claimed status: RESOLVED - VALIDATED
- Proof type: Empirical (speed benchmarks on synthetic data)
- Logical soundness: ADEQUATE (within scope)
- Claims match evidence: MOSTLY MATCHED (one critical caveat insufficiently highlighted)
- Dependencies satisfied: PARTIAL [tested only on synthetic data; gate threshold hardcoded at 0.8]
- Circular reasoning: NONE
- Post-hoc fitting: NONE
- Numerology: NONE
- Recommended status: RESOLVED (with caveats)
- Recommended R: 1000 (minor downgrade)
- Confidence: HIGH
- Issues: See detailed analysis below
```

## 1. Are Error Bounds Derived or Empirical?

**ENTIRELY EMPIRICAL. No analytical error bounds are provided.**

The document reports:
- 100% gate accuracy for 7/8 methods
- Up to 250% R-value error for sampled methods
- Empirical scaling exponents (O(n^0.18) for fixed-k sampling)

None of these have analytical derivations. A proper treatment would provide:
- **Concentration inequality**: P(|R_sample - R_exact| > epsilon) <= delta as a function of k, n, and the distribution of pairwise similarities
- **Confidence intervals**: For a given k, what is the expected error in R?
- **Threshold proximity analysis**: If R is within delta of the gate threshold, what k guarantees correct gate decision with probability 1 - alpha?

The CLT justification ("sample statistics converge rapidly") is hand-waved. The CLT tells you that the sample mean converges as O(1/sqrt(k)), but:
- The pairwise similarities are NOT independent (they share embedding vectors)
- The effective sample size for k samples is k*(k-1)/2 pairs, but these pairs are highly correlated
- No analysis of the correlation structure is provided

### 1.1 The 250% R-Value Error Is a Showstopper for Non-Gate Applications

The document buries a critical admission: "the actual R value can have significant error (up to 250%)." This is mentioned in the Limitations section (line 149) but does NOT appear in the Executive Summary or Key Findings.

If R is used for anything beyond a binary gate decision -- ranking, interpolation, monitoring, comparison -- a 250% error makes the approximation useless. The document only validates gate accuracy (binary threshold), not R-value accuracy. Any claim of "100% accuracy" applies ONLY to the binary gate, not to R itself.

### 1.2 The Gate Accuracy Is Tested at a Single Threshold

All tests use `threshold = 0.8` (hardcoded). The VERIFY report flags this as "LOW SEVERITY" but it is more serious than stated. Gate accuracy near a threshold depends on the distribution of R values relative to that threshold. If R values cluster near the threshold, even small approximation errors flip decisions. The test scenarios are:
- "High agreement" (R well above threshold)
- "Low agreement" (R well below threshold)
- "Medium" and "Mixed"

But the critical case -- R values NEAR the threshold -- is not specifically tested. The "100% accuracy" may hold only because the synthetic test data produces R values that are far from 0.8 in all scenarios.

### 1.3 Synthetic Data Only

Both audit reports acknowledge this and call it "ACCEPTABLE." I disagree with "acceptable" if the claim is "100-300x speedup with 100% gate accuracy" as a general statement. Synthetic embeddings have controlled statistical properties that may not match real data. Specifically:
- Real embeddings often have heavy-tailed similarity distributions (some pairs very similar, most dissimilar)
- Cluster structure in real data can cause sampling bias (random sample may miss entire clusters)
- The document itself notes "Non-random distributions: If embeddings have strong structure, stratified sampling may be needed" but never tests this

### 1.4 What the Evidence Actually Supports

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Random sampling achieves 100-300x speedup | SUPPORTED | Empirically measured, reproducible |
| 100% gate accuracy at threshold 0.8 | SUPPORTED | On synthetic data with controlled agreement levels |
| Scaling is subquadratic | SUPPORTED | Empirical exponents match theory |
| "100% gate decision accuracy" (as general claim) | OVERSTATED | Single threshold, synthetic data only |
| R-value error up to 250% | SUPPORTED (and honestly reported) | Buried in limitations |
| Pareto frontier analysis | VALID | Clean methodology |

### 1.5 Credit Where Due

Q30 is well-structured engineering work:
- Multiple methods compared fairly (8 methods)
- Negative results reported (combined method's 83.3% accuracy)
- Scaling behavior empirically verified
- Pareto frontier analysis is clean
- Practical recommendations are reasonable

**VERDICT: RESOLVED is acceptable. The speedup claims are real. The "100% accuracy" claim is overstated due to single-threshold testing and synthetic-only data. The error bounds are empirical, not analytical. The 250% R-value error should be elevated to a primary finding, not buried in limitations. Solid engineering, not science.**

---

# Q31: Compass Mode (R=1550)

**Target:** `THOUGHT/LAB/FORMULA/questions/high_q31_1550/q31_compass_mode.md`
**Phase 3 dependency:** `THOUGHT/LAB/FORMULA/verification/phase_3/verdict_3_Q43.md`

```
Q31: Compass Mode (R=1550)
- Claimed status: CONFIRMED (via Q43 rigorous validation)
- Proof type: Theoretical (formula proposed, not tested)
- Logical soundness: SEVERE GAPS
- Claims match evidence: WILDLY OVERCLAIMED
- Dependencies satisfied: MISSING [Q43 is OVERCLAIMED per P3 verdict; no navigation test; no action-conditioned test; no multi-task validation]
- Circular reasoning: DETECTED [Q43 "confirms" Q31 via tautological eigenvalue decomposition]
- Post-hoc fitting: DETECTED [compass formula constructed after Q43 results observed]
- Numerology: NONE
- Recommended status: OPEN (not CONFIRMED)
- Recommended R: 400 (down from 1550)
- Confidence: HIGH
- Issues: See detailed analysis below
```

## 1. Does the Implementation Match the Specification?

**There IS no implementation. Q31 is a theoretical proposal with no working code.**

The question explicitly asks for "a reproducible construction where argmax_a R(s,a) yields reliable navigation." The document provides formulas but zero implementations. The "success criterion" requires navigation on "multiple task families (not just one graph family)." The document tests on ZERO task families.

### 1.1 What Q31 Claims Is "CONFIRMED"

The document states: "CONFIRMED (2026-01-10, via Q43 rigorous validation)" with a receipt hash.

What Q43 supposedly proves for Q31:
- "QGT eigenvectors = MDS eigenvectors" (96.1% subspace alignment)
- "Eigenvalue correlation = 1.000"
- "Principal axes = covariance eigenvectors"

But as established in the Phase 3 verdict on Q43:
- The 96.1% alignment is a **mathematical tautology** (SVD theorem), not an empirical discovery
- The eigenvalue correlation = 1.000 is **guaranteed by linear algebra**
- "Principal axes = covariance eigenvectors" is **the definition of PCA**

Q31 claims "CONFIRMED" based on Q43, but Q43's "confirmations" are tautologies. The chain is: Q31 says "compass = J x principal_axis_alignment" -> Q43 says "principal axes are covariance eigenvectors" (which is the definition of PCA) -> Q31 says "CONFIRMED."

This confirms nothing about whether the compass WORKS for navigation. It only confirms that PCA exists.

### 1.2 The Success Criterion Is Not Met

The Q31 question specifies three concrete deliverables:

| Deliverable | Status |
|-------------|--------|
| "Define action-conditioned resonance R(s,a)" | FORMULA GIVEN, NOT TESTED |
| "Specify what nabla_S(s,a) means and when it defines a coherent direction field" | THEORETICAL ONLY |
| "State conditions for stability under reparameterization" | CLAIMED via "covariance spectrum is invariant" (which is trivially true) |

And the success criterion:
> "a reproducible construction where argmax_a R(s,a) yields reliable navigation / optimization direction across multiple task families"

**This criterion is NOT MET.** The document itself admits this under "What's Still Missing":
- "Action-conditioned test: We measured J on static embeddings, not on action transitions"
- "Navigation benchmark: Need to test if following nabla-J + principal axes actually reaches goals"
- "Cross-model axis alignment: Do different trained models share the same principal axes?"

If the authors themselves list these as "still missing," how can the status be "CONFIRMED"?

### 1.3 J Coupling Is NOT a Direction Field

The document proposes J coupling as the direction signal:
```
J(x, anchors) = mean cosine similarity between x and its k nearest anchors
```

J measures local density, as the document itself discovers:
> "Untrained BERT has HIGHER J than trained... J measures density, not semantic organization."

A direction field requires a VECTOR-valued function (or at minimum, a gradient), not a scalar. J is scalar. The gradient of J gives a direction, but:
- nabla-J is not computed anywhere in Q31
- nabla-J on a discrete set of anchors requires a smoothing/interpolation scheme that is never specified
- The gradient of "mean cosine similarity to k nearest neighbors" is extremely noisy in practice because which k neighbors are "nearest" changes discontinuously as x moves

### 1.4 The Contextual Phase Selection Boost Is a Separate Finding

The 2026-01-21 addition about "contextual compass" (Section "CONTEXTUAL PHASE SELECTION BOOST") is interesting but irrelevant to confirming Q31. It shows that adding context text to prompts changes embedding similarity structure. This is well-known in the NLP community (prompt engineering). It does not validate compass mode navigation.

The claimed "4.13x J coupling variance boost" means the spread of J values increases when context is added. This is expected: adding domain-specific context concentrates relevant embeddings and separates irrelevant ones, increasing variance in any similarity metric. This is not evidence that J provides navigation direction.

### 1.5 The Formula Mutates Without Control

The compass formula evolves through the document:

Version 1: `Direction = argmax_a [J(s+a) x alignment_to_principal_axes(s+a)]`
Version 2: `Direction = argmax_a [J(s+a) x alignment_to_principal_axes(s+a)]` (same, just called "CORRECT" after Q43)
Version 3: `Direction = argmax_a [J_ctx(s+a, axis) x alignment_ctx(s+a, axis)]` (context added)

None of these are tested. The "implementation" provided (compass_navigate function, lines 226-237) computes argmax of cosine similarity to current word -- this is just **nearest-neighbor search**, not compass navigation. The entire J coupling and principal axis machinery is absent from the "implementation."

### 1.6 Connection to Inherited Issues

Q31's "CONFIRMED" status depends entirely on Q43. Phase 3 found Q43 to be OVERCLAIMED: valid linear algebra mislabeled as quantum geometry. Specifically:
- Berry curvature is identically zero (acknowledged by Q43 itself)
- "QGT" is just covariance matrix renamed
- Topological protection not established

Q31 inherits all these problems. The "geodesic flow on sphere = following principal axes" claim in Q31 is correct in the trivial sense that geodesics on a sphere are great circles, and PCA gives you the directions of maximum variance. But this does not mean following PCA directions provides useful NAVIGATION in any task-specific sense. PCA directions maximize variance; they do not minimize path length to a goal, which is what navigation requires.

**VERDICT: CONFIRMED is egregiously wrong. Q31 has no implementation, no navigation test, no action-conditioned test, no multi-task validation, and its theoretical "confirmation" comes from a tautological eigenvalue decomposition (Q43). The compass formula is never tested. The implementation code is just nearest-neighbor search. Status should be OPEN. This is the most overclaimed question in Batch 6D.**

---

# Q46: Geometric Stability (R=1350, OPEN)

**Target:** `THOUGHT/LAB/FORMULA/questions/lower_q46_1350/q46_geometric_stability.md`
**Report:** `THOUGHT/LAB/FORMULA/questions/lower_q46_1350/reports/` (contains file with special character in name)

```
Q46: Geometric Stability (R=1350)
- Claimed status: OPEN
- Proof type: N/A (no research conducted)
- Logical soundness: N/A
- Claims match evidence: N/A
- Dependencies satisfied: N/A
- Circular reasoning: N/A
- Post-hoc fitting: N/A
- Numerology: N/A
- Recommended status: OPEN (correct)
- Recommended R: 1350 (unchanged -- cannot evaluate what does not exist)
- Confidence: HIGH (that OPEN is correct)
- Issues: See analysis below
```

## 1. What Is the Actual Status?

**OPEN is correct.** The main document is 23 lines long:

```
Status: OPEN
...
No dedicated research yet. Related to Q8 (topology) and Q43 (QGT).
```

The README.md confirms:
```
Status: OPEN - no dedicated research yet
```

There is a report file in the reports directory, but its filename contains a non-ASCII character (appears to be a heart symbol prefix), which raises a question about how it was generated. I was unable to read this file due to the special character, but the main document and README both clearly state no research has been conducted.

## 2. Should OPEN Remain the Status?

Yes. The question -- "How stable are the geometric properties of semantic space under perturbation?" -- is well-posed and potentially valuable. However:

- No hypothesis has been formulated
- No experiments have been designed
- No tests have been run
- No results exist to evaluate

## 3. What Q46 SHOULD Address (When Work Begins)

Given the inherited issues from earlier phases, Q46 is actually one of the more important open questions because it directly relates to:

1. **The sigma sensitivity problem (P3, P5):** sigma^Df is exponentially sensitive to sigma. Q46 could quantify how much sigma changes under embedding perturbation (noise, different models, different corpora). This would be far more useful than Q29's div/0 epsilon fix.

2. **The Df stability problem:** Df = 22 for trained BERT, Df = 63 for untrained BERT, Df = 99 for random. How stable is Df under perturbation of the training data or model architecture?

3. **The R sensitivity problem:** When embeddings are perturbed (noise, adversarial, domain shift), how much does R change? Is the gate decision robust?

4. **The cross-model alignment problem (Q31, Q34):** Do geometric properties (Df, principal axes, curvature) transfer between models?

These are all urgent questions that would provide genuine engineering value. The current OPEN status is honest, and the question should be prioritized.

**VERDICT: OPEN is the correct and honest status. No criticism possible for work that has not been done. The question itself is well-scoped and potentially one of the most practically important in the framework.**

---

# Batch 6D Summary

| Question | Claimed Status | Recommended Status | Recommended R | Primary Issue |
|----------|---------------|-------------------|---------------|---------------|
| Q26 | RESOLVED | RESOLVED (with caveats) | 1100 (from 1240) | Overgeneralized "no scaling" claim; post-hoc semantic finding; no theoretical bound |
| Q29 | SOLVED | SOLVED (div/0 only) | 400 (from 1180) | Ignores the actual stability crisis (sigma^Df overflow); solves the trivial problem |
| Q30 | RESOLVED | RESOLVED (with caveats) | 1000 (from 1160) | Error bounds empirical not analytical; 250% R-value error buried; synthetic data only |
| Q31 | CONFIRMED | OPEN | 400 (from 1550) | No implementation, no navigation test, "confirmed" by tautology from Q43 |
| Q46 | OPEN | OPEN | 1350 (unchanged) | No work done; honest status; important question |

## Cross-Cutting Issues for Batch 6D

### Issue 1: The Real Stability Problem Is Unstudied

Q29 fixes div/0. Q30 speeds up computation. Neither addresses the fundamental numerical instability of the R formula: sigma^Df is exponentially sensitive to sigma, and Df varies by 4.5x between contexts (22 vs 43.5 vs 99). Until Q46 or a new question addresses this, the R formula is numerically unstable in any context where sigma^Df is computed.

### Issue 2: Engineering Solutions for a Formula That May Not Work

Q29 and Q30 provide engineering optimizations for computing R = E/sigma. But earlier phases found:
- E has 5+ incompatible definitions (P1)
- sigma is not universal (P5, varies 15x across domains)
- sigma^Df overflows (P3)
- R is not intensive under scale changes (P3, synthetic test shows 1000x variation)

Optimizing the computation of a formula whose components are unstable, inconsistent, and not universal is premature engineering. It is like optimizing the aerodynamics of a car with no engine.

### Issue 3: Q31 Status Inflation

Q31 has the highest R-score in this batch (1550) and claims CONFIRMED status, but has ZERO working implementations and ZERO empirical tests of its core claim (navigation). The "confirmation" chain passes through Q43, which Phase 3 already found to be tautological relabeling of standard linear algebra. This is the most severe overclaim in the batch.

---

## Issue Tracker Additions

| ID | Issue | Severity | Source | Affects |
|----|-------|----------|--------|---------|
| P6D-Q26-01 | "No scaling with D" tested on only 2 distinct dimensionalities from independent models | MEDIUM | Q26 methodology | Q26 scaling claim |
| P6D-Q26-02 | CV < 0.10 threshold is arbitrary; no theoretical sample complexity bound provided | MEDIUM | Q26 methodology | Q26 N_min values |
| P6D-Q26-03 | Semantic structure finding is post-hoc, single model, unreplicated | LOW | Q26 semantic structure test | Q26 secondary claim |
| P6D-Q29-01 | sigma^Df overflow (the actual stability crisis) is completely unaddressed | CRITICAL | Q29 scope | R formula usability |
| P6D-Q29-02 | Edge case R values span 6 orders of magnitude (0.20 to 1,000,000) -- marked PASS | HIGH | Q29 validation table | R interpretability |
| P6D-Q29-03 | Mock embedder used exclusively; no real-data stability testing | MEDIUM | Q29 test infrastructure | Q29 completeness |
| P6D-Q30-01 | No analytical error bounds; 100% accuracy may not hold near threshold or on real data | HIGH | Q30 methodology | Q30 generality claims |
| P6D-Q30-02 | 250% R-value error buried in limitations rather than highlighted | HIGH | Q30 documentation | User expectations |
| P6D-Q30-03 | Single gate threshold (0.8) tested; no sensitivity to threshold choice | MEDIUM | Q30 test design | Q30 generality |
| P6D-Q31-01 | CONFIRMED status with zero implementations and zero navigation tests | CRITICAL | Q31 status | Q31 credibility |
| P6D-Q31-02 | "Confirmation" via Q43 is tautological (SVD theorem, not empirical discovery) | CRITICAL | Q31-Q43 dependency | Q31 theoretical foundation |
| P6D-Q31-03 | Compass formula never tested; "implementation" is just nearest-neighbor search | CRITICAL | Q31 code vs specification | Q31 success criterion |
| P6D-Q31-04 | J coupling measures density not direction; no gradient computed | HIGH | Q31 theory | Q31 direction field claim |
| P6D-Q31-05 | Three "still missing" items listed by authors, yet status is CONFIRMED | HIGH | Q31 self-assessment | Q31 status consistency |

---

*Phase 6D adversarial review completed: 2026-02-05*
*No charitable interpretations. Evidence weighed as presented.*
