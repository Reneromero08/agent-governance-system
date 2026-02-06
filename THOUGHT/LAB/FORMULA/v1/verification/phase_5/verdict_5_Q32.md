# Phase 5 Verdict: 5-Q32 -- Meaning as Physical Field (R=1670)

**Date:** 2026-02-05
**Reviewer:** Adversarial skeptic (Phase 5)
**Target:** `THOUGHT/LAB/FORMULA/questions/critical_q32_1670/q32_meaning_as_field.md`
**Reports reviewed:**
- `Q32_SOLVED_CRITERIA_AND_TEST_PLAN.md`
- `Q32_PHASE3_EVIDENCE_PACKAGE.md`
- `Q32_NEIGHBOR_FALSIFIER_DATA_TRAIL.md`
**Test code reviewed:** `q32_meaning_field_tests.py`, `q32_public_benchmarks.py`
**References:** GLOSSARY.md

---

## Summary Verdict

```
Q32: Meaning as Physical Field (R=1670)
- Claimed status: PARTIAL (semiosphere field claim; Phases 1-7 receipted)
- Proof type: empirical (synthetic + public NLI benchmarks) + analogy (free energy, EM field)
- Logical soundness: MODERATE GAPS
- Claims match evidence: PARTIALLY OVERCLAIMED (2 of 3 tracks)
- Dependencies satisfied: PARTIAL [E definition ambiguity inherited; R stability concerns inherited]
- Circular reasoning: DETECTED [see Sections 2, 4]
- Post-hoc fitting: DETECTED [see Sections 3, 5]
- Numerology: NOT DETECTED (Q32 is more empirical than other questions)
- Recommended status: PARTIAL (retained, but with scope reduction)
- Recommended R: 900-1100 (down from 1670)
- Confidence: HIGH
- Issues: See detailed analysis below
```

---

## Evaluation Question 1: Is "Physical Field" a Metaphor or a Testable Claim?

### What Q32 Claims

Q32 claims that "meaning" can be operationalized as a "real, measurable field" M := log(R) with dynamics, predictions, and falsifiers, analogous to electromagnetic fields in physics.

### The Verdict: IT IS A METAPHOR BEING DRESSED UP AS A TESTABLE CLAIM, BUT THE TESTS THEMSELVES ARE LEGITIMATE (FOR A WEAKER CLAIM)

#### 1.1 What "Field" Means in Physics vs. What Q32 Actually Delivers

A physical field has:
1. A well-defined value at every point in a continuous space (e.g., E(x,y,z,t) everywhere in spacetime)
2. Field equations: a PDE governing evolution (Maxwell's equations, wave equation, Klein-Gordon)
3. A propagator: how disturbances at one point influence another
4. Conservation laws: continuity equations, Noether currents
5. Coupling constants: quantified interaction with sources
6. Gauge structure / transformation laws

Q32 delivers NONE of these. Specifically:

- **No field equation.** There is no PDE for M(x,t). The "dynamics" described are sequential accumulation of evidence scores (adding more NLI cross-encoder scores and watching R change). This is a time series, not a field evolution.
- **No propagator.** The "propagation/gluing test" (Test 3) shows that merging two compatible observation sets preserves high M. This is an algebraic property of the R formula (averaging consistent scores stays high), not a propagator in any field-theoretic sense.
- **No conservation law.** There is no continuity equation dM/dt + div(J) = 0. The word "conservation" does not appear in the tests.
- **No coupling constant.** The "sources" and "sinks" described in q32_meaning_as_field.md (Section "Sources, sinks, coupling") are verbal descriptions: "independent evidence that increases E" is a source, "contradiction" is a sink. These are not quantified coupling terms.
- **No gauge structure.** The document mentions "Semiotic Mechanics Axiom 6" for coupling but provides no mathematical formulation.

What Q32 actually delivers is: **a scalar scoring function M(observations, checks) = log(E(obs, checks) / SE(obs))** that is computed from NLI cross-encoder outputs. This is a statistical quality metric for claim-evidence alignment, not a field.

#### 1.2 The Free Energy Bridge Is a Scope-Limited Mathematical Identity

The document claims: "M = log(R) = -F + const => R proportional to exp(-F)"

This is presented as connecting M to the Free Energy Principle (Friston, 2010). But the derivation requires assuming a specific Gaussian likelihood family and is noted as "family-scoped." Under this specific assumption, log(R) and negative free energy have a linear relationship -- which is true of *any* log-likelihood ratio under Gaussian assumptions. This is not evidence that M is a field; it is evidence that M is a log-likelihood ratio, which is what it was constructed to be.

#### 1.3 The "Semiosphere" Is Doing Rhetorical Work

The claim is carefully scoped as "field on the semiosphere" (a space of semantic interpretations), not a field in physical spacetime. This is an important distinction that Q32 handles correctly in the solved criteria document. But calling it a "field on the semiosphere" raises the question: what is the semiosphere's topology, metric, and dimensionality? None of these are defined. "Semiosphere" is used as a placeholder for "the space where we compute things," which makes "field on the semiosphere" equivalent to "function we compute on data."

**Assessment: The word "field" is a metaphor. What Q32 actually demonstrates is a statistical scoring function with some desirable properties (discriminates truth from echo chambers, transfers across domains). This is useful but does not warrant the label "field."**

---

## Evaluation Question 2: Phase 1-3 Evidence Audit -- What Actually Passed and What Failed?

### 2.1 Synthetic Tests (q32_meaning_field_tests.py) -- PASS, BUT TAUTOLOGICAL

The three synthetic tests all pass:

**Test 1 (Echo-Chamber Falsifier):** Generates independent vs. echo-chamber observations, computes R_grounded vs R_ungrounded, verifies that R_grounded discriminates while R_ungrounded does not.

**Problem:** This test is a tautology of the formula design. R_grounded uses `E_from_empirical_check(mu_hat, check)` which computes exp(-z^2/2) where z = |mu_hat - mean(check)| / SE(check). Echo-chamber observations have mu_hat biased away from truth; independent check data centers near truth; therefore z is large for echo chambers and E is small. This is not a discovery about meaning fields; it is verification that comparing a biased estimate against an independent sample produces a low similarity score. Any statistician would predict this result without invoking field theory.

R_ungrounded uses only internal dispersion (E_internal = 1/(1+std)), which naturally gives high scores to tight clusters regardless of truth. The test demonstrates that *not having a ground truth check* makes you vulnerable to echo chambers. This is epistemology 101, not a field property.

**Test 2 (Phase Transition Gate):** Streams evidence, watches M(t) cross a threshold, verifies that independent observations crystallize while echo-chamber observations do not.

**Problem:** M uses SE (std/sqrt(n)) as the denominator. As n increases, SE shrinks, so M = E/SE grows. For independent observations near truth, E stays near 1 and SE shrinks, so M grows and crosses the threshold. For biased echo chambers, E stays low (large z) while SE also shrinks but E dominates the outcome. The "phase transition" is just the threshold crossing of a monotonically growing function (for the truth-aligned case). A true phase transition would require nonlinear dynamics -- a qualitative change in behavior at a critical point. What is demonstrated is a score crossing a fixed threshold, which every accumulating statistic does.

**Test 3 (Propagation/Gluing):** Merges two observation sets; if both are above threshold, merged set should also be above.

**Problem:** Merging two sets of independent observations centered near truth produces a larger set still centered near truth with smaller SE, so M stays above threshold. Merging truth-aligned with echo-chamber data dilutes the echo-chamber's influence. This is the behavior of averages. It does not require or demonstrate a "field" propagation law.

### 2.2 Public Benchmark Tests (q32_public_benchmarks.py) -- LEGITIMATELY NONTRIVIAL

The public benchmark harness is significantly more credible than the synthetic tests because it operates on real data (SciFact, Climate-FEVER, SNLI, MNLI) with actual NLI models.

**What passes:**
- Intervention tests (correct checks vs. wrong checks): M_correct > M_wrong across datasets. This is the strongest result.
- Transfer without retuning: Thresholds calibrated on one dataset transfer to others across 4 domains, 12 ordered pairs.
- Negative controls fail: Agreement inflation, paraphrase, and shuffle controls reliably produce FAIL outcomes.
- Ablation shows E matters: Removing the empirical grounding (no_grounding ablation, R=1 constant) kills the signal.
- Stress tests pass at scale: Multi-seed, multi-trial stress runs maintain above-threshold pass rates.

**What fails or is concerning:**
- **SciFact streaming was fragile and had to be stabilized.** The data trail documents (Section "Multi-seed matrix failure") that SciFact streaming was "highly sensitive to which abstract sentences are sampled" and that different seeds flipped results. The fix was to hardcode the stream seed to 123 for SciFact streaming specifically. This means the "multi-seed" stress test actually varies seeds everywhere *except* where it matters most. This is a significant methodological concern.
- **The no_scale ablation did not kill the effect in fast mode.** The data trail records (Phase 2 ablation section): "no_scale did not hard-kill the effect in fast mode." This means the scale term grad_S (which is supposed to be a necessary part of the field definition) can be removed without destroying the signal. If the "field" works without one of its defining terms, the field equation is overcomplete -- some terms are decorative.
- **Phase 3 stress FAILS before being fixed.** The raw datatrail shows p3_stress_scifact_neighbor_full "FAILs the pass-rate gate" at first, followed by a v2 run with "fixed neighbor selection" that passes, followed by a v3 run that also passes. The iterative fixing of the test until it passes is a concerning pattern.

### 2.3 Phase 4-7 -- MIXED

**Phase 4 (Geometry):** The geometry tipping test passes, but the geometry signal is a participation ratio proxy, not a QGT metric. The QGTL integration is marked as done, but the actual geometry artifacts use participation ratio, which is just the effective rank of the embedding matrix. Effective rank changing when you inject contradictory evidence is expected behavior -- adding noise reduces the coherent structure of the embedding set. This does not require field theory.

**Phase 5 (Scale):** 4-domain transfer matrix, stress runs, sweeps, and negative controls across all domains. This is the strongest evidence in Q32. The cross-domain transfer without retuning is a legitimate empirical result.

**Phase 6 (Physical Force Harness):** Synthetic validator suite passes. This only shows the harness itself works correctly on synthetic data -- it does not demonstrate any physical coupling.

**Phase 7 (Real EEG Data):** Explicitly FAILS. The coupling test on OpenNeuro ds005383 data produces r=0.21 (below null threshold 0.36), p=0.11 (not significant), and directionality_ok=false. The document correctly interprets this as "the harness correctly rejects weak/spurious correlations," which is honest. But the fact remains: the only test against real physical data produces a null result.

---

## Evaluation Question 3: Do Negative Controls Fail Correctly?

### The Verdict: YES, WITH CAVEATS

The negative control design is one of the strongest aspects of Q32. Three types of negative controls are implemented:

1. **Agreement inflation:** Wrong checks selected from truth-consistent (SUPPORT) nearest neighbors of the claim. Designed to artificially inflate M_wrong. Result: gates correctly FAIL.

2. **Paraphrase/perfect overlap:** Wrong check = correct check (same evidence used for both). Should make M_correct = M_wrong. Result: gates correctly FAIL.

3. **Shuffle/echo:** Wrong check = observation evidence repeated (echo-chamber self-check). Result: gates correctly FAIL.

These are receipted across all 4 domains in both bench and streaming modes (Phase 5.3 datatrail).

**Caveats:**

- The negative controls test the formula, not the field claim. They show that R_grounded(obs, wrong_check) < R_grounded(obs, correct_check) when wrong_check is constructed to be truth-inconsistent. This is expected from the formula design: E = exp(-z^2/2) penalizes misaligned evidence. The controls confirm the formula works as designed, not that meaning is a field.

- The "null result" case (no meaning present) is not tested in the way a physicist would test it. A physics field experiment would include a "shielded" control where the field source is absent (e.g., vacuum baseline). The Q32 analog would be: random text with no semantic relationship to the claim. This is partially addressed by the shuffle control (which breaks semantic alignment) but not by a pure noise baseline.

---

## Evaluation Question 4: Is "Meaning Field" Distinguishable from "Embedding Space Has Geometric Structure"?

### The Verdict: BARELY. THE DISTINCTION IS THIN.

#### 4.1 What Q32 Measures vs. What "Embedding Geometry" Already Gives You

The core measurement pipeline in q32_public_benchmarks.py is:

1. Embed claim + evidence sentences using sentence-transformers (MiniLM-L6-v2)
2. Score (claim, evidence) pairs using a cross-encoder NLI model (nli-MiniLM2-L6-H768)
3. Compute R = E / grad_S where E = exp(-z^2/2) and z uses the standard error of these scores
4. Take M = log(R)

This is fundamentally: **use an NLI model to score claim-evidence pairs, then compute a statistical summary of those scores that penalizes high variance and rewards alignment with held-out evidence.**

The "embedding space has geometric structure" baseline would be: **use sentence embeddings to measure similarity, with some normalization for variance.** This is essentially what the `--scoring cosine` mode does (cosine similarity instead of cross-encoder NLI).

The key question is: does the "field" formulation (R_grounded with E/grad_S) provide something beyond what a simple cosine-similarity-based NLI score provides?

#### 4.2 The Ablation Evidence Is Ambiguous

The ablation results show:
- **no_grounding (R=1 constant):** Hard kills the effect. But this just means "using the formula at all matters vs. not using it."
- **no_essence (E=1):** Should kill the effect if empirical grounding matters. The result is not prominently reported in the datatrail for the public benchmarks (the ablation section only shows no_grounding and no_scale).
- **no_scale (grad_S=1):** Does NOT kill the effect in fast mode. This is significant: if removing the uncertainty normalization still produces passing gates, then the "field" behavior is carried primarily by the NLI cross-encoder score, not by the formula structure.

This suggests that the heavy lifting is done by the NLI model (which was trained on millions of examples of entailment/contradiction), and the R formula is a thin statistical wrapper. The "field" is the NLI model's learned representations, which is precisely "embedding space has geometric structure."

#### 4.3 The Transfer Result Is the Best Counterargument

The strongest evidence that Q32's formulation adds something beyond raw embedding geometry is the threshold transfer result: calibrate thresholds on one dataset, freeze them, apply to another dataset without retuning, and the gates still pass across 4 domains.

However, this could equally be explained by: "NLI models trained on broad data have consistent score distributions across NLI-style tasks." The transfer is across SciFact, Climate-FEVER, SNLI, and MNLI -- all of which are textual entailment/inference tasks. The "transfer without retuning" is arguably a property of the NLI model's robustness, not of the "meaning field" formulation.

A stronger test would be transfer to a fundamentally different domain (e.g., code correctness, mathematical proof, image captioning) where the NLI model was not trained.

---

## Evaluation Question 5: Is PARTIAL Status Appropriate?

### The Verdict: YES, PARTIAL IS APPROPRIATE -- BUT ONLY FOR THE WEAKENED CLAIM

#### 5.1 What the Status Should Apply To

Q32's document distinguishes two tracks:
1. **Semiosphere field claim:** M := log(R) is a "measurable field on the semiosphere" with operational tests.
2. **Physical spacetime field claim:** Meaning is a fundamental physical force with sensor-measurable observables.

The physical spacetime claim (Track 2) is clearly not supported. Phase 7 EEG data produced a null result. Phase 8 (lab-grade falsification) has not been attempted. The document correctly does not claim this is answered.

The semiosphere field claim (Track 1) has substantial empirical support (Phase 5 pass), but the "field" label is overclaimed. What is demonstrated is:

- A statistical scoring function that discriminates truth-aligned from truth-inconsistent evidence
- This function transfers across 4 NLI-style domains without retuning
- Negative controls correctly fail
- Ablations show the empirical grounding term matters

This is a legitimate empirical contribution. But calling it a "field" adds connotations (field equations, propagators, gauge structure) that are not supported.

#### 5.2 Incomplete Gates

The roadmap itself identifies incomplete items:
- Phase 4.2 (Independence stress): "Define a public 'independence' proxy per dataset" -- NOT DONE
- Phase 4.3 (Causal intervention falsifiers): "Expand the existing wrong-check intervention into a causal suite" -- NOT DONE
- The echo-chamber collapse prediction (the most distinctive claim of Q32) has been tested only synthetically, not on real public data with genuine correlated sources

The core falsification claim -- that meaning fields collapse in echo chambers under independence stress -- is verified only in the synthetic generator (q32_meaning_field_tests.py Test 1), where the test is tautological. The public benchmark tests use "neighbor wrong checks" (semantically similar but truth-inconsistent evidence from CONTRADICT-labeled examples), which is a good adversarial test but not the same as the echo-chamber prediction.

#### 5.3 Status Recommendation

PARTIAL is appropriate, but the claim scope should be narrowed from "meaning is a physical field" to "R_grounded is a statistical scoring function with cross-domain transfer properties that discriminates truth-aligned evidence from adversarial alternatives."

---

## Section 6: Inherited Issues from Phases 1-4

| Phase | Issue | Impact on Q32 |
|-------|-------|---------------|
| P1 | 5+ incompatible E definitions | Q32's E = exp(-z^2/2) is a Gaussian kernel, not the E in GLOSSARY (pairwise cosine similarity); definitional mismatch |
| P1 | All evidence synthetic | Q32 partially mitigates this with public benchmarks, but synthetic tests are still tautological |
| P2 | Quantum interpretation falsified | Q32 does not depend on quantum claims directly, but the QGTL geometry backend (Phase 4) inherits this |
| P3 | R numerically unstable | Q32's R_grounded uses SE in the denominator; when SE -> 0 (highly consistent observations), R diverges; EPS guard is present but the instability is structural |
| P1-4 | "Field" language without field equations | Q32 is the primary perpetrator of this pattern |

---

## Section 7: What Q32 Gets Right

In fairness, Q32 is one of the better-executed questions in this framework:

1. **Honest about scope.** The document carefully separates the semiosphere claim from the physical spacetime claim and does not pretend Phase 7's failure is a success.

2. **Genuine negative controls.** Three types of negative controls (inflation, paraphrase, shuffle) are implemented, receipted, and demonstrably fail. This is better methodology than most questions in the framework.

3. **Public data.** Unlike many other questions (which test only on synthetic/curated data), Q32 uses real public datasets (SciFact, Climate-FEVER, SNLI, MNLI).

4. **Receipted and reproducible.** SHA256 hashes, pinned environments, exact rerun commands, and replication bundles are provided. The datatrail is extensive and auditable.

5. **Falsification-oriented.** The test plan is structured around "what would make us reject the claim," which is the correct scientific orientation.

6. **The no_grounding ablation is a legitimate falsifier.** Showing that R=1 (constant) fails all gates while the grounded R passes is a meaningful control.

7. **Phase 7 EEG result is honestly reported as FAIL.** Many research programs would bury or spin a null result. Q32 reports it prominently and correctly interprets it.

---

## Section 8: Critical Flaws

### 8.1 The "Field" Label Is Unjustified

Nothing in Q32's evidence supports calling M a "field" in any technical sense. There are:
- No field equations (no PDE for M)
- No propagator (the "gluing" test is set merging, not propagation)
- No conservation law
- No coupling constant with units
- No gauge structure
- No wave-like behavior (superposition, interference, diffraction)

Calling M a "field" because it is "a scalar function defined on a space" is like calling temperature a "field" because T(x,y,z) is a scalar function on space. Temperature IS a field in the mathematical sense, but nobody claims "temperature as a physical field" is a novel discovery. The novelty claim of Q32 rests on the "field" language, which is not earned.

### 8.2 The Synthetic Tests Are Tautological

All three tests in q32_meaning_field_tests.py verify that the R formula behaves as designed:
- Test 1: Grounding against independent data catches bias (by design)
- Test 2: Accumulating evidence shrinks SE, increasing M past threshold (by construction)
- Test 3: Merging consistent data preserves quality (by arithmetic)

None of these require or demonstrate field properties. They demonstrate that the scoring formula has sensible statistical behavior.

### 8.3 SciFact Streaming Stabilization Is Concerning

The datatrail reveals that SciFact streaming was unstable across seeds, and the fix was to hardcode seed=123 for the streaming sample selection. This means:
- The "multi-seed" stress tests are not truly multi-seed for the most sensitive component
- The streaming results are deterministic by construction, not robust by nature
- The system's behavior under genuine stochastic variation is unknown for the most important test mode

### 8.4 The NLI Model Does the Heavy Lifting

The cross-encoder NLI model (nli-MiniLM2-L6-H768) is trained on millions of NLI examples. It already knows how to distinguish entailment from contradiction. The R formula adds a thin statistical layer (Gaussian kernel + SE normalization) on top of the model's predictions. The "meaning field" is largely the NLI model's learned decision boundary, repackaged with different notation.

Evidence for this interpretation: the no_scale ablation (removing SE normalization) does not kill the effect in fast mode, suggesting the NLI scores alone carry most of the signal.

### 8.5 E Definition Mismatch

The GLOSSARY defines E as: "Mean pairwise cosine similarity of embedding cluster" (semantic domain). But Q32's R_grounded uses E = exp(-z^2/2) where z = |mu_hat - mu_check| / SE(check). These are different functions applied to different inputs. Q32's E is a Gaussian kernel on standardized residuals; the GLOSSARY's E is a cosine similarity aggregate. The formula R = (E / grad_S) * sigma^Df is the "Living Formula," but Q32's implementation omits the sigma^Df term entirely (depth_power defaults to 0.0, making the depth term equal to 1). Q32 is testing a simplified version of the formula while claiming to validate the full framework.

---

## Section 9: The Echo-Chamber Prediction -- Strongest and Weakest Point

The echo-chamber collapse is Q32's signature prediction: high consensus that is correlated (not independent) should NOT sustain high M. This is the one prediction that could genuinely distinguish a "meaning field" from an "agreement field."

**Strength:** The synthetic test (Test 1) demonstrates this cleanly. The negative controls on public data (inflation, where wrong checks are truth-consistent but from different claims) partially test this.

**Weakness:** The real echo-chamber prediction has NOT been tested on actual correlated data. The roadmap items 4.2 (independence stress on public data) and 4.3 (causal intervention falsifiers) are both marked as NOT DONE. The core distinguishing prediction of the "meaning field" claim is unvalidated on real data.

Until genuine independence stress is tested on real correlated sources (not synthetically generated or simulated via NLI model interventions), the echo-chamber prediction remains a theoretical claim supported only by a tautological synthetic test.

---

## Final Assessment

Q32 presents a legitimate empirical contribution: a statistical scoring function (R_grounded) that discriminates truth-aligned from truth-inconsistent evidence, transfers across multiple NLI domains without retuning, and correctly fails under negative controls. This is useful and well-documented.

However, Q32 overclaims in three ways:

1. **"Field" label:** The scoring function M := log(R) has none of the properties that define a physical field (field equations, propagator, conservation laws, coupling constants, gauge structure). Calling it a "field" borrows the prestige of physics for what is a statistical quality metric.

2. **"Physical field" track:** The only real-data physical test (Phase 7, EEG) produced a null result. The claim that meaning is a "physical field" is not supported.

3. **Echo-chamber prediction:** The most distinctive prediction (correlated consensus collapses under independence stress) is tested only synthetically, where the result is tautological by formula construction. The public benchmark tests do not directly test this prediction with genuine correlated sources.

**Recommended status: PARTIAL** -- retained, because the public benchmark evidence is substantial and the methodology is sound for the weaker claim. But the "field" language should be retracted in favor of "cross-domain statistical scoring function with transfer properties," and R should be reduced to reflect that the most novel predictions (echo-chamber collapse on real data, physical field coupling) are either untested or falsified.

**Recommended R: 900-1100** (down from 1670). The reduction reflects:
- The "field" label is unsupported (-200)
- The physical field track produced a null result (-200)
- The echo-chamber prediction is untested on real data (-170)
- Partial credit for: extensive public benchmark evidence, honest reporting, strong negative controls, cross-domain transfer

---

## Appendix: Issue Tracker Additions

| ID | Issue | Severity | Source | Affects |
|----|-------|----------|--------|---------|
| P5-Q32-01 | "Field" label applied without field equations, propagator, or conservation laws | CRITICAL | q32_meaning_as_field.md | Central framing |
| P5-Q32-02 | Synthetic tests (q32_meaning_field_tests.py) are tautological: verify formula behavior, not field properties | HIGH | q32_meaning_field_tests.py | Phase 1 evidence |
| P5-Q32-03 | SciFact streaming stabilized by hardcoding seed=123, undermining multi-seed robustness claim | HIGH | Q32_NEIGHBOR_FALSIFIER_DATA_TRAIL.md | Phase 3 stress tests |
| P5-Q32-04 | no_scale ablation does not kill effect in fast mode; formula terms may be decorative | HIGH | P2 datatrail | Formula validity |
| P5-Q32-05 | E definition in Q32 (Gaussian kernel on residuals) differs from GLOSSARY E (cosine similarity) | HIGH | q32_public_benchmarks.py vs GLOSSARY.md | Framework consistency |
| P5-Q32-06 | sigma^Df term omitted (depth_power=0); Q32 tests simplified formula, not full Living Formula | MEDIUM | q32_public_benchmarks.py line 49 | Formula coverage |
| P5-Q32-07 | NLI cross-encoder model does heavy lifting; "field" may be a thin wrapper on existing ML | MEDIUM | q32_public_benchmarks.py | Novelty claim |
| P5-Q32-08 | Echo-chamber collapse prediction (Q32's signature claim) untested on real correlated data | CRITICAL | Roadmap 4.2, 4.3 both NOT DONE | Core prediction |
| P5-Q32-09 | Phase 7 EEG coupling test: FAIL (r=0.21, p=0.11, directionality wrong) | HIGH | Q32_NEIGHBOR_FALSIFIER_DATA_TRAIL.md Phase 7 | Physical field track |
| P5-Q32-10 | Phase 3 stress test initially FAILS, then passes after fixing neighbor selection method | MEDIUM | Datatrail p3_stress v1 vs v2 vs v3 | Robustness |
| P5-Q32-11 | All 4 benchmark domains are NLI-style tasks; no transfer to fundamentally different domains | MEDIUM | q32_public_benchmarks.py | Transfer generality |
| P5-Q32-12 | "Free energy bridge" is a scope-limited identity under Gaussian assumptions, not a general result | MEDIUM | q32_meaning_as_field.md | Theoretical claims |
| P5-Q32-13 | "Semiosphere" topology, metric, and dimensionality undefined; "field on semiosphere" is vacuous without this | MEDIUM | q32_meaning_as_field.md | Field claim |
| P5-Q32-14 | Phase transition test shows threshold crossing of growing score, not genuine nonlinear dynamics | MEDIUM | q32_meaning_field_tests.py Test 2 | Nonlinear time claim |
