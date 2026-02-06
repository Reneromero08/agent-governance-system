# Verdict 1-Q2: Falsification Criteria (R=1750)

**Reviewer:** Claude Opus 4.6 (adversarial skeptic)
**Date:** 2026-02-05
**Primary file:** `THOUGHT/LAB/FORMULA/questions/critical_q02_1750/q02_falsification_criteria.md`
**Test files:** `q2_falsification_test.py`, `q2_echo_chamber_deep_test.py`
**References:** GLOSSARY.md, SPECIFICATION.md, SEMIOTIC_AXIOMS.md

---

## Verification Rubric

```
Q02: Falsification Criteria (R=1750)
- Claimed status: ANSWERED
- Proof type: empirical (synthetic simulation)
- Logical soundness: GAPS
- Claims match evidence: OVERCLAIMED
- Dependencies satisfied: MISSING [independence assumption not formalized; no real-data tests]
- Circular reasoning: DETECTED [see Issue 1 below]
- Post-hoc fitting: DETECTED [see Issue 6 below]
- Recommended status: PARTIAL
- Confidence: LOW
- Issues: see detailed analysis below
```

---

## Detailed Analysis

### 1. CIRCULAR REASONING: The Criteria Retroactively Excuse the Formula's Biggest Weakness

The Q02 answer discovers that echo chambers break R (correlated observations
give high R on wrong answers). The response to this devastating finding is to
declare: "R assumes independence. This is not a bug -- it is the
epistemological boundary."

This is a textbook escape hatch. The discovery is:

- R can be arbitrarily high while the underlying claim is arbitrarily wrong.

The response is:

- That does not count because the observations were correlated.

But the formula R = (E / grad_S) * sigma^Df contains NO term that measures
or penalizes correlation. There is no independence test built into R. So the
falsification criterion effectively says: "R is wrong only when its hidden
assumption (independence) is violated, but we have no way to know from R
alone whether the assumption holds." This makes the criterion unfalsifiable
in practice -- any failure can be attributed to "violated independence" after
the fact.

The "bootstrap defense" (add fresh data, see if R crashes) does NOT fix this.
It moves the goalpost from "R is a reliable signal" to "R is reliable only
if you also run this external validation procedure." If you need an external
procedure to check whether R is trustworthy, then R itself is not a
falsification-ready quantity.

**Verdict on circularity:** The falsification criterion for R's main failure
mode (echo chambers) is: "If R fails, independence was probably violated."
This is circular because it uses the failure to retroactively explain the
failure, without any independent test of the independence assumption.

### 2. COMPLETENESS: Major Failure Modes NOT Covered

The Q02 analysis tests exactly four attack vectors:

1. Echo chamber (correlated, biased)
2. Adversarial (identical wrong values)
3. Systematic bias (constant offset)
4. Bimodal trap (two clusters)

Missing attack vectors that would also break R:

- **Confounded variables:** Observations are independent but all rely on the
  same flawed premise. Example: 20 independent LLMs trained on the same
  biased corpus will give independent-looking but systematically wrong
  embeddings. R would be high and the independence assumption technically
  holds, but the answer is wrong. This is NOT covered by any Q02 criterion.

- **Dimensionality collapse:** The formula computes E from std(observations)
  in 1D. Real semantic embeddings are high-dimensional. Two embedding vectors
  can have low scalar std but high angular divergence. The 1D projection
  hides real disagreement. No criterion addresses this.

- **Scale dependence:** R depends on sigma and Df, which are fixed parameters
  in the tests (sigma=0.5, Df=1.0). What if the "correct" sigma varies by
  domain? SPECIFICATION.md Section 4.1 (Conjecture: sigma Universality) is
  OPEN. If sigma is wrong, R is systematically wrong. No falsification
  criterion addresses sigma being incorrect.

- **grad_S degeneracy:** When grad_S approaches 0, R diverges. The GLOSSARY
  acknowledges this as a "degenerate case that should be flagged." But the
  falsification criteria do not include "R divergence due to grad_S -> 0"
  as a failure mode. The test code adds 1e-10 to prevent division by zero
  but this creates artificially enormous R values that are meaningless.

- **Adversarial embedding attacks:** In the semantic domain (which is the
  primary claimed application), adversarial perturbations to embeddings can
  preserve cosine similarity while changing semantic content. No criterion
  addresses this.

### 3. SPECIFICITY: No Numerical Thresholds for Falsification

The Q02 document states three falsification criteria:

1. "Formula is CORRECT: It measures local agreement, which is what it claims"
2. "Formula FAILS when: Observations are correlated (independence violated)"
3. "Defense: Add fresh independent data; if R crashes, it was echo chamber"

None of these have numerical thresholds. What counts as "crashes"? The deep
test uses a 30% drop threshold, but this is chosen ad hoc, not derived, and
not stated in the main Q02 answer. A proper falsification criterion would
state: "If R drops by more than X% (where X is derived from the expected
variance of R under H0), the formula is falsified under condition Y."

The echo chamber deep test reports some numbers (R drops 93% vs 75%), but
these are results of a specific simulation, not general thresholds. They
depend on the specific noise levels, sample sizes, and bias magnitudes chosen
for the test.

### 4. INDEPENDENCE OF CRITERIA: All Criteria Reduce to One Test

The three stated criteria all reduce to a single test: "Is the independence
assumption satisfied?" There is no separate criterion for:

- E being miscalculated
- grad_S being a poor measure of variability
- sigma being wrong
- Df being incorrectly estimated
- The functional form R = (E/grad_S) * sigma^Df being wrong versus
  some alternative like R = E / (grad_S + sigma^Df)

A robust falsification framework would have INDEPENDENT criteria for each
component of the formula. Instead, Q02 addresses only one failure mode
(correlation) and declares the formula "correct" for everything else.

### 5. MODUS TOLLENS STRUCTURE: Absent

A proper falsification criterion has the structure:

  If the formula is correct, then P. NOT P observed. Therefore the formula
  is incorrect.

The Q02 criteria do NOT have this structure. Instead they have:

  If R is high and the answer is wrong, then independence was violated.

This is the CONVERSE of modus tollens. It protects the formula from
falsification rather than exposing it. The correct modus tollens would be:

  If the formula is correct, then for independent observations, high R
  implies high accuracy. For independent observations, high R was observed
  with low accuracy. Therefore the formula is incorrect.

But the Q02 answer never specifies what "high R" or "high accuracy" mean
numerically, so even this structure cannot be applied.

### 6. POST-HOC FITTING: The "Defense" Was Designed After the Attack

The echo chamber vulnerability was discovered by running the attack tests.
The "bootstrap defense" was then designed to address the specific failure
that was found. This is post-hoc fitting of the falsification criteria to
the observed failure mode. Proper falsification criteria should be stated
BEFORE testing, not constructed to explain away failures after they are
found.

The test code makes this explicit: the `test_practical_defense()` function
was written as part of the Q02 investigation, not as a pre-registered
protocol. The 30% threshold was tuned to achieve good F1 score on the
specific simulation parameters used.

### 7. THE compute_R IMPLEMENTATION IS NOT THE REAL FORMULA

The test code defines:

```python
E = 1.0 / (1.0 + np.std(observations))
grad_S = np.std(observations) + 1e-10
```

This means E = 1/(1 + grad_S), so:

```
R = (1/(1+s)) / s * sigma^Df = sigma^Df / (s * (1+s))
```

where s = std(observations). This is a monotonically decreasing function of
s. R is entirely determined by the standard deviation of the observations
and the fixed parameters sigma, Df.

Compare to GLOSSARY.md Definition 2, which says E is "mean pairwise cosine
similarity of embedding cluster" in the semantic domain. The test code does
NOT implement this. It uses a toy definition of E that makes R trivially
reducible to 1/variance.

Therefore: the falsification tests are testing a DIFFERENT formula than the
one described in the specification. Any conclusions about falsifiability of
the actual Living Formula R = (E/grad_S) * sigma^Df (with proper E and
grad_S definitions) are not supported by these tests.

### 8. SYNTHETIC-ONLY EVIDENCE

All tests use numpy random number generation. No real embedding data, no
real semantic tasks, no real experimental observations. The HONEST_FINAL_STATUS
document (from Q54 review) explicitly calls out this problem:

> "Stop using simulations that implement the formula."

The Q02 falsification criteria have never been tested against real data
where ground truth is known independently.

---

## Summary of Issues

| Issue | Severity | Type |
|-------|----------|------|
| 1. Circular escape hatch (independence excuse) | CRITICAL | Circular reasoning |
| 2. Missing attack vectors (confounders, dimensionality, scale) | HIGH | Incompleteness |
| 3. No numerical thresholds | HIGH | Vagueness |
| 4. All criteria reduce to one test | MEDIUM | Redundancy |
| 5. No modus tollens structure | HIGH | Logical weakness |
| 6. Bootstrap defense is post-hoc | MEDIUM | Post-hoc fitting |
| 7. Test code uses wrong formula | CRITICAL | Implementation error |
| 8. Synthetic-only evidence | HIGH | Insufficient evidence |

---

## Recommended Actions

1. **Rewrite falsification criteria with modus tollens structure.** Each
   criterion must have the form: "If R is correct, then [specific measurable
   prediction]. If [specific measurable counter-observation], then R is
   falsified."

2. **Add numerical thresholds.** Define what R value constitutes "high,"
   what accuracy constitutes "correct," and what drop percentage constitutes
   "crash." These must be stated before running tests.

3. **Add independent criteria for each formula component.** Separate criteria
   for E, grad_S, sigma, Df, and the functional form.

4. **Fix the test implementation.** Use the actual definitions of E and
   grad_S from GLOSSARY.md, not toy proxies.

5. **Test on real data.** Use published embedding benchmarks with known
   ground truth (e.g., STS-B, MTEB) to test whether high R correlates with
   correct semantic similarity judgments.

6. **Address the confounded-variables attack.** This is the most dangerous
   gap: independent but systematically biased observations pass all current
   criteria while still being wrong.

7. **Pre-register the falsification protocol.** State all criteria, thresholds,
   and test procedures before running any validation experiment.

---

## Final Verdict

**Claimed status: ANSWERED. Recommended status: PARTIAL.**

The Q02 document identifies one genuine vulnerability (echo chambers) and
proposes a post-hoc defense. However, the falsification criteria themselves
are unfalsifiable in practice due to the independence escape hatch, lack of
numerical specificity, absence of modus tollens structure, and reliance on
a toy implementation that does not match the formal specification. Major
failure modes are uncovered. All evidence is synthetic. The question "Under
what conditions would we say the formula is wrong?" remains inadequately
answered -- the current criteria allow the formula to escape falsification
in essentially all realistic scenarios by invoking the independence caveat.

**Confidence: LOW.** The current falsification criteria do not provide a
credible basis for claiming the formula is scientifically falsifiable.
