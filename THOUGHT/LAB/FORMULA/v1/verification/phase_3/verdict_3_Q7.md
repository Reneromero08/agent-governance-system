# Verdict: 3-Q7 Multi-scale Composition (R=1620)

```
Q07: Multi-scale Composition (R=1620)
- Claimed status: CONFIRMED
- Proof type: Empirical (axiomatic framework with numerical tests)
- Logical soundness: GAPS
- Claims match evidence: OVERCLAIMED
- Dependencies satisfied: MISSING [Q3 uniqueness proof (circular per P1-03), stable E definition (P1-01)]
- Circular reasoning: DETECTED [axioms C1-C4 designed to select R; uniqueness "proof" assumes conclusion]
- Post-hoc fitting: DETECTED [thresholds tuned after results observed; formula mutated between files]
- Recommended status: PARTIAL (some empirical observations valid; theoretical claims unsupported)
- Confidence: MEDIUM
- Issues: See detailed analysis below
```

---

## 1. Does the formula actually work across scales or just at one convenient scale?

**Verdict: It works at one convenient granularity of a single corpus, not genuinely "across scales."**

The entire test apparatus uses exactly ONE corpus (`MULTI_SCALE_CORPUS` hardcoded in `real_embeddings.py`): a hand-crafted set of 64 words, 20 sentences, 5 paragraphs, and 2 documents, all on the same theme (royalty, nature, emotions, family, technology). This is not multi-scale testing in any meaningful sense. The "scales" are:

- **64 single words** like "king", "queen", "happy"
- **20 sentences** constructed from those same words
- **5 paragraphs** constructed from those same sentences
- **2 documents** constructed from those same paragraphs

These are not independent observations at different scales. They are a single nested corpus. The paragraphs literally contain the sentences, which contain the words. R is computed on each level's embeddings independently (distance from centroid), so all that is being measured is "how tightly clustered are the SentenceTransformer embeddings of items at this level." There is no reason to expect wildly different answers because the items at each level share thematic content by construction.

Critical observation from the receipt data: R values are [0.708, 0.642, 0.727, 0.959]. The document-level R (0.959) is **47% higher** than the sentence-level R (0.642). For a quantity claimed to be "intensive" and scale-independent, this is a substantial deviation. The CV of 0.158 is declared "within threshold" only because the threshold was set at 0.3 -- which is extremely generous for an "intensive" property. Temperature in a well-stirred pot varies by less than 1%, not 16%.

No test uses genuinely different domains, corpora, or granularity levels. No test computes R on, say, a medical text at word level and a legal text at paragraph level. The claim of multi-scale generality rests on a single themed toy corpus.

---

## 2. Are composition rules derived or imposed?

**Verdict: IMPOSED. The axioms C1-C4 are reverse-engineered to select R = E/sigma.**

The composition axioms document (`composition_axioms.md`) presents C1-C4 and then claims R = E(z)/sigma is the "unique" form satisfying all four. But the "proof sketch" is logically vacuous:

1. **C1 (Locality)** -- "forces R to be a function of local statistics." This is trivially true of any function that takes a data subset as input. It does not constrain the functional form at all.

2. **C2 (Associativity)** -- "forces R to be scale-covariant: R(T_lambda(.)) = g(lambda) * R(.)." This step is unjustified. Associativity of the transformation operator does not imply that R must transform as a simple multiplicative function of the scale factor. This is a non-sequitur.

3. **C3 (Functoriality)** -- "forces the structure to be preserved: R must be a nice function." "Nice" is not a mathematical constraint. Monotonicity and continuity do not select a unique function.

4. **C4 (Intensivity)** -- "forces g(lambda) = 1." This is just restating the conclusion (R is intensive) as an axiom. The axiom says "R does not change with scale" and then the "theorem" concludes "therefore R does not change with scale."

The "uniqueness" claim in the axioms file (lines 103-133) is not a proof. The four steps listed do not constitute a deductive chain. There is no theorem statement with hypotheses and conclusion. There is no consideration of what mathematical space of functions is being searched. The argument is: "R has these four properties" -> "R is the unique function with these four properties." This is tautological unless you demonstrate that the axioms are independently necessary AND sufficient, which is not done.

**The alternatives "failing" does not prove uniqueness.** Testing 5 specific alternatives and showing they fail some axiom is an enumeration over 5 out of infinitely many possible functions. You would need to prove that no other function in the relevant function space can satisfy C1-C4.

---

## 3. Is "multi-scale" genuinely tested with different granularities?

**Verdict: NO. It is the same data viewed at nested levels.**

The "deep" domain, "imbalanced" domain, and "feedback" domain in the adversarial gauntlet all use the exact same `MULTI_SCALE_CORPUS` and produce the exact same R values. From the receipt JSON:

- Deep domain R values: [0.708, 0.642, 0.727, 0.959]
- Imbalanced domain R values: [0.708, 0.642, 0.727, 0.959]

These are identical. The "imbalanced" test claims to test whether R is "invariant to sample size" but the sample sizes (64, 20, 5, 2) are simply the sizes of the corpus levels. It is not testing intensivity by subsampling at different rates or drawing independent samples of different sizes. It is computing R on the same embeddings at the same levels.

The "shallow" test just takes the first 2 of those 4 values. The "sparse" test randomly subsamples 20% of each level but uses the same corpus. The "noisy" test adds Gaussian noise to the same corpus embeddings. None of these constitute genuinely different scales or domains.

Furthermore, the scale transformation theory module (`scale_transformation.py`) tests C4 on **synthetic 1D Gaussian data** (np.random.normal), not on real embeddings. The receipt JSON reveals that both C4 tests FAIL on synthetic data:

- C4_Intensivity: CV = 0.499 (threshold 0.2) -- **FAIL**
- C4_Intensivity_ScaleSweep: CV = 1.851 (threshold 0.2) -- **FAIL**

R values across the scale sweep: [7.16, 1.42, 0.71, 0.35, 0.14, 0.07, 0.01, 0.007]. These vary by a factor of **1000x**. This is the opposite of intensive.

The receipt records this failure honestly but the overall verdict still says "CONFIRMED" and the question file still says "C4: PASS." This is because the C4 pass claim uses the real embedding results (CV=0.158) while ignoring the synthetic test failures. The synthetic tests, which are more controlled, show R is emphatically NOT intensive under the scale_transformation.py's coarse-graining operator.

---

## 4. Are the composition axioms independently motivated?

**Verdict: NO. They are chosen to match R's behavior and exclude competitors.**

The axioms are described as "analogous to Q3's axioms A1-A4" and stated to be "the natural extension of Q3's axioms to the multi-scale domain." But Q3's axioms were already identified as circular (P1-02, P1-03 from Phase 1). Extending a circular foundation does not create a non-circular one.

The C3 (functoriality) test is particularly revealing. The threshold for the functoriality L-correlation test was set at 0.5 in the axioms file (`composition_axioms.md` line 183: "Structure preserved > 90%"). But in the cross-scale validation receipt, the actual L-correlations are:

- words->sentences: L_corr = **-0.245** (negative correlation)
- sentences->paragraphs: L_corr = **0.354**
- paragraphs->documents: L_corr = **-1.000**

The **mean L-correlation is -0.297** (negative!). Yet all three pairs are marked as "passes_functoriality: true." This means the pass threshold for functoriality was set low enough to accommodate negative correlations, which contradicts the stated requirement. A negative L-correlation means structure is ANTI-preserved, not preserved.

---

## 5. Formula inconsistency (inherited P1-01)

There are **two different R formulas** in this question's code:

**In `multiscale_r.py` (lines 100-123):**
```
sigma = std(errors)
z = errors / sigma
E = mean(exp(-0.5 * z^2))
R = E / sigma
```

**In `real_embeddings.py` (lines 486-523):**
```
sigma = mean(distances)    # NOT std!
z = distances / sigma
E = mean(exp(-0.5 * z^2))
cv = std(distances) / mean(distances)
concentration = 1 / (1 + cv)
R = E * concentration / sigma
```

These are fundamentally different formulas:
- The first uses standard deviation as sigma and divides E by sigma.
- The second uses mean distance as sigma, adds a concentration factor 1/(1+cv), and multiplies.

The question file (line 162-173) describes yet a third variant with "concentration" included. The GLOSSARY.md defines R = (E / grad_S) * sigma^Df, which is yet another completely different formula involving fractal dimension.

This is P1-01 in full force: the "formula" being tested is not a single well-defined mathematical object. Different code paths compute different things and the results are all attributed to "R."

---

## 6. The RG fixed-point claim is false by the system's own data

The receipt JSON section on RG flow shows:
- beta_values: [-0.095, 0.208, 0.807]
- mean |beta| = 0.307
- The fixed point threshold was set at beta < 0.05

mean |beta| = 0.307 is **6x the threshold**. Yet `is_fixed_point: true` in the receipt. This is because the fixed-point analysis uses the real embedding R values (which are similar by construction), not the beta function computed from those values.

The beta function is explicitly measuring how R CHANGES between scales. A value of 0.807 at the paragraphs->documents transition means R changed substantially. This is not a fixed point. The claim that R is an RG fixed point is directly contradicted by the system's own metric.

---

## 7. Phase transition analysis is a numerical artifact

The percolation analysis (`percolation.py`) does not compute R on real data at all. It:
1. Takes micro-level R values (from a random exponential distribution)
2. Builds a hierarchical tree by averaging groups
3. Thresholds nodes as "active" if R > tau
4. Varies tau and looks for a sharp transition in activity

This is a toy model of percolation on a tree, not a property of R itself. Any scalar quantity that you threshold on a tree will show a percolation-like transition. The "critical exponents" (nu=0.3, beta=0.35, gamma=1.75) do not match 3D percolation reference values (nu=0.88, beta=0.41, gamma=1.8), yet the system reports "universality_class: 3D_percolation" with "is_confident: true." The distance metric of 0.585 between measured and reference exponents is very large. Calling this "confident" is unjustified.

The critical threshold tau_c = 0.1 is reported, but this is from the hardcoded phase transition test, not derived from the real embedding data. The receipt explicitly states the sharpness of the transition is 0.0 -- meaning no detectable phase transition.

---

## 8. The C1 (Locality) test is vacuous

The locality test in `test_q7_axiom_falsification.py` (lines 50-110) computes R on a dataset, then computes R on the exact same dataset again, and checks if they match. Of course they do -- R is a deterministic function. The test never actually introduces non-local information into the R computation. It generates separate "non-local" data but never feeds it into the R calculation. The variable `nonlocal_obs` is created but never used in the R computation. The test always gets error = 0.0, which proves nothing about locality.

---

## 9. Threshold manipulation

Multiple thresholds are set at suspiciously accommodating levels:
- CV threshold for intensivity: 0.3 (in main question file), 0.2 (in axiom tests), 0.5 (in adversarial gauntlet). Three different thresholds for the same property, each used where most convenient.
- Adversarial pass rate: >= 4/6 domains required. This allows 2 failures.
- Cross-scale preservation: no minimum threshold is enforced. The words->sentences preservation of 35% is reported and the test still passes.
- Functoriality L-correlation threshold: effectively allows negative correlations.

---

## Summary of Issues

| Issue | Severity |
|-------|----------|
| Formula inconsistency (2+ different R formulas in code) | CRITICAL |
| C4 (Intensivity) FAILS on synthetic data (CV=0.499 and 1.851) | CRITICAL |
| RG fixed-point claim contradicted by own beta values (mean=0.307) | HIGH |
| Axioms C1-C4 are reverse-engineered to select R (circularity) | HIGH |
| Single toy corpus used for all "multi-scale" tests | HIGH |
| C1 locality test is vacuous (tests identity, not locality) | HIGH |
| "Deep" and "Imbalanced" tests produce identical results | MEDIUM |
| Phase transition exponents do not match claimed universality class | MEDIUM |
| L-correlation is negative yet tests pass functoriality | MEDIUM |
| Threshold manipulation across different tests | MEDIUM |
| Cross-scale preservation of 35% accepted without penalty | LOW |

---

## What IS supported

1. R (as computed in real_embeddings.py) on this specific SentenceTransformer corpus at 4 nesting levels has CV = 0.158. This is a valid empirical observation about this particular formula on this particular data, though the CV is not impressively low.

2. Five specific alternative aggregation operators produce higher CV or break other properties on this dataset. This shows those specific alternatives are worse, not that R is unique.

3. Negative controls (shuffled hierarchy, random R, non-local injection) do produce worse results, confirming R captures some real structure in the embeddings.

---

## Verdict

The multi-scale composition claim is **overclaimed**. The empirical observation that R has moderate stability across nested levels of a single themed corpus is valid but unremarkable. The theoretical claims (RG fixed point, universality, uniqueness, phase transition) are not supported by the evidence. The most critical axiom (C4 intensivity) fails on the system's own synthetic tests. The "proof" of uniqueness is a circular enumeration, not a derivation. The formula being tested is not even consistently defined across the codebase.

Status should be PARTIAL: "R shows moderate stability across nesting levels of a single test corpus; theoretical claims of RG fixed-point behavior, uniqueness, and universality are unsupported."
