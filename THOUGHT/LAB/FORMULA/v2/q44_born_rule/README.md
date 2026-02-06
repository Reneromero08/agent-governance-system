# Q44: E Follows Born Rule Statistics

## Hypothesis
The Essence component (E) of the R formula computes the quantum Born rule probability. Specifically: E = mean(<psi|phi_i>) behaves as a quantum inner product amplitude, and E^2 = |<psi|phi>|^2 equals the Born rule probability exactly. Semantic space is a quantum system (not merely quantum-inspired) where embeddings are quantum state vectors, inner products are quantum amplitudes, and R-gating corresponds to quantum measurement with a threshold. The Born rule structure is universal across all embedding architectures.

## v1 Evidence Summary
v1 claimed "QUANTUM VALIDATED - UNIVERSAL (r = 0.9726 +/- 0.0126)" across 5 architectures:
- Single model (MiniLM-L6): E vs P_born correlation r=0.9768, p=0.000000. E^2 vs P_born: r=1.0000.
- 95% CI for E correlation: [0.968, 0.984]. Spearman rho: 0.9798.
- Cross-architecture (5 models: MiniLM, MPNet, Paraphrase-MiniLM, MultiQA-MiniLM, BGE-small): All r > 0.96.
- Full R formula correlation with P_born: r=0.156 (R_full), r=0.251 (R_simple), r=0.977 (E alone).
- 100 test cases across 4 categories: 30 HIGH, 40 MEDIUM, 20 LOW, 10 EDGE.
- Bootstrap validation: 1000 samples for CI. Permutation: 10000 samples for p-value.
- Synthetic receipt verdict: R correlation 0.265, verdict "NOT_QUANTUM."

## v1 Methodology Problems
Phase 3 verification identified the correlation as a mathematical tautology:

1. **E and P_born are computed from the same quantities.** E = mean(dot(psi, phi_i)) and P_born = mean(|dot(psi, phi_i)|^2). The correlation between mean(x) and mean(x^2) is algebraic, not physical. For any set of positive values x_i, these will be highly correlated.

2. **E^2 = P_born is literally an identity.** E_squared = mean(overlap_i^2) and P_born = mean(|overlap_i|^2). Since overlaps are real-valued, |x|^2 = x^2. The r=1.000 result is computing the same quantity twice.

3. **R itself fails.** The full R formula (R = (E/grad_S) * sigma^Df) gives r=0.156 with P_born. The claim abandons R and promotes E alone, which is just "mean cosine similarity."

4. **No quantum states involved.** Embedding vectors are classical real-valued vectors in R^384 or R^768 produced by neural networks. They are not subject to superposition, entanglement, collapse, or unitary evolution. Dirac notation |psi> is borrowed metaphor.

5. **No null model.** The appropriate null is: "Does mean(x) correlate with mean(x^2) for any set of bounded positive reals?" Answer: yes, always. The correct null hypothesis -- "Does E correlate better with P_born than a random monotone function of cosine similarity?" -- was never tested.

6. **Cross-architecture validation is vacuous.** All models produce normalized vectors. Cosine similarity has the same algebraic properties regardless of which transformer produced it. Testing 5 models tests the same algebra on 5 sets of real vectors.

7. **Entirely post-hoc.** Multiple R variants were tried, the best-correlating one (E) was selected, and the rest were discarded. No prediction was made before the experiment.

8. **Self-contradictory evidence.** The synthetic receipt gives verdict "NOT_QUANTUM" for R. The GLOSSARY and SPECIFICATION acknowledge "not tested on real experimental data."

## v2 Test Plan

### Test 1: Break the Tautology -- Proper Null Model
**Goal:** Determine whether the E-P_born correlation exceeds what any monotone function of cosine similarity would produce.
**Method:**
- Compute cosine similarities {x_i} between psi and context vectors {phi_i}
- Compute E = mean(x_i) and P_born = mean(x_i^2)
- Also compute 20 alternative "measurement probabilities": mean(x^3), mean(exp(x)), mean(log(1+x)), mean(x/(1+x)), mean(sin(pi*x/2)), Gini(x), median(x)^2, etc.
- Correlate each with P_born
- If E is specifically the Born rule, its correlation should be uniquely high. If mean(x^k) for any k near 1 also gives r>0.95, the Born rule claim is falsified.
- Report all correlations, not just the best one

### Test 2: Interference Test
**Goal:** Test for genuine quantum interference in semantic space.
**Method:**
- The Born rule arises from quantum interference: P(a or b) != P(a) + P(b) due to cross-terms
- Construct "superposition" states: psi = alpha*w1 + beta*w2 (linear combination of two word embeddings)
- Compute P_born for the superposition and compare to alpha^2*P(w1) + beta^2*P(w2)
- If Born rule applies: P(superposition) should contain interference terms 2*alpha*beta*Re(<w1|phi><phi|w2>)
- If classical: P(superposition) = weighted average with no interference
- Measure the interference term explicitly and test whether it is statistically significant
- Null: for classical (real-valued, linear) systems, the "interference" term reduces to a simple inner product -- distinguish this from genuine quantum interference

### Test 3: Complementarity and Uncertainty
**Goal:** Test whether semantic space has complementary observables that satisfy an uncertainty relation.
**Method:**
- In quantum mechanics, conjugate observables (position/momentum, spin-x/spin-z) satisfy Heisenberg uncertainty
- Define candidate semantic observables: (a) topic similarity, (b) syntactic role similarity, (c) sentiment similarity
- For each pair, compute product of uncertainties: Delta(A)*Delta(B)
- Test whether there is a lower bound (uncertainty relation) or whether both can be simultaneously sharp
- If no uncertainty relation exists, the space is classical

### Test 4: Full R Formula vs. Born Rule
**Goal:** Honestly assess the full R formula's relationship to Born-rule-like statistics.
**Method:**
- Compute the FULL R = (E/grad_S)*sigma^Df for 1000+ test cases
- Compute P_born = mean(|<psi|phi_i>|^2)
- Report the correlation honestly (v1 found r=0.156)
- If R does not correlate with P_born, the "Born rule" claim is specific to E, not to R
- Test whether grad_S and sigma^Df serve as normalization that could recover the Born rule structure

### Test 5: Real Quantum System Comparison
**Goal:** Compare embedding space statistics to actual quantum experimental data.
**Method:**
- Obtain published Born rule experimental data (e.g., photon polarization, Stern-Gerlach, double slit)
- Compute analogous statistics in embedding space
- Compare: do embedding E values follow the SAME distribution as quantum probabilities in real experiments?
- If the distributions differ qualitatively (e.g., embeddings show no interference fringes), the quantum interpretation fails
- This is the most decisive test: comparison to actual quantum phenomena

## Required Data
- Pre-trained embeddings from 5+ architectures (including non-transformer: GloVe, Word2Vec)
- Standard semantic benchmarks for test cases
- Published quantum experimental data: photon polarization measurements, Stern-Gerlach results, double-slit patterns
- Random vector baselines in matching dimensions
- 20+ monotone functions of cosine similarity for null model testing

## Pre-Registered Criteria
- **Success (confirm):** Interference terms are statistically significant (p < 0.001) AND qualitatively match quantum predictions (constructive/destructive pattern), AND uncertainty relations exist for at least one pair of semantic observables, AND E outperforms all 20 alternative monotone functions by >0.05 in correlation with a genuine Born-rule probability from a quantum system
- **Failure (falsify):** E's correlation with P_born is matched (within 0.02) by 5+ other monotone functions, OR interference terms are not significant (p > 0.05), OR no uncertainty relations exist, OR embedding statistics diverge qualitatively from real quantum experimental data
- **Inconclusive:** Interference terms marginally significant (0.001 < p < 0.05), or 1-4 alternative functions match E's correlation

## Baseline Comparisons
- Tautology null: correlation between mean(x) and mean(x^2) for random positive reals in [0,1] -- establishes the algebraic baseline
- 20 alternative functions: if mean(x^1.5) or median(x)^2 correlates equally well, E is not special
- Random vectors: compute E and P_born for random unit vectors -- what correlation arises from geometry alone?
- Classical wave analogy: classical wave interference produces cross-terms too -- distinguish quantum from classical interference

## Salvageable from v1
- The cross-architecture data showing consistent E behavior across 5 models is genuine (even if the interpretation is wrong)
- The test infrastructure (100 test cases, 4 categories, bootstrap/permutation validation) is well-designed
- The honest reporting of R_full correlation (r=0.156) is valuable -- the failure of the full formula is a real finding
- The observation that E (mean cosine similarity) is the "active ingredient" in R is empirically useful, even if it is not quantum
- The GLOSSARY/SPECIFICATION admission ("not tested on real experimental data") is honest and should guide v2 toward real quantum comparisons
