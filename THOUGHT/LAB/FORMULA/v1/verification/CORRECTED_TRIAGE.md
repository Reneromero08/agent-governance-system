# Living Formula: Corrected Salvageability Triage

**Date:** 2026-02-05
**Correction:** Previous triage confused "bad test" with "false hypothesis."
**Principle:** A flawed test tells you the test was flawed. It tells you NOTHING about the hypothesis. Only a sound test can falsify.

---

## Categories

- **A: Properly Falsified** -- A methodologically sound test produced a clear negative result. The hypothesis is dead. Accept it.
- **B: Properly Confirmed** -- A methodologically sound test produced a clear positive result. The hypothesis stands.
- **C: Improperly Tested** -- The test had fatal methodology flaws (wrong E, circular setup, synthetic-only, tautological). The hypothesis remains **OPEN**. Needs a proper test.
- **D: Untested** -- No test was ever conducted. The hypothesis remains **OPEN**. Needs a first test.

---

## Summary

| Category | Count | Meaning |
|----------|-------|---------|
| A: Properly Falsified | 6 | Hypothesis dead, by valid evidence |
| B: Properly Confirmed | 3 | Hypothesis stands, by valid evidence |
| C: Improperly Tested | 33 | Hypothesis OPEN -- test was bad, not the idea |
| D: Untested | 12 | Hypothesis OPEN -- nobody tried yet |

**44 of 54 hypotheses remain open.** They need proper tests, not burial.

---

## A: Properly Falsified (6)

These had sound methodology and clear negative results. The hypothesis is dead.

| Q | Hypothesis | What Falsified It | Why the Test Is Valid |
|---|-----------|-------------------|---------------------|
| Q22 | Universal R threshold exists | 3/7 real domains fail 10% criterion | 7 real external datasets (STS-B, SST-2, SNLI, Market, AG-News, Emotion, MNLI). Pre-registered threshold. Clear pass/fail. |
| Q23 | sqrt(3) has geometric derivation | Alpha ranges 1.4-2.5 across models | Multi-model grid search. 3 geometric theories tested and failed. The search was methodologically sound. |
| Q25 | Sigma is universal (e^(-4/pi)) | Sigma varies 15x across real domains (1.92-39.44) | Sigma = std of observations is E-definition-independent. Measured on real data. R^2_cv = 0.0. |
| Q42 | R exhibits quantum nonlocality | S_max = 0.36 (far below classical bound of 2.0) | Valid CHSH apparatus (quantum states hit Tsirelson bound, classical states hit Bell bound). The apparatus works; embeddings just don't violate Bell inequalities. |
| Q52 | R negatively correlates with chaos | Positive Lyapunov correlation found | Logistic map sweep verified against theoretical Lyapunov values (0.004% error). The measurement is clean. Correlation direction is E-independent. |
| Q53 | Pentagonal/phi geometry in embeddings | 3/5 tests falsified, 72-deg = arccos(0.3) | 5 real embedding models. 4 independent audits agree. The 72-degree finding has a known geometric explanation (typical cosine similarity for related words). |

**Note:** Q52 and Q53's falsifications are actually productive -- they found something real (R measures dimensionality, 72-deg is a cosine similarity property). These deserve follow-up, but the ORIGINAL hypotheses are dead.

---

## B: Properly Confirmed (3)

These had sound methodology and positive results.

| Q | Hypothesis | What Confirmed It | Caveats |
|---|-----------|-------------------|---------|
| Q16 | R discriminates domain boundaries | SNLI (n=500) and ANLI (n=300) from HuggingFace. Reproducible pipeline. Three independent audits. | ANLI showed weaker discrimination (adversarial NLI is harder). Confirms a known property of cosine similarity, not necessarily a novel finding. |
| Q26 | Minimum data requirements can be determined | Multi-model bootstrap (7 configs). N_min=5 for coherent content. | Only D=384 and D=768 tested. Engineering guidance, not theoretical bound. |
| Q30 | Fast approximations of R exist | 8 methods benchmarked. 100-300x speedup. Pareto frontier. | Synthetic data only. 250% R-value error needs attention. Speedup is real; accuracy tradeoff needs real-data validation. |

---

## C: Improperly Tested (33)

**The hypothesis remains OPEN.** The test was flawed, not the idea. Each needs a proper test designed.

### C1: Test used wrong E definition or circular setup (15)

These have existing code and results, but the E definition crisis or circular reasoning invalidates the test -- NOT the hypothesis.

| Q | Hypothesis | What Went Wrong With the Test | What a Proper Test Looks Like |
|---|-----------|-------------------------------|-------------------------------|
| Q1 | grad_S is the uniquely correct normalization | Uniqueness proof encodes answer in axiom. Free Energy identity uses Gaussian E, not operational cosine E. | Head-to-head benchmark: E/sigma vs bare E vs E/MAD vs other normalizations on real NLP tasks (STS-B, SNLI, etc.). Let the data pick the winner. |
| Q2 | These criteria can falsify R | Test code uses E=1/(1+std) instead of GLOSSARY E. Independence escape hatch makes criteria unfalsifiable. | Rewrite criteria with modus tollens structure, numerical thresholds, one consistent E. Then attempt to falsify R using the criteria. |
| Q3 | R generalizes across domains | "Cross-domain" = same formula on different synthetic arrays. Axiom A4 IS the conclusion. | Test R on genuinely different domains (text, audio, image, genomic embeddings) using consistent E. Compare generalization to simpler metrics. |
| Q7 | R composes across scales | Two different R formulas in code. C4 intensivity test fails. Single toy corpus. | Use one R formula. Test on real multi-scale data (word -> sentence -> paragraph -> document). Accept intensivity failure if it replicates. |
| Q9 | log(R) = -F + const (FEP connection) | Identity proven for Gaussian E, not operational cosine E. Empirical R-F correlation is -0.23 with possibly wrong E. | Compute R with consistent cosine E. Compute F from actual variational model. Measure correlation. If weak, the connection is notational only. |
| Q12 | R shows phase transition during training | Weight interpolation is not real training. Tests 8-9 reverse-engineered to pass. | Use real training checkpoints from actual model training runs. Measure R at each checkpoint. Look for discontinuities. |
| Q13 | 36x universal ratio exists | "Blind prediction" = f(x)==f(x). Ratio depends on E_MIN clamping parameter. | Remove clamping. Measure actual R distribution across domains. If ratio depends on arbitrary parameters, it is not universal. |
| Q15 | R has Bayesian interpretation | r=1.0 is identity 1/x=1/x when E hardcoded to 1. Earlier proper test showed falsification but was "rescued." | Use consistent operational E. Compute R. Compute proper Bayesian posterior precision. Compare. Accept whatever the correlation shows. |
| Q21 | dR/dt predicts system degradation | No real temporal data. dR/dt itself has AUC=0.10. Alpha-drift AUC=0.9955 but synthetic. | Use real temporal data (evolving conversations, model fine-tuning logs, document revision histories). Measure dR/dt and alpha-drift against known degradation events. |
| Q27 | R exhibits hysteresis | Own admission: answer is NOT hysteresis. Noise injection test measures selection bias. | Design test that actually measures hysteresis (path-dependent R values under cycling conditions). If R doesn't depend on path history, falsify honestly. |
| Q33 | R shows emergent properties | Own document admits "tautology by construction." sigma^Df = N is circular. | Design non-circular emergence test. Measure whether R at macro scale is predictable from micro-scale R values. If predictable, no emergence. |
| Q40 | Embeddings exhibit quantum error correction | Alpha drift detection is real finding but QECC framework was never actually tested. | Either formally map to QECC (stabilizer codes, syndrome measurement) or test alpha-drift as a classical corruption detector against real baselines. |
| Q43 | Embedding spaces have Quantum Geometric Tensor | Code computes np.cov() and labels it QGT. Berry curvature = 0 for real vectors (mathematical proof). | Berry curvature IS properly proven zero for real vectors. But: does a real-valued analog of QGT exist? Test whether covariance structure captures meaningful embedding geometry that other metrics miss. |
| Q44 | E = |<psi|phi>|^2 (Born Rule) | Test was E vs E^2 from same computation (tautology). R scored r=0.156 but with potentially wrong E. | Compute cosine similarities (operational E). Compute Born probabilities from properly normalized state vectors. Compare. Use consistent E throughout. The r=0.156 result needs replication with correct E. |
| Q50 | Df*alpha is conserved across models | 24 "independent" models are ~19 transformers on overlapping data. CV=6.93% is not physics-grade. | Use genuinely independent architectures (word2vec, GloVe, BERT, GPT, Llama, vision transformers, audio models). Measure Df*alpha. Report true variance. Accept whatever CV you get. |

### C2: Test methodology was sound but incomplete (10)

These have some valid results but the testing was insufficient to draw a conclusion about the hypothesis.

| Q | Hypothesis | What's Valid | What's Missing |
|---|-----------|-------------|----------------|
| Q4 | Formula makes novel predictions | THINGS-EEG test methodology was valid. Result: null (|r|=0.109, p=0.266). | One null result on one prediction doesn't falsify "makes ANY novel predictions." Need to identify what the formula uniquely predicts (vs. what cosine similarity alone predicts) and test those specific predictions. |
| Q10 | R detects alignment | Spectral contradiction test well-designed. Finding: raw E outperforms R (4.33x vs 1.79x). | The E>R finding is real but on limited test. Needs validation on real alignment benchmarks (TruthfulQA, ETHICS). If E consistently beats R, that's a real finding about the formula's added value. |
| Q14 | Category theory illuminates R | R-cover NOT a Grothendieck topology (valid negative). Non-monotonicity finding genuine. | Sheaf satisfaction at 97.6% needs proper mathematical assessment (sheaf requires 100%). Need to determine if any categorical structure is rigorously present, not just approximately. |
| Q19 | R correlates with value learning | Real data (OASST, SHP, HH-RLHF). Simpson's Paradox correctly identified. OASST r=0.505 positive. | 1/3 datasets positive, 2/3 null or negative. Need more datasets with genuine multi-annotator agreement labels to determine if OASST is the signal or the outlier. |
| Q24 | R failure modes are characterizable | Real SPY market data (751 points). WAIT strategy honestly falsified. | n=17 low-R periods is marginal. Single domain. Replicate on 2+ additional domains with larger n to confirm failure mode taxonomy. |
| Q28 | R has attractor structure | Real market data. Autocorrelation 0.70-0.90 across regimes. | Threshold manipulation caught by audits. Need pre-registered thresholds. Need non-financial domains. Need to distinguish "attractor" from "mean-reverting stochastic process." |
| Q32 | Meaning behaves like a physical field | Real public data (SciFact, Climate-FEVER, SNLI, MNLI). Working negative controls. Honest EEG null. | NLI cross-encoder does the heavy lifting. Need ablation: does the R formula add value beyond what the NLI model provides alone? Test propagation, conservation, superposition -- actual field properties. |
| Q34 | Embedding models converge to shared geometry | Eigenvalue correlation across models is real observation. | Shared training data confound is unaddressed. Need controlled experiment: train models on non-overlapping corpora, measure convergence. If it persists, the finding is real. |
| Q37 | Semiotic properties evolve with patterns | Real HistWords/WordNet data. Phylogenetic FMI=0.60. | Results replicate known computational linguistics findings (Hamilton et al. 2016). Need to show R adds insight beyond existing metrics. Cross-lingual tests used multilingual models (architectural confound). |
| Q29 | R is numerically stable | div/0 fix works. | sigma^Df overflow (10^47) completely unaddressed. Edge case R values span 6 orders of magnitude. Need comprehensive stability analysis across the full parameter space. |

### C3: Test was tautological or self-referential (8)

These tests proved properties of the test setup, not of the hypothesis.

| Q | Hypothesis | Why the Test Was Tautological | What a Real Test Looks Like |
|---|-----------|-------------------------------|----------------------------|
| Q5 | High agreement reveals truth | "Agreement = truth" asserted, not tested. No scenario where agreement diverges from truth. | Find datasets where consensus is wrong (flat earth surveys, historical scientific errors, adversarial crowd manipulation). Test whether R is high for wrong-but-agreed-upon claims. |
| Q6 | R connects to IIT (Phi) | "Consensus filter discovery" = restating R's definition. No formal Phi computation. | Compute actual IIT Phi for small systems. Compute R for same systems. Measure correlation. If no correlation, connection is verbal only. |
| Q11 | R has predictable failure modes (valley blindness) | R was never computed in any of 12 tests. Taxonomy is philosophical. | Actually compute R in scenarios designed to trigger each failure mode. Measure whether the predicted failures occur. |
| Q20 | R is not tautological | Examined whether 8e is universal (wrong question). Never examined whether R=E/sigma is a tautological SNR. | Test whether R predicts anything that signal-to-noise ratio (E/sigma) does not. If R=E/sigma makes identical predictions to generic SNR, R is tautological. |
| Q36 | R connects to Bohm's implicate order | SLERP conservation counted 4 times. V7 removed 5/10 tests as wrong. Metaphorical only. | Define specific testable mapping between Bohm's algebra and R's mathematical structure. Test whether Bohmian predictions differ from classical predictions for R. |
| Q38 | R obeys Noether conservation | Geodesics conserve angular momentum by definition. CV=10^-15 tests NumPy, not meaning. | Define a genuine time evolution for R. Identify a symmetry. Derive the conserved quantity from Noether's theorem. Test whether it is conserved in real evolving data. |
| Q39 | R exhibits homeostatic regulation | "Recovery" is SLERP returning to predetermined path. Test design creates the result. | Perturb real embedding systems (inject noise into fine-tuning, corrupt training data). Measure whether R returns to baseline WITHOUT designed-in recovery mechanism. |
| Q45 | Pure geometry suffices for navigation | sigma^Df overflow forced abandonment of R. E alone was tested. Navigation = word2vec analogy (Mikolov 2013). | Test whether R (full formula, with overflow handled) provides better navigation than E alone and better than word2vec arithmetic. On real tasks (analogy, nearest neighbor, document retrieval). |

---

## D: Untested (12)

No test was ever conducted. The hypothesis is completely open.

| Q | Hypothesis | What Exists | What's Needed |
|---|-----------|-------------|---------------|
| Q8 | Embedding spaces have topological structure | Kahler test = false. No persistent homology computed. FALSIFIED status suppressed. | Actually compute persistent homology (Betti numbers, persistence diagrams) on real embedding subsets. This is a well-defined computation. The topology question is real -- nobody answered it. |
| Q17 | R-gating improves governance decisions | 1617-line spec. Zero performance data. Tests verify arithmetic, not effectiveness. | Run R-gating on real decision scenarios with ground truth. Measure precision/recall/F1. Compare to baselines (random, threshold on E alone, human judgment). |
| Q18 | R detects deception | Red team found 3/5 falsified. But red team methodology itself needs assessment. | Test on established deception benchmarks (LIAR dataset, FEVER, propaganda detection). Compare to existing deception detection tools. Accept whatever the comparison shows. |
| Q31 | R enables compass-mode navigation | Zero implementations. Zero tests. "Confirmed" by citing Q43's tautological result. | Build the compass. Test it on real navigation tasks (document retrieval, semantic search). Benchmark against existing methods (BM25, dense retrieval, FAISS). |
| Q35 | R threshold defines Markov blanket | Zero experiments, zero data, zero scripts. Conceptual mapping only. | Define the Markov blanket formally in terms of R. Test conditional independence property: given R-boundary observations, are interior and exterior observations independent? Use real data. |
| Q41 | Geometric Langlands structure in embeddings | No actual Langlands math. K-means called "primes." 0.53 S-duality. | Either engage with actual Langlands mathematics (automorphic forms, L-functions, Hecke operators) or acknowledge the vocabulary was aspirational. If pursuing: compute actual L-functions for embedding spaces. |
| Q46 | Geometric properties are stable under perturbation | 23-line placeholder. No hypothesis, no experiments. | Define "geometric stability" precisely. Perturb embeddings (noise injection, dimension reduction, quantization). Measure which geometric properties survive. |
| Q47 | Bloch sphere representation is meaningful | 23-line placeholder. Premise may conflict with Q42 (R is classical). | Either: (a) define a classical Bloch sphere analog for embeddings and test it, or (b) acknowledge Q42 shows R is classical and reformulate. |
| Q48 | Eigenvalue statistics connect to Riemann zeta | GUE hypothesis falsified (Poisson found). Semantic zeta has no functional equation, no Euler product. | The GUE mechanism is dead. But: is there ANY connection between eigenvalue distributions and number-theoretic objects? Test other RMT ensembles (GOE, GSE, Wishart). If all fail, the connection is dead. If one matches, investigate why. |
| Q49 | Df*alpha ~ 8e for a reason | Monte Carlo p=0.55 (if properly designed). Own docs: 15% confidence. | First: validate the Monte Carlo methodology. If p=0.55 holds up, 8e is not special and the "why" question is moot. If Monte Carlo was flawed, redesign and rerun. |
| Q51 | Embeddings have intrinsic complex structure | 0/19 on random bases (if test was valid). PCA projection imposes structure. | The random basis test needs methodology review. If valid, intrinsic complex structure is falsified. But: test for INDUCED complex structure (complexification of real space). Also test whether phase-like quantities (angles between subspaces) carry semantic meaning. |
| Q54 | Energy-like quantity is conserved in embeddings | All 5 cited foundations compromised. Zero external data. | Start from scratch. Define "energy" operationally in embedding space. Measure it across real processes (training, fine-tuning, inference). Test whether it is conserved, increases, decreases, or has no pattern. |

---

## Corrected Summary

| Previous Triage | Corrected Triage | What Changed |
|-----------------|-----------------|--------------|
| 6 Tier 1 (Valid) | 3 Confirmed + 6 Falsified = 9 properly tested | Separated confirmed from falsified |
| 11 Tier 2 (Needs data) | Merged into C (improperly tested) | "Needs data" was often "needs proper test" |
| 14 Tier 3 (Wrong answer) | Merged into C (improperly tested) | "Wrong answer" was really "untested hypothesis" |
| 23 Tier 4 (Dead) | 12 Untested + rest moved to C | **Most "dead" hypotheses were never properly tested** |

### The Honest Scoreboard

| Status | Count | Qs |
|--------|-------|-----|
| **Properly Falsified** | 6 | Q22, Q23, Q25, Q42, Q52, Q53 |
| **Properly Confirmed** | 3 | Q16, Q26, Q30 |
| **Hypothesis OPEN (needs proper test)** | 45 | Everything else |

**83% of hypotheses have never been properly tested.** The project's problem is not that the ideas are wrong -- it's that the tests were not rigorous enough to determine whether the ideas are right or wrong.

---

## What v2 Should Be

For each of the 45 open hypotheses:
1. **State the original hypothesis exactly as intended**
2. **Document what went wrong with the test** (from verification)
3. **Design a proper test** with:
   - Consistent E definition (ONE, throughout)
   - Real external data source (named specifically)
   - Pre-registered success/failure criteria (numerical)
   - Comparison to baseline (what does bare cosine similarity do?)
   - No post-hoc metric swapping
4. **Run the test**
5. **Accept the result** -- confirmed, falsified, or inconclusive

The 6 properly falsified hypotheses get documented as falsified with their evidence. The 3 confirmed ones get documented with their evidence. That's the honest starting point.

---

*Corrected triage: 2026-02-05*
*Principle applied: "bad test" != "false hypothesis"*
