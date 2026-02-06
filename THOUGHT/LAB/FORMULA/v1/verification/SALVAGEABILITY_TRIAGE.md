# Living Formula: Salvageability Triage

**Date:** 2026-02-05
**Reviewer:** Claude Opus 4.6 (adversarial skeptic)
**Source:** All 34 verdict files from 6-phase adversarial verification
**Methodology:** Every Q rated on what ACTUALLY EXISTS (code, data, results), not what is claimed.

---

## How to Read This Document

Each Q is classified into exactly one tier:

- **Tier 1: Already Valid or Near-Valid** -- Real work exists, methodology sound, honest claims. Needs only minor corrections.
- **Tier 2: Salvageable Question, Needs Real Data** -- The question is scientifically meaningful AND real code/methodology exists, but all evidence is synthetic. Running on real external data could produce genuine results.
- **Tier 3: Salvageable Question, Wrong Answer** -- The question is worth asking, but the current answer is wrong (circular, tautological, overclaimed). Must restart from scratch with correct methodology.
- **Tier 4: Dead** -- Either the question is ill-posed, the answer is unfalsifiable, the foundations are demolished, or there is nothing real to build on.

**Rule applied throughout:** If a Q has no real code, no real data, and no testable methodology, it is Tier 4 regardless of how interesting the question sounds.

---

## Summary Counts

| Tier | Count | Percentage |
|------|-------|------------|
| Tier 1: Already Valid | 6 | 11.1% |
| Tier 2: Salvageable, Needs Real Data | 11 | 20.4% |
| Tier 3: Salvageable, Wrong Answer | 14 | 25.9% |
| Tier 4: Dead | 23 | 42.6% |

---

## TIER 1: Already Valid or Near-Valid (6 Qs)

These have real methodology, real (or at least honest) results, and need only minor framing corrections.

---

### Q16: Domain Boundaries (R=1440)
**What exists:** Real external data (SNLI n=500, ANLI n=300 from HuggingFace), reproducible pipeline, honest negative result on ANLI, proper controls, effect sizes reported. Three independent audits all reproduce identical numbers.
**Fix needed:** Reframe from "R discovers domain boundary" to "cosine similarity distinguishes topical shifts but not adversarial logical contradictions" -- which is a known property, not a discovery.
**Honest achievable claim:** R (via cosine similarity) reliably detects topical divergence but fails on adversarial NLI. ANLI failure is a genuine, useful boundary characterization.

---

### Q22: Threshold Calibration (R=1320, FALSIFIED)
**What exists:** Pre-registered hypothesis tested on 7 real-world domains (STS-B, SST-2, SNLI, Market, AG-News, Emotion, MNLI). Three independent audits. Clean, decisive falsification: only 3/7 domains pass the 10% universal threshold criterion.
**Fix needed:** None. This is already correctly labeled FALSIFIED. The negative result is the contribution.
**Honest achievable claim:** No universal R threshold exists. Domain-specific calibration is required. Confirmed on 7 external datasets.

---

### Q23: sqrt(3) Geometry (R=1300, CLOSED)
**What exists:** Multi-model grid search (5 models), honest falsification of 3 geometric theories (hexagonal, Berry phase, winding angle), proper negative controls.
**Fix needed:** Propagate the finding that sqrt(3) is empirically fitted (not geometrically derived) to all documents that claim theoretical grounding for the formula's constants.
**Honest achievable claim:** The optimal alpha parameter is in the range 1.4-2.5 and is model-dependent. No geometric derivation of sqrt(3) survives testing.

---

### Q42: Nonlocality / Bell's Theorem (R=1400)
**What exists:** Validated CHSH apparatus (quantum states hit Tsirelson bound, classical states hit Bell bound), honest null result on semantic embeddings (S_max=0.36, far below classical bound of 2.0), R vs. Phi complementarity test.
**Fix needed:** Acknowledge the null result was trivially expected (classical correlations cannot violate Bell inequalities by definition). Reduce R-score to reflect this.
**Honest achievable claim:** R is fundamentally local (as guaranteed by Axiom A1). No quantum nonlocality in embedding spaces. R and Phi measure complementary properties.

---

### Q52: Chaos Theory (R=1180, FALSIFIED then RESOLVED)
**What exists:** Logistic map sweep (n=100, r=2.5 to 4.0), Henon attractor test, Lyapunov computation verified against theoretical values (0.004% error at r=4.0). Pre-registered hypothesis cleanly falsified (positive correlation found instead of negative).
**Fix needed:** Correct the overclaim that participation ratio equals fractal dimension. Note 2/5 hypotheses remain untested.
**Honest achievable claim:** R (participation ratio) increases with chaos, not decreases. R measures effective dimensionality of attractors, not predictability. Original hypothesis falsified; reinterpretation as dimensionality measure is physically sound.

---

### Q53: Pentagonal Phi Geometry (R=1200, FALSIFIED)
**What exists:** Real sentence-transformer embeddings (5 models), comprehensive test battery (5 tests), four independent audits (DEEP, OPUS, ULTRA_DEEP, VERIFY). 3/5 tests outright falsified, 1/5 is semantic artifact, 1/5 invalid test.
**Fix needed:** Update main file status from PARTIAL to FALSIFIED (all four audits agree). Fix test code verdict logic that still outputs "SUPPORTED."
**Honest achievable claim:** No pentagonal or phi geometry exists in embedding spaces. The 72-degree clustering is arccos(0.3), a consequence of typical cosine similarity for related words.

---

## TIER 2: Salvageable Question, Needs Real Data (11 Qs)

These have a meaningful question, some existing methodology or code, but are currently validated only on synthetic data. Running on real external data is the path forward.

---

### Q1: Why grad_S? (R=1800)
**What exists:** Location-scale normalization derivation (valid within scope), synthetic tests showing E/grad_S captures signal-to-noise. Free Energy identity is algebraic tautology by construction.
**Fix needed:** Test E/grad_S (with GLOSSARY-consistent E = cosine similarity) on real NLP benchmarks. Drop uniqueness proof (circular). Drop Free Energy identity (tautological).
**Honest achievable claim:** Location-scale normalization (E/sigma) is a reasonable signal-to-noise measure for embedding agreement, pending real-data validation.

---

### Q10: Alignment Detection (R=1560)
**What exists:** Code for alignment detection tests, spectral contradiction experiment (well-designed, honestly falsified). VALUE_ALIGNMENT test fails (ratio=0.99). Raw E outperforms R (4.33x vs 1.79x).
**Fix needed:** Test on real alignment benchmarks (e.g., TruthfulQA, ETHICS dataset). Address the finding that raw E outperforms R. Drop the FALSE POSITIVE problem (consistently-wrong systems get high R).
**Honest achievable claim:** E (cosine similarity) may detect misalignment in some settings. The full R formula does not outperform E alone for alignment detection.

---

### Q12: Phase Transitions (R=1520)
**What exists:** Weight interpolation framework, statistical mechanics test suite (12 tests, 10 genuine), cross-architecture comparison. Tests 8 and 9 reverse-engineered to pass.
**Fix needed:** Replace weight interpolation with ACTUAL training checkpoints. Remove Tests 8 and 9 (constructed to guarantee passage). Test on real partially-trained models.
**Honest achievable claim:** Eigenspectrum statistics change non-linearly during training (interpolation shows this). Whether this constitutes a genuine phase transition requires real training checkpoint data.

---

### Q19: Value Learning (R=1380)
**What exists:** Real external data (OASST, SHP, HH-RLHF from HuggingFace), pre-registered hypothesis, Simpson's Paradox correctly identified and reported. Resolved test methodology is sound.
**Fix needed:** More real datasets with genuine multi-annotator agreement labels (OASST is the only positive signal, 1/3). Drop HH-RLHF length proxy. Pre-register the log transform.
**Honest achievable claim:** R may correlate with inter-annotator agreement in well-annotated datasets (OASST r=0.505), but 2/3 datasets show null or negative correlation. Inconclusive.

---

### Q24: Failure Modes (R=1280)
**What exists:** Real external data (3 years SPY market data via yfinance, 751 points), four strategies tested, WAIT hypothesis honestly falsified (0% success).
**Fix needed:** Replicate on at least 2 more domains (text, genomics). Increase n beyond 17 low-R periods. Test HIGH_SIGMA-only and LOW_E-only closure reasons (currently only BOTH tested).
**Honest achievable claim:** In financial time-series, waiting when R is low is counterproductive; changing observation window helps. Single-domain finding, n=17.

---

### Q26: Minimum Data Requirements (R=1240)
**What exists:** Multi-model bootstrap (7 configurations), real embedding models, honest self-correction from underpowered original test. N_min=3 for diverse content, N_min=5 for coherent content.
**Fix needed:** Test with more dimensionalities from independently trained models (currently only D=384 and D=768). Pre-register the semantic structure finding. Provide theoretical sample complexity bound.
**Honest achievable claim:** For sentence-transformer embeddings, N >= 5 observations produces stable R (CV < 10%). Practical engineering guidance, not a theoretical bound.

---

### Q28: Attractors (R=1200)
**What exists:** Real market data (SPY via yfinance), 7 market regimes, autocorrelation analysis (0.70-0.90 across all regimes). Two independent audits caught threshold manipulation.
**Fix needed:** Use original pre-registered thresholds (42.8% pass) or transparently justify relaxation. Replicate on non-financial domains. Properly classify as "mean-reverting stochastic process" instead of "attractor."
**Honest achievable claim:** R values computed from SPY market data show mean-reversion (high autocorrelation), not chaos. Single-domain finding with threshold manipulation caveat.

---

### Q30: Approximations (R=1160)
**What exists:** 8 approximation methods benchmarked, Pareto frontier analysis, scaling behavior verified. 100-300x speedup demonstrated. Negative result honestly reported (combined method 83.3%).
**Fix needed:** Test on real embeddings (currently synthetic only). Test gate accuracy near threshold boundary. Elevate the 250% R-value error to a primary finding. Provide analytical error bounds.
**Honest achievable claim:** Random sampling of pairwise comparisons achieves 100-300x speedup with correct binary gate decisions on synthetic data. R-value accuracy has up to 250% error.

---

### Q32: Meaning as Physical Field (R=1670)
**What exists:** Best methodology in the project. Real public data (SciFact, Climate-FEVER, SNLI, MNLI). Working negative controls. Honest null reporting (Phase 7 EEG FAIL). NLI cross-encoder does heavy lifting.
**Fix needed:** Test whether grad_S (the scaling term) actually adds value -- the no_scale ablation does not kill the effect. Test echo-chamber prediction on real correlated data. Drop the "field" label (no field equations, propagator, or conservation laws).
**Honest achievable claim:** A scoring function built on NLI cross-encoder + cosine similarity can distinguish factual claims from misinformation on real benchmarks. The "field" metaphor is unjustified.

---

### Q34: Platonic Convergence (R=1510)
**What exists:** Genuine observation (eigenvalue correlation across models). Multiple models compared. R never computed anywhere in Q34.
**Fix needed:** Address the shared training data confound. Replace fabricated null baseline with proper random/shuffled baselines. Cite the existing NLP literature on representation convergence.
**Honest achievable claim:** Trained embedding models show correlated eigenspectrum structure, likely due to shared training data and objectives. This is a known observation in the NLP community, not a Platonic discovery.

---

### Q37: Semiotic Evolution (R=1380)
**What exists:** Real data (HistWords, WordNet, multilingual embeddings). 15 tests across 4 tiers. Some genuine findings (phylogenetic reconstruction FMI=0.60).
**Fix needed:** Remove the "symmetry breaking" framing (no symmetry identified or broken). Acknowledge cross-lingual convergence is architecturally imposed. Use independent monolingual models for true cross-lingual tests. Drop the Df*alpha=8e conservation law (varies 2.6x between systems).
**Honest achievable claim:** Embedding statistics change over time in ways consistent with known computational linguistics findings (Hamilton et al. 2016). Cross-model spectral stability within model families. No symmetry breaking demonstrated.

---

## TIER 3: Salvageable Question, Wrong Answer (14 Qs)

These ask questions worth answering, but the current answer is fundamentally wrong -- circular, tautological, or built on falsified foundations. The question itself can be restarted with correct methodology.

---

### Q2: Falsification Criteria (R=1750)
**What exists:** Documented criteria. Independence escape hatch identified as making criteria unfalsifiable.
**Fix needed:** Complete rewrite with modus tollens structure, numerical thresholds, and no independence loophole.
**Honest achievable claim (if fixed):** A set of concrete, falsifiable criteria for the R formula.

---

### Q3: Why Does It Generalize? (R=1720)
**What exists:** Tests across "domains" that are actually the same computation on different distributions. Axiom A4 IS the conclusion.
**Fix needed:** Test on genuinely different domains with independent E definitions. Remove circular axiom dependency.
**Honest achievable claim (if fixed):** Whether E/sigma generalizes across genuinely different measurement contexts.

---

### Q5: Agreement vs. Truth (R=1680)
**What exists:** Philosophical argument. No empirical test of whether high agreement implies truth vs. convention.
**Fix needed:** Design empirical test with known-ground-truth data where agreement and truth diverge (e.g., expert vs. crowd consensus on factual claims).
**Honest achievable claim (if fixed):** Under what conditions high inter-observer agreement correlates with factual accuracy.

---

### Q7: Multi-scale Composition (R=1620)
**What exists:** Code for multi-scale tests. C4 (Intensivity) FAILS on synthetic data. Two different R formulas in code. Single toy corpus.
**Fix needed:** Fix intensivity test or accept R is not intensive. Use consistent R formula. Test on real multi-scale data (word -> sentence -> paragraph).
**Honest achievable claim (if fixed):** Whether R computed at different scales gives consistent results (current evidence: it does not).

---

### Q9: Free Energy Principle (R=1580)
**What exists:** Mathematical identity log(R) = -F + const. FEP connection is notational relabeling. Empirical R-F correlation is only -0.23.
**Fix needed:** Either derive R from FEP using proper recognition density and generative model, or drop the connection entirely.
**Honest achievable claim (if fixed):** Whether the R formula can be given a variational inference interpretation with genuine predictive content.

---

### Q11: Valley Blindness (R=1540)
**What exists:** Philosophical taxonomy of failure modes (valid but repackaging known concepts). R formula never computed in any of 12 tests.
**Fix needed:** Actually compute R in the test scenarios. Show R exhibits the predicted failure modes (valley blindness, plateau traps).
**Honest achievable claim (if fixed):** A taxonomy of scenarios where consensus-based measures fail, with empirical demonstration.

---

### Q14: Category Theory (R=1480)
**What exists:** Code implementing categorical concepts. "Presheaf" is not a presheaf. "Sheaf" satisfaction is 97.6% (not a sheaf -- universal quantifier required). R-cover is NOT a Grothendieck topology (genuinely useful negative finding).
**Fix needed:** Either prove the categorical structures rigorously or reframe as "inspired by" rather than "is."
**Honest achievable claim (if fixed):** R-cover fails to form a Grothendieck topology (honest negative result). Categorical language is decorative unless formal proofs are provided.

---

### Q15: Bayesian Inference (R=1460)
**What exists:** Core "discovery" (R = 1/std when E=1) is a tautology. Earlier falsification was more rigorous than the "rescue."
**Fix needed:** Let the original falsification stand. If restarting, use the GLOSSARY E definition throughout.
**Honest achievable claim (if fixed):** Whether R has a non-trivial Bayesian interpretation when E is not hardcoded to 1.

---

### Q17: Governance Gating (R=1420)
**What exists:** 1617-line implementation guide. 8/8 tests that verify arithmetic properties, not governance effectiveness. Zero false positive/negative data.
**Fix needed:** Test on real governance scenarios with ground truth. Compute precision/recall/ROC at multiple thresholds. Compare to baselines.
**Honest achievable claim (if fixed):** Whether R-based gating improves governance decisions compared to simpler alternatives.

---

### Q20: Tautology Risk (R=1360)
**What exists:** Best-methodology framework (pre-registered predictions, negative controls). But addresses the WRONG tautology (8e universality instead of R=E/sigma).
**Fix needed:** Apply the same rigorous methodology to the actual tautology question: Is R = E/sigma merely restating signal-to-noise ratio?
**Honest achievable claim (if fixed):** Whether the R formula adds predictive power beyond the trivial observation that SNR = signal/noise.

---

### Q21: Rate of Change / dR/dt (R=1340)
**What exists:** Alpha-drift detection code. AUC=0.9955 on synthetic classification task. dR/dt itself has AUC=0.10 (worse than random).
**Fix needed:** Use real temporal data (not synthetic noise injection). Drop Riemann/8e interpretation. Compare alpha to simpler eigenspectrum statistics.
**Honest achievable claim (if fixed):** Whether eigenspectrum statistics (alpha, entropy, Df) change before aggregate quality metrics during real system degradation.

---

### Q27: Hysteresis (R=1220)
**What exists:** Noise injection experiment on geometric memory gate. r=+0.714 on live system.
**Fix needed:** The answer is NOT hysteresis (document admits this). Rename to "selection bias under noise" and test whether the effect is useful in practice.
**Honest achievable claim (if fixed):** Adding noise to a threshold-based gate increases the mean quality of accepted items (selection bias). This is statistics 101, not a discovery about intelligence.

---

### Q40: Quantum Error Correction (R=1420)
**What exists:** Alpha drift detection (legitimate empirical finding). "Holographic" distribution (real but is standard PCA redundancy). QECC framework inapplicable.
**Fix needed:** Strip quantum framing. Rename to "Structured Embeddings Resist Noise." Test alpha drift as a practical corruption detector against real baselines.
**Honest achievable claim (if fixed):** Eigenspectrum statistics (alpha) detect corruption in structured embeddings. This is classical signal detection, not quantum error correction.

---

### Q43: Quantum Geometric Tensor (R=1530)
**What exists:** Code computing np.cov(centered.T) -- standard covariance matrix, not QGT. Berry curvature correctly proven to be zero for real vectors. 96% alignment between "QGT" and MDS eigenvectors is SVD theorem.
**Fix needed:** Strip quantum labels. Rename to "Covariance Matrix Analysis." The linear algebra is correct; the labeling is wrong.
**Honest achievable claim (if fixed):** Embedding covariance matrices capture meaningful structure. Berry curvature is zero for real-valued embeddings (correctly demonstrated negative result).

---

## TIER 4: Dead (23 Qs)

These are either built on falsified foundations, contain no real work, are unfalsifiable, or are pure numerology. Nothing real to build on.

---

### AX: Axiom Foundation
**What exists:** 10 axioms. Axiom 5 IS the formula (circular). 5/10 axioms have no formal representation. Bait-and-switch between Axioms 0-9 and Prop 3.1's (i)-(iv).
**Why dead:** The axiom system is not a foundation -- it is a post-hoc rationalization. The formula came first; the axioms were written to justify it. Cannot be salvaged without rebuilding from scratch, which would mean abandoning the current axiom system entirely.

---

### Q4: Novel Predictions (R=1700)
**What exists:** 4 "novel predictions" that are restatements of basic statistics (CLT, selection effects). Test R formula bears no resemblance to GLOSSARY R. Only real-data test (THINGS-EEG brain-stimulus): NO SIGNIFICANT CORRELATION (max |r|=0.109, p=0.266).
**Why dead:** The only real-data test failed. The "predictions" are not novel. There is nothing to salvage because the predictions do not predict anything.

---

### Q6: IIT Connection (R=1650)
**What exists:** Loose verbal analogy between R and IIT quantities. No formal mapping. 4th incompatible E definition (1/(1+error)). "Consensus Filter Discovery" is restating R's definition.
**Why dead:** No formal connection to IIT exists. The analogy is too loose to test. Uses a different E than the rest of the project.

---

### Q8: Topology Classification (R=1600)
**What exists:** NO persistent homology computed. Kahler structure test FAILED. Master results show 3/4 tests FALSIFIED but lab notes claim 5/5 PASS (tests replaced post-hoc). c_1 "computation" is circular.
**Why dead:** Counter-evidence was suppressed. The honest results show FALSIFICATION. No topological computation was actually performed.

---

### Q13: The 36x Ratio (R=1500)
**What exists:** "Blind prediction" reimplements the measurement function (identity check). Cross-domain "universality" hardcodes the same formula everywhere. E_MIN clamping determines the ratio. Three tests redesigned post-hoc.
**Why dead:** The 36x ratio is an artifact of the E_MIN clamping parameter, not a discovery. The "blind prediction" is circular.

---

### Q18: Intermediate Scales / Deception Detection (R=1400)
**What exists:** Massive investigation (10+ reports). Red team found 3/5 falsified, 1/5 partially falsified, 1/5 robust. Protein folding r=0.749 is overfit (trained on test data). 8e at 50D is parameter-tuned (random data fits better). Delta-R is 3-6x WORSE than SIFT/PolyPhen.
**Why dead:** The project's own red team demolished the results. The one surviving finding (cross-species r=0.828) demonstrates known biology, not R's capability.

---

### Q25: What Determines Sigma? (R=1260)
**What exists:** sigma = e^(-4/pi) is post-hoc selection from 7 candidates. Real data: sigma varies 15x across domains (1.92 to 39.44). R^2_cv = 0.0 on real data.
**Why dead:** FALSIFIED. Sigma is not universal. The "derivation" is textbook numerology. Real data decisively contradicts the claim.

---

### Q29: Numerical Stability (R=1180)
**What exists:** epsilon = 1e-6 floor for division by zero. Solves the TRIVIAL problem only.
**Why dead (as currently scoped):** The real stability crisis (sigma^Df overflow, where 3.7% change in sigma produces 4.4x change in output) is completely unaddressed. Edge case R values span 6 orders of magnitude and are marked "PASS." The question was scoped so narrowly it answers nothing useful. A new question about sigma^Df stability would be Tier 3.

---

### Q31: Compass Mode (R=1550)
**What exists:** Formulas on paper. Zero implementations. Zero navigation tests. "Confirmed" via Q43's tautological eigenvalue decomposition (SVD theorem). Implementation code is just nearest-neighbor search.
**Why dead:** The success criterion ("reproducible construction where argmax_a R(s,a) yields reliable navigation") is completely unmet. The authors' own "still missing" list invalidates the CONFIRMED status. Nothing was built.

---

### Q33: Conditional Entropy / Semantic Density (R=1410)
**What exists:** An acknowledged tautology. The document explicitly states: "This definition makes sigma^Df = N a tautology by construction." H(X) is incorrectly equated with token count. Df can be negative.
**Why dead:** The "derivation" is a circular definition. Admitted by the document itself. sigma^Df = N tells you nothing you did not put in by defining sigma and Df.

---

### Q35: Markov Blankets / Spectral Gap (R=1450)
**What exists:** Zero experiments, zero data, zero test scripts. Conceptual analogy mapping sync protocol to Active Inference vocabulary. No audit reports.
**Why dead:** There is literally nothing here except a conceptual mapping paper. "R > tau defines a Markov blanket" is a definition, not a discovery. "The handshake IS Active Inference" applies equally to any request-response protocol (HTTP, TCP, DNS).

---

### Q36: Bohm Implicate/Explicate (R=1480)
**What exists:** Metaphorical connection only. SLERP conservation is mathematical tautology counted 4 times. Two contradictory test versions.
**Why dead:** No structural connection to Bohm's algebra. The "conservation" tests verify properties of SLERP interpolation, not of meaning.

---

### Q38: Noether Conservation Laws (R=1520)
**What exists:** Core claim is a tautology (geodesics conserve angular momentum by definition). Cross-architecture "validation" tests SLERP math, not embeddings. No time evolution defined.
**Why dead:** Geodesic conservation is definitional, not a discovery. There is no dynamics in the framework for Noether's theorem to apply to. Original hypothesis (scalar momentum) was honestly falsified -- let that stand.

---

### Q39: Homeostatic Regulation (R=1490)
**What exists:** Test code that creates SLERP trajectories with single-point perturbation. The "recovery" is the trajectory returning to a predetermined geodesic path. tau_relax is determined by test parameters, not physics.
**Why dead:** The "homeostasis" is entirely an artifact of test design. There is no genuine self-regulation, no feedback loop, no dynamics. The SLERP trajectory after perturbation continues on its predetermined path.

---

### Q41: Geometric Langlands (R=1500)
**What exists:** 23 tests using Langlands vocabulary for standard computational procedures. Procrustes alignment is not a derived functor. K-means clusters are not primes. S-duality score of 0.53 (barely above chance). Modularity score of 0.66 (would FALSIFY actual Langlands).
**Why dead:** None of the actual Langlands mathematical machinery appears anywhere. A 53% S-duality score is evidence AGAINST Langlands correspondence, not for it. The document's own caveat admits these are "semantic analogs" not "Langlands structure."

---

### Q44: Quantum Born Rule (R=1850)
**What exists:** Central tautology: E and P_born computed from same cosine similarities. E_squared vs P_born at r=1.000 is IDENTICAL computation. R_full gets r=0.156.
**Why dead:** FALSIFIED. The Born rule "confirmation" (r=0.999) tests the formula against its own output. The actual R formula achieves r=0.156, which is NOT_QUANTUM by the project's own criteria. The formula was abandoned and subcomponent E was promoted.

---

### Q45: Pure Geometry Navigation (R=1900)
**What exists:** Tests demonstrating known embedding arithmetic (word2vec, Mikolov et al. 2013). sigma^Df = 1.73^200 = 10^47 caused numerical explosion. R was abandoned; E tested alone.
**Why dead:** The "geometric navigation" is word2vec analogy arithmetic, published in 2013. sigma^Df overflow makes the full formula unusable. Nothing novel exists.

---

### Q46: Geometric Stability (R=1350)
**What exists:** 23-line document. No hypothesis, no experiments, no tests, no results.
**Why dead (as current work):** Nothing exists. The question itself is important and well-posed (how stable are geometric properties under perturbation?), but there is zero work to salvage. If work began, this could become Tier 2 or 3.

---

### Q47: Bloch Sphere / Holography (R=1350)
**What exists:** 23-line placeholder. No research conducted.
**Why dead:** Ill-posed. The Bloch sphere requires complex quantum amplitudes. Q42 confirms R is fundamentally local/classical. The premise is contradicted by the framework's own findings.

---

### Q48: Riemann-Spectral Bridge (R=1900)
**What exists:** Core GUE hypothesis cleanly FALSIFIED (Poisson spacing, not GUE). Semantic zeta function has NONE of Riemann zeta properties. 8e selected post-hoc from 5 candidate constants. alpha=0.5 is the most common spectral exponent in nature.
**Why dead:** The central hypothesis was falsified. The "bridge" does not exist. The only empirical finding (Df*alpha ~ 22) has no connection to the Riemann zeta function.

---

### Q49: Why 8e? (R=1880)
**What exists:** Monte Carlo test: p=0.55 (55% of random constants fit equally well). 8 from Peirce is a category error. e from max entropy contradicted by own appendix. Three "independent" paths share data. Own HONEST_FINAL_STATUS: 15% confidence, labels it "NUMEROLOGY."
**Why dead:** The project's own honest assessment says it is numerology at 15% confidence. Monte Carlo confirms. There is no "why 8e" because 8e is not special.

---

### Q50: Completing 8e (R=1920)
**What exists:** CV=6.93% across 24 non-independent models. e-per-octant is acknowledged tautological. 24 models are ~19 transformer encoders on overlapping web text.
**Why dead:** 6.93% CV is not a conservation law (physics laws hold to <10^-8). The "independent" models are not independent. The core finding (Df*alpha ~ 22) may be real but modest, and lives better in Q48 data.

---

### Q51: Complex Plane & Phase Recovery (R=1940)
**What exists:** Try3 experiments show 0/19 models pass intrinsic complex structure test on random bases. Complex structure imposed by PCA projection choice. "Berry phase" is 2D winding number (integer by topological necessity). Pinwheel test: 13.0% vs 12.5% random (chance).
**Why dead:** REFUTED. The complex structure does not exist in the data -- it is an artifact of PCA projection. 0/19 is definitive.

---

### Q54: Energy Spiral to Matter (R=1980)
**What exists:** Speculative framework citing 5 foundations, ALL of which are falsified, tautological, or circular (Q44 synthetic-only, Q51 refuted, Q9 relabeling, 8e numerology, QD tautological). Zero predictions tested. Zero external data. Highest R-score in the project.
**Why dead:** Every cited foundation has been independently demolished. The "Existing Evidence" table references items the project's own audit labeled "NUMEROLOGY" and "POST-HOC FIT." The HONEST_FINAL_STATUS.md and README.md are more honest than the main document. R=1980 for this is incoherent.

---

## One-Line Summary Table

| Q | Tier | One-Line Summary | Fix / Why Dead |
|---|------|-----------------|----------------|
| AX | 4 | Axiom 5 IS the formula; 5/10 axioms unformalized | Axiom system is post-hoc rationalization |
| Q1 | 2 | Location-scale normalization valid but all synthetic | Test E/sigma on real NLP benchmarks |
| Q2 | 3 | Falsification criteria have unfalsifiable escape hatch | Rewrite with modus tollens and thresholds |
| Q3 | 3 | "Generalization" is same computation on different distributions | Test on genuinely different domains |
| Q4 | 4 | All 4 "predictions" restate basic statistics; real-data test FAILED | Nothing novel; only real test failed |
| Q5 | 3 | Agreement=truth is assertion, not proof | Design empirical test where agreement != truth |
| Q6 | 4 | Loose verbal analogy to IIT; 4th incompatible E definition | No formal mapping exists |
| Q7 | 3 | Intensivity FAILS; two R formulas in code | Fix intensivity or accept R is not intensive |
| Q8 | 4 | No persistent homology; Kahler FAILED; counter-evidence suppressed | Honest results show falsification |
| Q9 | 3 | Free Energy identity is notational relabeling | Need genuine FEP derivation or drop connection |
| Q10 | 2 | Spectral contradiction well-designed; raw E outperforms R | Test on real alignment benchmarks |
| Q11 | 3 | Philosophical taxonomy valid; R never computed | Actually compute R in test scenarios |
| Q12 | 2 | Weight interpolation framework exists; Tests 8-9 reverse-engineered | Use real training checkpoints |
| Q13 | 4 | 36x ratio is artifact of E_MIN clamping parameter | Ratio is fake |
| Q14 | 3 | Categorical structures are not rigorous; R-cover not a topology | Either prove rigorously or reframe |
| Q15 | 3 | Core discovery (R=1/std when E=1) is tautology | Let original falsification stand |
| Q16 | 1 | Real SNLI/ANLI data; honest negative result; reproducible | Reframe as confirming known cosine similarity behavior |
| Q17 | 3 | 1617-line spec but zero performance data | Test on real governance scenarios with baselines |
| Q18 | 4 | Red team found 3/5 results falsified; protein fix is overfit | Own red team demolished results |
| Q19 | 2 | Real data; Simpson's Paradox correctly identified | More datasets with real agreement labels |
| Q20 | 3 | Good methodology but addresses wrong tautology | Apply methodology to actual tautology (R=E/sigma) |
| Q21 | 3 | Alpha-drift real but no temporal data; question was changed | Use real temporal data; compare to simple alternatives |
| Q22 | 1 | Clean falsification on 7 real domains; 3 audits | None needed; exemplary negative result |
| Q23 | 1 | sqrt(3) honestly shown to be empirical fit, not geometric | Propagate finding to other documents |
| Q24 | 2 | Real SPY data; WAIT honestly falsified | Replicate on 2+ more domains; increase n |
| Q25 | 4 | sigma varies 15x; R^2_cv=0.0 on real data; FALSIFIED | Sigma is not universal. Dead. |
| Q26 | 2 | Multi-model bootstrap; practical N>=5 guidance | Test more dimensionalities; add theory |
| Q27 | 3 | Answer is NOT hysteresis; it is selection bias | Rename and test practical utility |
| Q28 | 2 | Real market data; core non-chaos claim is valid | Use pre-registered thresholds; replicate |
| Q29 | 4 | Solves div/0 only; sigma^Df overflow unaddressed | Trivial problem solved; real problem ignored |
| Q30 | 2 | 8 methods benchmarked; 100-300x speedup | Test on real data; analytical error bounds |
| Q31 | 4 | Zero implementations; "confirmed" by tautology | Nothing was built |
| Q32 | 2 | Best methodology; real public data; NLI cross-encoder works | Test whether grad_S adds value; drop "field" label |
| Q33 | 4 | Admitted tautology (sigma^Df=N by construction) | Circular by the document's own admission |
| Q34 | 2 | Genuine observation; shared training confound | Address confound; cite existing literature |
| Q35 | 4 | Zero experiments; conceptual mapping only | Nothing here except relabeling |
| Q36 | 4 | Metaphorical connection; SLERP tautology x4 | No structural connection to Bohm |
| Q37 | 2 | Real historical/multilingual data; modest effects | Remove physics framing; cite comp. ling. literature |
| Q38 | 4 | Core claim is tautology (geodesics conserve by definition) | No dynamics for Noether to apply to |
| Q39 | 4 | "Homeostasis" is SLERP interpolation artifact | Test design creates the result |
| Q40 | 3 | Alpha drift is real but classical; QECC inapplicable | Strip quantum framing; test as corruption detector |
| Q41 | 4 | Langlands vocabulary for standard computations; 53% S-duality | No actual Langlands mathematics |
| Q42 | 1 | Correct null result; validated apparatus; R is local | Acknowledge result was trivially expected |
| Q43 | 3 | Valid linear algebra mislabeled as quantum geometry | Strip quantum labels; rename to covariance analysis |
| Q44 | 4 | FALSIFIED: r=0.156 for full R; r=0.999 is tautological | Born rule confirmation is formula tested on own output |
| Q45 | 4 | Known embedding arithmetic (word2vec 2013); sigma^Df overflows | Nothing novel; formula unusable |
| Q46 | 4 | 23-line placeholder; no work done | Nothing exists |
| Q47 | 4 | No research; premise contradicted by Q42 (R is classical) | Ill-posed for classical embeddings |
| Q48 | 4 | GUE hypothesis cleanly FALSIFIED (Poisson spacing) | Central hypothesis disproven |
| Q49 | 4 | Numerology; p=0.55 Monte Carlo; own docs say 15% confidence | 8e is not special |
| Q50 | 4 | CV=6.93% is not a conservation law; non-independent models | 6.93% variance is not physics-grade conservation |
| Q51 | 4 | REFUTED: 0/19 models on random bases | Complex structure is PCA artifact |
| Q52 | 1 | Clean falsification; sound reinterpretation as dimensionality | Minor: 2/5 hypotheses untested |
| Q53 | 1 | Four audits confirm FALSIFIED; comprehensive test battery | Update status from PARTIAL to FALSIFIED |
| Q54 | 4 | All 5 foundations demolished; zero data; highest R-score | Built on wreckage |

---

## What Is Actually Worth Keeping

If you stripped away ALL the overclaiming, decorative vocabulary, and circular reasoning, the Living Formula project contains these genuine, defensible findings:

### Validated Observations (Tier 1 core)
1. **Cosine similarity detects topical divergence but not adversarial logical contradictions** (Q16)
2. **No universal R threshold exists across domains** (Q22)
3. **sqrt(3) is empirically fitted, not geometrically derived** (Q23)
4. **R is fundamentally local/classical** (Q42)
5. **R (participation ratio) measures effective dimensionality, not predictability** (Q52)
6. **No pentagonal or phi geometry in embedding spaces** (Q53)

### Promising Directions (Tier 2 core -- pending real data)
1. **E/sigma as a signal-to-noise measure** for embedding agreement (Q1, Q32)
2. **Eigenspectrum statistics as embedding health monitors** (Q10, Q12, Q21, Q40)
3. **NLI cross-encoder + cosine similarity** for misinformation scoring (Q32)
4. **Cross-model spectral convergence** as a real NLP phenomenon (Q34, Q37)
5. **Practical engineering: N >= 5 for stable R, sampling for 100x speedup** (Q26, Q30)

### The Core Insight Worth Preserving
Cosine similarity normalized by local variance -- i.e., E/sigma, the simplest version of R without sigma^Df -- captures something meaningful about semantic agreement. This is not revolutionary (it is SNR applied to embeddings), but it is a useful metric. The sigma^Df term causes more problems than it solves (overflow, instability, numerology). The honest version of the formula is:

```
R_simple = E / sigma
```

where E = cosine similarity and sigma = standard deviation of observations.

Everything else -- the quantum interpretation, the 8e conservation law, the complex plane, the Langlands/Bohm/Noether connections, the "field" metaphor -- is decorative vocabulary that adds no predictive power.

---

*Salvageability triage completed: 2026-02-05*
*Based on 34 verdict files from 6-phase adversarial verification with 54 dedicated subagents.*
*Standard applied: if there is nothing real to build on, it is Tier 4.*
