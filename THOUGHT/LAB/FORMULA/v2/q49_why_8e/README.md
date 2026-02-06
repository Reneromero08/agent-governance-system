# Q49: Df * alpha = 8e for a Derivable Reason

## Hypothesis
The product of participation ratio (Df) and eigenvalue decay exponent (alpha) across trained embedding models is a universal constant equal to 8e = 21.746, arising from the 2^3 = 8 binary semiotic octants of Peirce's three irreducible categories, each contributing Euler's number e in natural information units. This conservation law is derivable from first principles through three independent paths: topological (Chern number), information-theoretic (Shannon capacity per octant), and thermodynamic (maximum entropy).

## v1 Evidence Summary
- Df * alpha measured across 6 models: mean = 21.84, CV = 2.69%. MiniLM matched 8e to 0.15%.
- Across 24 models: mean = 21.57, CV = 6.93%. Mean error vs. 8e: 0.82% (range: 0.03% to 23.15%).
- Random matrices produced Df * alpha ~ 14.5 (different from trained models at ~21.8).
- All 8 PCA octants populated (chi-squared p = 0.023 for non-uniformity).
- Falsification battery: Random matrix test passed (different product). Permutation test passed (p < 0.001). Vocabulary independence: CV = 1.66%.
- Monte Carlo falsification: 2,749 of 5,000 random constants matched as well or better; p = 0.5498 (FAILED to show 8e is special).
- GloVe models deviated by 4-5% while transformer models clustered at 1-2%.

## v1 Methodology Problems
1. **Monte Carlo falsification FAILED.** 55% of random constants fit equally well or better than 8e. This directly undermines the claim that 8e is special. The result was deferred as "needs reinterpretation" instead of accepted as falsification.
2. **8e is not distinguishable from alternatives.** 7*pi = 21.991, 22 (integer), and 4*pi*sqrt(3) = 21.77 all fit comparably. The 8E_VS_7PI_COMPARISON.md showed no single constant dominates across all datasets and metrics. T-tests cannot reject any candidate (p = 0.320 for 8e, p = 0.488 for 7pi, p = 0.451 for 22).
3. **Cherry-picked model subset.** Q49 reports CV = 2.69% on 6 models; the full 24-model set has CV = 6.93%, nearly tripling the variation. The 6-model figure is prominently displayed.
4. **Peirce mapping is a category error.** Peirce's Reduction Thesis is about relational arity (logic of predicates), not spatial dimensionality (PCA components). The 10-class sign taxonomy is ignored. The PC-to-category mapping (PC1=Secondness, PC2=Firstness, PC3=Thirdness) actually uses Osgood's 1957 semantic differential dimensions, not Peirce's categories.
5. **Binary discretization is arbitrary.** Treating each PC as binary (+/-) yields 2^3 = 8. Ternary (+/0/-) yields 3^3 = 27. Continuous yields infinite states. The choice of binary is what produces 8, not a derivation.
6. **e derivation is self-contradicted.** The derivation document's own Appendix B.3 states "the truly universal quantity is 8 nats," not 8e. The factor e enters by exponentiating an entropy value, which is a unit conversion, not a derivation.
7. **Three "independent" paths share data.** All three derivation paths use the same eigenvalue spectrum, the same PCA decomposition, and the same CP^n geometric assumption. The framework's own HONEST_FINAL_STATUS.md acknowledges: "These are NOT independent derivations."
8. **Non-independent model samples.** Of 24 models, ~19 are transformer encoders trained on overlapping English web text. The effective number of independent observations is closer to 3-5.
9. **Internal alpha inconsistency.** GLOSSARY defines alpha = 1/(2*Df) for CP^n, which would give Df*alpha = 0.5 always. The measured Df*alpha = 21.75. Both appear simultaneously; they are incompatible.

## v2 Test Plan

### Test 1: Proper Monte Carlo with Structured Null
- Generate 10,000 structure-preserving random matrices (matching the rank, condition number, and approximate spectral shape of real embedding covariance matrices, but with randomized eigenvectors).
- Compute Df * alpha for each.
- Determine the null distribution of Df * alpha.
- Compute p-value: what fraction of null matrices produce Df * alpha within 3%, 5%, 10% of the observed mean?
- Pre-register 8e, 7*pi, and 22 as the three candidate constants.

### Test 2: Vocabulary Size Sweep
- Measure Df * alpha at vocabulary sizes N = 50, 100, 200, 500, 1000, 2000, 5000 on a single model (e.g., BERT-base).
- Use 10 random vocabulary subsets at each size for error bars.
- Test whether Df * alpha is N-independent (true constant) or N-dependent (artifact).
- Fit Df * alpha = f(N) and extrapolate to N -> infinity.

### Test 3: Genuinely Independent Architectures
- Measure Df * alpha on at least 10 models from genuinely different families:
  - Count-based: PPMI matrix, LSA/SVD
  - Prediction: Word2Vec (skip-gram), Word2Vec (CBOW)
  - Factorization: GloVe (50d, 100d, 300d)
  - Transformer encoder: BERT, RoBERTa
  - Transformer decoder: GPT-2
  - Contrastive: SimCSE, CLIP text encoder
  - Non-English: Independent Chinese word2vec, Arabic FastText
- Each model tested on 3+ vocabulary subsets.
- Report per-family means with confidence intervals.

### Test 4: Bayesian Model Comparison
- Collect 100+ independent (model, vocabulary) measurements of Df * alpha.
- Fit each candidate model: (a) Df * alpha = 8e, (b) Df * alpha = 7*pi, (c) Df * alpha = 22, (d) Df * alpha = free parameter mu.
- Compute Bayes factors comparing each fixed-constant model to the free-parameter model.
- If BF < 3 for all constants vs. free parameter, conclude no specific constant is warranted.

### Test 5: Octant Structure Analysis
- For each model, compute the population of all 2^k sign-pattern regions for k = 1, 2, 3, 4, 5, 6 PCs.
- Test whether octant structure (k=3) is special: does Df * alpha / 2^k approximate a recognizable constant specifically at k=3 but not at k=2 or k=4?
- If Df * alpha / 2^k gives "nice" values for multiple k, the octant decomposition is not special.

### Test 6: Training Dynamics
- Train a small transformer from scratch, saving checkpoints every 1000 steps.
- Track Df * alpha throughout training.
- Determine when (if ever) Df * alpha converges to ~22, and whether it converges from above or below.
- Compare to random initialization baseline trajectory.

## Required Data
- All embedding models listed in Test 3 above (publicly available from HuggingFace, Stanford NLP, Google)
- Standard frequency-ranked vocabulary lists at multiple sizes
- Non-English embedding models trained independently (not translated/aligned)
- Computing resources for training a small transformer from scratch (~100M parameters)

## Pre-Registered Criteria
- **Success (conservation law):** Df * alpha is N-independent (regression slope not significantly different from 0, p > 0.1) AND CV < 5% across all genuinely independent architectures AND Bayes factor > 10 for one specific constant over the free-parameter model.
- **Failure (conservation law):** Df * alpha depends on N (regression slope p < 0.01) OR CV > 15% across architectures OR Bayes factor < 3 for all constants (no specific value is special).
- **Success (8e specifically):** Bayes factor > 10 for 8e over 7*pi AND over 22.
- **Failure (8e specifically):** Bayes factor < 3 for 8e vs. 7*pi or 8e vs. 22. In this case, report "Df * alpha ~ 22 (approximate)" as the honest conclusion.
- **Inconclusive:** CV between 5% and 15%, or Bayes factor between 3 and 10.

## Baseline Comparisons
- **Random matrix null:** Structure-preserving randomized matrices (eigenvector randomization preserving eigenvalue distribution).
- **i.i.d. Gaussian null:** Standard random matrices for comparison to Marchenko-Pastur predictions.
- **Architecture-family null:** Within-family CV vs. between-family CV to test whether "universality" is actually "shared architecture."
- **Alternative constants:** 7*pi, 22, 4*pi*sqrt(3), and the free-parameter mean as competitors to 8e.

## Salvageable from v1
- Df and alpha computation code (participation ratio and power-law fitting) is correct and reusable.
- The falsification battery structure (random matrix baseline, permutation test, vocabulary independence) is good methodology, though the null models need improvement.
- The 8E_VS_7PI_COMPARISON.md analysis is intellectually honest and its methodology (multi-metric comparison) should be adopted as standard.
- The observation that instruction-tuning compresses Df * alpha is genuinely interesting and worth investigating independently.
- HONEST_FINAL_STATUS.md ratings (15% confidence, "NUMEROLOGY") should be the starting prior for v2.
