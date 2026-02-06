# Q25: Sigma Is Derivable from First Principles

## Hypothesis
The noise floor parameter sigma in R = (E / grad_S) * sigma^Df is predictable from dataset properties (R^2 > 0.7) and can be derived from first principles rather than being an irreducibly empirical quantity.

## v1 Evidence Summary
- Cross-validated R^2 = 0.8617 on 22 synthetic datasets, exceeding the 0.7 threshold.
- Predictive formula: log(sigma) = 3.456 + 0.940 * log(mean_dist) - 0.087 * log(effective_dim) - 0.021 * eigenvalue_ratio.
- Mean pairwise distance dominated the prediction (exponent ~0.94).
- Sigma ranged from 4.15 to 100.0 across synthetic datasets (24x range).
- Domain-specific sigma patterns: NLP ~40, Market ~54, Image ~65, Graph ~26.
- A separate derivation proposed sigma = e^(-4/pi) = 0.2805 as a universal constant from solid angle geometry.

## v1 Methodology Problems
1. **Real data falsification ignored.** The deep audit and verification found R^2_cv = 0.0 on real external data (HuggingFace, NCBI GEO). The positive R^2 = 0.86 result was on synthetic data only. The status was not updated from "CONFIRMED" despite this falsification.
2. **Definitional confusion.** The GLOSSARY defines sigma in (0, 1) with empirical value ~0.27, but Q25 tests found sigma from 4.15 to 100.0. These are likely different quantities sharing a name -- the GLOSSARY sigma is a dimensionless noise floor; Q25 sigma is a kernel bandwidth parameter.
3. **Circular optimization.** "Optimal sigma" was defined as the value minimizing bootstrap CV of R, but R is a direct function of sigma. The regression may recover the optimization criterion rather than a fundamental relationship.
4. **Contradictory claims.** If sigma is a universal constant (~0.27, Conjecture 4.1), it should not depend on dataset properties. If sigma is predictable from dataset properties (Q25 hypothesis), it is not a universal constant. Both cannot be true.
5. **sigma = e^(-4/pi) derivation was post-hoc.** Seven candidate formulas were tried, and the best-fitting one was selected. The "solid angle geometry" proof sketch changed its own methodology mid-argument (steps 1-6 abandoned at step 7) to arrive at a predetermined target.
6. **Exponential sensitivity.** A 3.9% error in sigma amplifies to order-of-magnitude errors in sigma^Df at typical Df values (~43), making the distinction between candidate values meaningless in practice.
7. **Resolution test was spurious.** R^2_cv = 0.99 was achieved by the model learning a trivial mapping: 384-dim embeddings -> sigma ~2.7, 12-dim embeddings -> sigma ~9.7. It predicted embedding architecture, not sigma principles.

## v2 Test Plan

### Test 1: Sigma Definition Audit
Establish a single, unambiguous operational definition of sigma. Determine whether GLOSSARY sigma (noise floor in (0,1)) and Q25 sigma (kernel bandwidth ranging 4-100) are the same or different quantities. Document exact computation procedure.

### Test 2: Sigma Predictability on External Data
- Compute sigma (using the fixed operational definition) on 50+ datasets from diverse domains.
- Fit the predictive regression on a training split of datasets (not data points within datasets).
- Evaluate R^2_cv using leave-one-domain-out cross-validation (train on NLP/market/graph, predict image, etc.).
- Report confidence intervals on R^2.

### Test 3: Sigma Universality Test
- If sigma is claimed to be universal (~0.27), measure it independently on 20+ external datasets across 5+ domains.
- Report mean, standard deviation, and the full distribution of observed sigma values.
- Test H0: sigma is drawn from a distribution with CV < 10% (universal) vs H1: CV > 30% (domain-dependent).

### Test 4: First-Principles Derivation Validation
- If a derivation produces a specific predicted value (e.g., e^(-4/pi) = 0.2805), pre-register this prediction before any measurement.
- Measure sigma on 30+ held-out datasets never used in the derivation process.
- Compare observed values to the prediction with proper confidence intervals.
- Do NOT search for matching mathematical expressions after observing the value.

### Test 5: Sensitivity Analysis
- Compute R = (E/grad_S) * sigma^Df for sigma = 0.25, 0.27, 0.2805, 0.30, 1/4, 2/7 across realistic Df values (10, 20, 40, 60).
- Report how much R changes for each candidate sigma value.
- Determine minimum precision required for sigma to be useful in the formula.

## Required Data
- **GloVe embeddings** (6B tokens, 300d) -- Stanford NLP
- **Word2Vec** (Google News, 300d) -- pre-trained
- **BERT/RoBERTa** embeddings extracted from HuggingFace models
- **Sentence-transformers** (MiniLM, MPNet) from HuggingFace
- **NCBI GEO** gene expression datasets (accessions from GEO DataSets)
- **Financial time series** from Yahoo Finance or FRED
- **Image embeddings** from CLIP or ResNet on ImageNet validation set
- **Audio embeddings** from Whisper or wav2vec on LibriSpeech
- At least 50 datasets spanning 6+ domains

## Pre-Registered Criteria
- **Success (predictability):** R^2_cv > 0.5 on leave-one-domain-out cross-validation using external data only.
- **Success (universality):** CV of sigma < 15% across all domains, measured on 50+ external datasets.
- **Failure (predictability):** R^2_cv < 0.3 on external data, indicating sigma is not reliably predictable.
- **Failure (universality):** CV of sigma > 30% across domains, indicating sigma is domain-specific.
- **Inconclusive:** R^2_cv between 0.3 and 0.5, or CV between 15% and 30%.

## Baseline Comparisons
- **Null model for predictability:** Predict sigma = grand mean for all datasets. R^2 = 0.
- **Null model for universality:** Sigma is a free parameter per dataset (no constraint).
- **Alternative model:** Sigma = f(embedding_dimension) -- a trivial predictor that the v1 "resolution" test may have been measuring.

## Salvageable from v1
- The pre-registration structure (hypothesis + falsification criteria stated upfront) is good methodology.
- test_q25_real_data.py attempted external validation and correctly falsified the synthetic-only result -- this code can be extended.
- The 22 synthetic datasets provide a useful development/debug set, though they must not be used for final evaluation.
- The observation that mean pairwise distance is the dominant predictor is a reasonable starting hypothesis, even if unvalidated on external data.
