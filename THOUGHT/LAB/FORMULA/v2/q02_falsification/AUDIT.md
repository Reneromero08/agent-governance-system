# Q02 Adversarial Audit Report

**Auditor:** Opus 4.6 (adversarial mode)
**Date:** 2026-02-06
**Files reviewed:** formula.py, test_v2_q02_fixed.py, test_v2_q02_fixed_results.json, VERDICT_v2.md

---

## 1. Code Bugs Found

### BUG-1: R4 is mathematically identical to R0 for Spearman correlation (MAJOR)

`R4 = log(E) - log(grad_S) = log(E/grad_S) = log(R0)`. Since log is a monotonic transformation, Spearman rank correlation is invariant under monotonic transforms. Therefore R4 and R0 produce **identical** Spearman rho values by mathematical necessity. The results confirm this: both show rho = 0.879 for model 1, 0.862 for model 2, and 0.858 for model 3.

This means R0 is being compared against only **4 genuine alternatives**, not 5. Every place that says "R0 beats X/5 alternatives" should say "R0 beats X/4 alternatives." The pre-registered criterion "R0 outperforms >= 4 of 5 alternatives" is impossible to satisfy because one "alternative" is mathematically guaranteed to tie. The bootstrap correctly shows R4 win_rate = 0.0 (R0 never beats R4 because they are identical), but the counting logic treats this as R0 "losing" to R4.

**Impact:** The pre-registered criterion for Test 2 is structurally unfair to R0. With 4 genuine alternatives, the criterion should be "R0 beats >= 3 of 4" or the R4 comparison should be excluded. This does not change the verdict on its own (R0 beats 0 genuine alternatives at p < 0.01 anyway), but it is methodologically unsound.

### BUG-2: grad_S test has reversed alternative hypothesis (MEDIUM)

At line 235: `stats.mannwhitneyu(gS_mixed, gS_pure, alternative='greater')`. The test asks whether gS_mixed > gS_pure. But the **observed data** shows gS_pure > gS_mixed (pure_mean=0.124, mixed_mean=0.094 for model 1). The test correctly gets p ~ 1.0 because the data contradicts the hypothesis.

This is not technically a bug -- the test is intentionally checking the theoretical prediction that pure clusters should have LOWER grad_S. When the data shows the opposite, p goes to 1.0, which triggers FAIL. The test logic is correct for testing the stated prediction. But see "Theoretical Predictions" section below for whether the prediction itself is correct.

### BUG-3: Cohen's d sign convention inconsistency (MINOR)

For the E test (line 217): `effect_d = (mean(E_pure) - mean(E_mixed)) / pooled_std` -- this produces positive d when pure > mixed.

For the grad_S test (line 237): `effect_d = (mean(gS_mixed) - mean(gS_pure)) / pooled_std` -- the subtraction order is reversed. This produces a negative d (-3.63, -2.75, -2.18) which correctly indicates the hypothesis direction was wrong. The sign is informative, but the inconsistent convention is confusing.

### BUG-4: Pooled standard deviation uses incorrect formula (MINOR)

Lines 216-217 and 236-237 compute: `pooled_std = sqrt((std_a^2 + std_b^2) / 2)`. This is the formula for equal-sized groups. With n=20 per group (pure) and n=20 (mixed), this happens to be correct. But for Test 4 where group sizes could differ, the same formula is used and would still be correct (both are n=20). Not a real bug in this case, but fragile.

### BUG-5: sigma and Df tests use only pure clusters (DESIGN ISSUE)

The sigma CV test (lines 253-263) and Df CV test (lines 265-276) compute coefficient of variation only across the 20 pure clusters (one per category). They ask: "Does sigma/Df vary across different topic categories?" This is a valid question but is a weaker test than comparing pure vs mixed clusters as done for E and grad_S. The choice of CV > 0.2 and CV > 0.1 thresholds appears arbitrary -- there is no theoretical basis cited for why these specific values should indicate "meaningful" variation.

### BUG-6: No multiple comparisons correction in Test 2 (MINOR)

The bootstrap pairwise comparisons test R0 against 5 (really 4) alternatives without Bonferroni or other correction. At p < 0.01 with 4 tests, the effective threshold should be p < 0.0025. This makes the criterion slightly harder to pass. Since R0 fails to beat any alternative at even p < 0.01, the correction is moot.

---

## 2. Statistical Errors

### STAT-1: n=60 clusters for Test 2 is adequate but underpowered for detecting small differences

With n=60 and rho values all in the 0.79-0.88 range, the differences between formulas are in the range of 0.01-0.04 rho units. Detecting a difference of 0.04 in Spearman rho between two correlated metrics with 80% power would require approximately n=200-400 clusters. The test has adequate power to detect large differences (rho gap > 0.10) but not the small differences actually observed. This means the bootstrap "R0 wins X% of the time" results are meaningful, but the inability to reach p < 0.01 may be partly a power issue.

### STAT-2: Modus tollens 95% CIs are extremely wide (CONCERN)

The 95% CIs for violation rates are [1.5%, 37.7%], [3.5%, 36.0%], and [1.5%, 35.4%]. These are so wide that they are almost uninformative -- the true violation rate could be anywhere from 2% to 36%. With only n=11, 16, and 12 clusters exceeding the threshold, the test has very low precision. The verdict says "passes modus tollens" but the CIs are consistent with violation rates that would be well above the 10% threshold.

The n_high_R values (11, 16, 12) are far fewer than the claimed "n >> 6" fix from v1. While technically larger than 6, they do not provide the statistical precision implied by the methodology description.

### STAT-3: Q_min = 1.0 is suspicious (CONCERN)

All three models calibrate Q_min (10th percentile of purity among high-R training clusters) to exactly 1.0. This means the modus tollens test is checking: "among clusters with R > T, do any have purity < 1.0?" A purity of 1.0 means every document in the cluster comes from a single category. This is an extremely lenient criterion -- any cluster that is not perfectly pure triggers a "violation," even one with 99% purity. This inflates the violation count and makes the test harder to pass, but in a somewhat arbitrary way.

### STAT-4: Train/test split is sound but document reuse across tests is not addressed

Tests 1, 2, 3, and 4 all draw from the same 8,000 subsampled documents. Tests 1 and 4 use the same random seeds and get the same "pure clusters" (same 20 categories x 100 docs). Test 3 uses a separate train/test split but the documents overlap with Tests 1/2/4. This is not data leakage in the traditional sense (Test 3 has its own split), but the tests are not fully independent. Failure on one test given a specific dataset feature (e.g., 20 Newsgroups category structure) would likely imply failure on others.

### STAT-5: fast_spearman uses formula that assumes no ties

Line 424: `1 - 6 * sum(d^2) / (n * (n^2 - 1))`. This is the Spearman formula WITHOUT tie correction. Since `scipy.stats.rankdata` handles ties with average ranks, the formula can produce slightly inaccurate rho values when ties exist. For continuous data (cosine similarities), ties are unlikely, so this is a minor concern. But for purity values (discrete, many clusters may share purity = 1.0), ties are common and the formula will be slightly off. The point Spearman values are computed correctly using scipy; only the bootstrap inner loop uses this approximation.

---

## 3. Methodological Issues

### METHOD-1: The theoretical predictions for grad_S are WRONG (CRITICAL)

This is the single most important finding of this audit.

The test assumes: "Pure clusters should have LOWER grad_S (more uniform similarities)." But there is no derivation from the formula's theory that establishes this. Let me reason about what actually happens:

**In a pure cluster** (all docs from one topic): Some document pairs within the topic are very similar (e.g., two posts about the same subtopic), while others are moderately similar (same broad topic but different subtopics). This creates a **wide spread** of pairwise similarities, hence HIGH grad_S.

**In a random cluster** (docs from many topics): All pairwise similarities converge toward a narrow band of low values (near zero). This is because random pairs from different topics are approximately uniformly dissimilar. The central limit theorem ensures low variance when averaging over many topic-pairs. Hence LOW grad_S.

This is exactly what the data shows: grad_S pure > grad_S mixed consistently across all 3 models. **The data is behaving correctly -- the theoretical prediction was wrong.**

Now here is the critical insight for the formula R = E/grad_S: When pure clusters have high E AND high grad_S, the ratio E/grad_S is high E / high grad_S. For random clusters, it is low E / low grad_S. The formula works because E increases more than grad_S does. The grad_S denominator is NOT acting as a "noise correction" that decreases for good clusters. Instead, it acts as a **normalizer** that partially controls for the overall scale of similarities. This is still a valid role -- just different from what the test assumed.

**Impact on verdict:** The grad_S "failure" is a failure of the TEST's prediction, not a failure of the component. grad_S IS varying between pure and mixed clusters with massive effect sizes (d = 2.2-3.6). It just varies in the "wrong" direction relative to an unjustified prediction. If the test had predicted grad_S_pure > grad_S_mixed (and used `alternative='less'` or equivalently tested in the other direction), grad_S would have PASSED with flying colors (p ~ 1e-8, d > 2).

### METHOD-2: sigma and Df "nearly constant" may be EXPECTED behavior (SIGNIFICANT)

The test checks whether sigma and Df vary across the 20 pure clusters (different topic categories). But consider what sigma and Df measure:

- **sigma** = participation ratio / ambient dimensionality. For 100 documents in 384-dim space, the eigenvalue structure is primarily determined by the model architecture and training, not by the specific topic. All MiniLM models produce embeddings that occupy similar effective dimensionality regardless of topic. Sigma being constant is a property of the embedding model, not a failure of the formula.

- **Df** = 2/alpha where alpha is the power-law exponent of eigenvalue decay. For neural network embeddings, the eigenvalue spectrum follows a characteristic decay that is largely model-dependent, not content-dependent. Df ~ 0.136 across all topics and models is expected for transformer sentence encoders. Again, this is a model property.

The test assumes sigma and Df should vary across categories. But the formula may intend sigma^Df to vary across **different types of data** (different embedding models, different domains, different modalities), not across categories within a single model on a single dataset. The test's CV criterion (sigma CV > 0.2, Df CV > 0.1) is testing the wrong axis of variation.

To properly test sigma and Df, one should:
1. Vary the embedding model architecture (done -- 3 models, but CV is computed per-model)
2. Vary the data domain (not done -- only 20 Newsgroups)
3. Vary cluster size (not done -- all clusters are 100 docs)
4. Compare pure vs mixed clusters for sigma and Df (not done -- only CV across pure clusters)

### METHOD-3: "Pure vs mixed" conflates two things

The test constructs "pure" clusters (single category) and "mixed" clusters (random documents or multi-category). But "cluster quality" in real-world applications is not binary. The formula is presumably intended to measure something about the geometric structure of embedding clusters, not just whether a cluster happens to contain a single newsgroup category.

Two newsgroup categories can be very similar (e.g., `rec.sport.baseball` and `rec.sport.hockey`) or very different (e.g., `rec.sport.baseball` and `sci.crypt`). A "mixed" cluster from two similar categories might have better coherence than a "pure" cluster from a diverse category. The test's purity metric (fraction from dominant category) does not capture semantic coherence -- it captures label homogeneity.

### METHOD-4: 20 Newsgroups is a reasonable but limited test domain

20 Newsgroups is a classic text classification dataset with well-separated categories. This is a favorable test case for any coherence metric. The findings may not generalize to:
- Domains with overlapping categories
- Non-text modalities (images, code, etc.)
- Different embedding models (e.g., larger models, different architectures)
- Different scales (cluster sizes 10-10000)

This is acknowledged implicitly but not flagged as a limitation of the falsification claim.

### METHOD-5: Three architectures are not diverse enough

All three models are MiniLM variants with 384 dimensions:
- all-MiniLM-L6-v2 (6-layer, general)
- multi-qa-MiniLM-L6-cos-v1 (6-layer, QA-tuned)
- paraphrase-MiniLM-L3-v2 (3-layer, paraphrase-tuned)

These share the same base architecture (MiniLM) and output dimensionality (384). A proper robustness check should include:
- Different architectures (BERT, RoBERTa, MPNet)
- Different dimensionalities (768, 1024)
- Non-transformer models (TF-IDF + SVD, GloVe averages)

The finding that all three models show nearly identical sigma and Df values is likely because they share architecture, not because sigma and Df are fundamentally constant.

---

## 4. Verdict Assessment

### Original verdict: FALSIFIED

### Audit verdict: **CHANGE TO INCONCLUSIVE**

### Detailed reasoning:

The FALSIFIED verdict rests on a single trigger: ">= 3 components fail (3/4 failed)." But this audit finds that the component-level test criteria contain fundamental errors:

**1. grad_S "failure" is a test error, not a formula error.**

The test predicted grad_S should be lower for pure clusters. The data shows the opposite. But the test's prediction has no theoretical derivation -- it is an assumption the test author made. The actual behavior (grad_S higher for pure clusters) is physically reasonable and does not mean grad_S "fails." It means the test's expectation was wrong.

If we correct the prediction (grad_S_pure > grad_S_mixed), grad_S passes with d > 2.0 and p < 1e-6 across all models. This is a massive, highly significant effect -- grad_S strongly discriminates pure from mixed clusters, just in the other direction.

With corrected grad_S: E passes, grad_S passes (corrected direction), sigma fails, Df fails. That is 2/4 components passing, which triggers NEITHER the confirm criterion (4/4) NOR the falsify criterion (>= 3 fail). This lands in the INCONCLUSIVE zone.

**2. sigma and Df "failures" may test the wrong axis of variation.**

The CV criteria test whether sigma and Df vary across newsgroup categories within a single embedding model. The formula may intend these components to capture cross-model or cross-domain variation. The test does not provide evidence that sigma and Df fail to vary when the embedding model or data domain changes -- only that they do not vary across topics within one model.

**3. The formula DOES work in practice.**

All other tests support the formula's utility:
- Modus tollens: avg violation rate 10.0%, 2/3 models pass
- Adversarial: massive effect sizes (d > 4.5) on all models
- Functional form: R0 ranks 1st or 2nd on all models
- Spearman rho with purity: 0.83-0.90 on held-out data

A formula that achieves rho = 0.88 with its target variable and d > 4.5 in discrimination tests should not be labeled "falsified" based solely on component predictions that the test author may have gotten wrong.

**4. However, the formula is NOT confirmed either.**

sigma and Df genuinely contribute nothing in this test setting. R_full and R_simple perform nearly identically. The formula is over-parameterized: two of its four components are effectively constant multipliers. And the grad_S denominator, while discriminative, works opposite to its stated theoretical interpretation. The formula predicts correctly for potentially wrong reasons.

---

## 5. Issues Requiring Resolution

### ISSUE-1 (BLOCKING): Derive theoretical predictions from the formula's actual theory

The test assumes four predictions about component behavior but cites no theoretical derivation. The critical question: **where do these predictions come from?** If there is a paper or derivation that says "grad_S should decrease with cluster quality," that derivation is wrong and the formula's theory needs revision. If the predictions are just the test author's intuitions, then the test is checking intuitions, not falsifying the formula.

**Action needed:** Either provide the theoretical derivation for each prediction, or replace the component-level test with one that tests what the formula actually claims.

### ISSUE-2 (BLOCKING): Re-analyze grad_S with corrected prediction

The verdict would change from FALSIFIED to INCONCLUSIVE if grad_S is counted as passing (with reversed direction). This needs explicit adjudication: Is the formula's theory committed to grad_S being a "noise" measure (should be lower for good clusters), or is grad_S simply a normalizer that can go either direction?

### ISSUE-3 (IMPORTANT): Test sigma and Df across models and domains, not just across categories

The current test asks the wrong question about sigma and Df. The right test would:
- Compute sigma and Df for the same clusters across all 3 models
- Compute sigma and Df on different datasets (Wikipedia, scientific papers, code, etc.)
- Check if sigma^Df varies when these axes change

### ISSUE-4 (IMPORTANT): Restate the pre-registered criterion for Test 2 to exclude R4

R4 = log(R0) and is mathematically identical for Spearman. The test should compare R0 against 4 genuine alternatives, not 5.

### ISSUE-5 (MINOR): Increase n for modus tollens

n_high_R = 11-16 gives CIs up to 37%. With n=50-100 clusters exceeding threshold, the CIs would narrow enough to be informative.

---

## 6. What Would Change the Verdict

### To CONFIRMED:
1. Provide a theoretical derivation showing grad_S SHOULD increase for coherent clusters (making the observed direction the correct prediction), AND
2. Show sigma and Df vary meaningfully across a broader set of models/domains (not just 20 Newsgroups categories), AND
3. Demonstrate R0 beats E alone with statistical significance (p < 0.01, n > 200 clusters)

### To FALSIFIED (upheld):
1. Provide a clear theoretical derivation that grad_S SHOULD decrease for coherent clusters, confirming the test's original prediction is correct, AND
2. Show that the reversed direction truly means the formula's mechanism is broken (not just a different-from-expected but still valid mechanism)

### To INCONCLUSIVE (recommended):
1. Accept that the grad_S prediction was wrong and count grad_S as passing with reversed direction (2/4 components pass), OR
2. Accept that the component-level predictions are not well-enough grounded to trigger falsification

---

## Summary Table

| Category | Count | Severity |
|----------|-------|----------|
| Code bugs | 6 | 1 Major, 1 Medium, 3 Minor, 1 Design |
| Statistical errors | 5 | 2 Concerns, 3 Minor |
| Methodological issues | 5 | 1 Critical, 1 Significant, 3 Moderate |
| Verdict | CHANGE TO INCONCLUSIVE | |
| Blocking issues | 2 | Must resolve before final verdict |

**The single most important finding:** The grad_S "failure" (METHOD-1) is almost certainly a test error, not a formula error. The test predicted the wrong direction for grad_S, and this wrong prediction accounts for 1 of the 3 component failures that triggered the FALSIFIED verdict. Correcting this prediction changes the count to 2/4 components failing, which is INCONCLUSIVE under the pre-registered criteria.

The formula remains over-parameterized (sigma and Df contribute nothing), and the theoretical interpretation of grad_S needs revision, but "falsified" overstates what the evidence shows. The formula works well in practice (rho = 0.88, d > 4.5) and the core ratio E/grad_S is the best or near-best predictor tested.
