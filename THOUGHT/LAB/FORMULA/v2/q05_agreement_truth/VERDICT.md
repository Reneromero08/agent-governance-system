# Q05 Verdict

## Result: FALSIFIED

Two of three pre-registered criteria triggered falsification. The hypothesis that "high R implies proximity to objective ground truth" is not supported by this evidence. R does not track truth in the ways claimed, and R is dramatically higher for echo chambers than for genuine agreement.

---

## Agreement-Truth Correlation (Test 1)

**Spearman(R_simple, truth_proximity) = -0.6848, p = 0.029**

This is a statistically significant *negative* correlation. Higher R is associated with *lower* truth proximity on STS-B, not higher. The pre-registered confirm threshold was rho > 0.5; the actual result is rho = -0.68. This is worse than the falsification threshold (rho < 0.2) -- R is actively anti-correlated with truth.

Details by human-score bin (0-5 scale, 10 bins of width 0.5):
- R_simple ranges from 0.21 to 0.32 across bins, with no monotonic relationship to truth proximity.
- The highest R values occur in the lowest-score bins (dissimilar sentences), where truth proximity is also lowest.
- R_full shows no significant correlation either (rho = -0.32, p = 0.37).
- The alternative analysis using difference embeddings shows zero correlation (rho = 0.09, p = 0.80).

Baseline context: The raw cosine similarity from all-MiniLM-L6-v2 correlates at rho = 0.82 with human scores. The model works well. R simply does not track truth proximity.

**Criterion 1: FALSIFIED (rho = -0.68, far below the 0.2 falsification threshold)**

---

## Multi-Model Agreement (Test 2)

Two independent embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2) on 500 STS-B pairs:

- Inter-model agreement: rho = 0.978 (extremely high)
- MiniLM vs Human: rho = 0.928
- mpnet vs Human: rho = 0.931
- **Spearman(R_multi, truth_error) = 0.115, p = 0.751** (not significant)
- Spearman(model_agreement, truth_error) = 0.030, p = 0.934 (not significant)

Both models track human truth excellently (rho > 0.92). They agree with each other almost perfectly (rho = 0.978). Yet the R metric computed from their binned cluster embeddings shows no relationship to truth error whatsoever. The positive sign (rho = +0.115) means higher R is weakly associated with *more* error, though this is not significant.

The formula adds no information about truth beyond what raw cosine similarity already provides. The multi-model agreement structure (high inter-model correlation, high individual truth-tracking) does not propagate through R in a useful way.

---

## Systematic Bias Attack (Test 3)

Three bias phrases prepended to all 200 sentences:
1. "In conclusion, " -- R inflated from 0.22 to 0.89 (+305%)
2. "According to recent studies, " -- R inflated from 0.22 to 0.72 (+232%)
3. "The committee determined that " -- R inflated from 0.22 to 1.97 (+815%)

**Technically, no attack "succeeded" by our pre-registered definition** because the bias also slightly *improved* truth-tracking (rho went from 0.871 to 0.882-0.891). This is because prepending a common phrase makes all embeddings more similar, compressing the variance, which happens to preserve rank-order correlations.

However, this result is deeply concerning for a different reason: **R is massively inflated by a trivial surface manipulation that has nothing to do with genuine semantic content.** The 4-9x increase in R from prepending boilerplate text means R is primarily measuring embedding homogeneity -- how similar the vectors are in absolute terms -- not truth. Any operation that pushes embeddings closer together in the space will inflate R regardless of its relationship to ground truth.

**Criterion 2: Technically CONFIRMED (attacks did not degrade truth), but the massive R inflation from surface manipulation undermines R's interpretability.**

---

## Echo Chamber Detection (Test 4)

This is the most decisive failure.

| Cluster Type | Size=10 | Size=20 | Size=50 |
|---|---|---|---|
| Echo (noise=0.01) | **401.4** | **375.3** | **363.2** |
| Echo (noise=0.05) | **16.7** | **15.6** | **15.5** |
| Genuine agreement | 0.30 | 0.26 | 0.27 |
| Random baseline | 0.25 | 0.22 | 0.23 |

Echo chambers produce R values that are **1,000-1,300x higher** than genuine agreement at the low noise level and **50-60x higher** at the moderate noise level. The Mann-Whitney U test gives p < 7e-18 at all cluster sizes -- the separation is total, in the wrong direction.

The hypothesis claimed R should be *lower* for echo chambers and *higher* for genuine agreement. The opposite is true. R is a measure of embedding homogeneity. Near-duplicate vectors (echo chambers) produce extremely high E (high mean cosine similarity) and extremely low grad_S (low dispersion), driving R = E/grad_S to extreme values. Genuinely diverse sentences that happen to be semantically related still have significant embedding variation, producing modest R.

**Criterion 3: FALSIFIED -- R is 1,000x+ higher for echo chambers than genuine agreement at all tested sizes.**

---

## Data

- **Dataset:** STS-B (Semantic Textual Similarity Benchmark) test split, 1,379 sentence pairs
- **Ground truth:** Human similarity judgments on 0-5 scale
- **Models:** all-MiniLM-L6-v2 (384-dim), all-mpnet-base-v2 (768-dim)
- **Sample sizes:** Test 1: 1,379 pairs / 10 bins; Test 2: 500 pairs / 2 models; Test 3: 200 pairs / 3 bias phrases; Test 4: 2,552 unique sentences / 50 trials per condition
- **Seed:** 42
- **Total runtime:** ~32 seconds

---

## Limitations

1. **STS-B is a single dataset.** Results might differ on other semantic tasks (NLI, QA, retrieval). However, STS-B is the standard benchmark for semantic similarity -- exactly the domain where "agreement reveals truth" should be strongest.

2. **Only two embedding models tested.** More diverse models (different architectures, training data) could change multi-model results. However, the fundamental R formula issue is structural, not model-dependent.

3. **Echo chamber simulation is synthetic.** Real echo chambers (social media, coordinated information campaigns) may have more complex structure than simple noise-perturbed duplicates. However, the formula's behavior on this simple case reveals a fundamental property: R = E/grad_S inherently rewards homogeneity over diversity.

4. **Bias phrases are relatively benign.** We used common English preambles rather than adversarial attacks. More sophisticated bias (e.g., training-data-correlated perturbations) could produce worse results.

5. **Truth proximity metric choice.** We used 1 - MAE(model_cosine, human_norm). Other definitions of truth proximity might yield different correlations with R, but the negative correlation we found is robust to reasonable alternatives.

---

## Interpretation

The formula R = E / grad_S (or R = (E/grad_S) * sigma^Df) measures the ratio of mean pairwise similarity to similarity dispersion. This is structurally a measure of **embedding homogeneity** -- how tightly clustered the vectors are.

- **High homogeneity (near-duplicates):** E is high, grad_S is low, R is extreme.
- **Genuine semantic agreement (diverse sentences, similar meaning):** E is moderate, grad_S is moderate, R is modest.
- **Disagreement (dissimilar sentences):** E is low, grad_S is moderate/high, R is low.

R cannot distinguish between "vectors are similar because they represent the same truth" and "vectors are similar because they are copies of each other." This is not a failure of calibration or threshold-setting -- it is a structural property of the formula. Any ratio of mean similarity to similarity variance will exhibit this behavior.

The claim that "high agreement reveals truth" requires an external condition -- independence of observations -- that R itself cannot verify. When that condition holds, agreement may indeed track truth (by the Condorcet jury theorem or its continuous analogs). But R the formula does not check, encode, or enforce independence. It simply measures how similar the inputs are.
