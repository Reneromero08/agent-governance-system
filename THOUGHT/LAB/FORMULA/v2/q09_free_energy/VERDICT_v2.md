# Q09 v2 Fixed Verdict

## Result: INCONCLUSIVE (formally) -- PARTIALLY CONFIRMED (substantively)

The formal verdict is INCONCLUSIVE because one of two pre-registered criteria is trivially uninformative, preventing a clean CONFIRM. But the correlation evidence is overwhelming and demands honest interpretation.

---

## What Happened

### Test 1: log(R) vs -F Correlation -- CONFIRM

| Architecture | Pearson r | p-value | Spearman rho | p-value |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (384d) | **0.9723** | 2.7e-38 | 0.9396 | 1.1e-28 |
| all-mpnet-base-v2 (768d) | **0.9745** | ~0 | 0.9507 | ~0 |
| multi-qa-MiniLM-L6-cos-v1 (384d) | **0.9741** | ~0 | 0.9595 | ~0 |

All three architectures show Pearson r > 0.97 with p-values indistinguishable from zero. This exceeds the CONFIRM threshold of r > 0.5 in >= 2/3 architectures. The result is consistent across architectures with a spread of only 0.002 in Pearson r.

### Test 2: R-Gating vs Alternatives -- UNINFORMATIVE

| Method | F1 | Accuracy |
|---|---|---|
| R_simple | 1.0000 | 1.0000 |
| R_full | 1.0000 | 1.0000 |
| E alone | 1.0000 | 1.0000 |
| 1/variance | 1.0000 | 1.0000 |
| 1/grad_S | 0.6207 | 0.4500 |
| Random | 0.5002 | ~0.50 |

Every method except 1/grad_S and random achieves perfect F1. The discrimination task is too easy: pure newsgroup clusters (purity = 1.0) vs random mixtures (purity < 0.4) are trivially separable by any coherence or variance metric. R shows 0% advantage over 1/variance because both hit the ceiling.

This means the CONFIRM criterion for gating (+10% advantage) cannot be met, but the FALSIFY criterion (R worse than 1/variance) is also not met. The test provides no discriminating evidence either way.

### Test 3: Rank-Order Consistency -- STRONG

| Architecture | Spearman rho (R_simple vs -F) | p-value |
|---|---|---|
| all-MiniLM-L6-v2 | 0.9396 | 1.1e-28 |
| all-mpnet-base-v2 | 0.9507 | ~0 |
| multi-qa-MiniLM-L6-cos-v1 | 0.9595 | ~0 |

R ranks clusters in almost exactly the same order as -F. This is the weakest claim (merely ordinal agreement), and it passes with extremely high confidence.

### Test 4: Cross-Architecture Consistency -- CONFIRMED

| Metric | Min | Max | Spread |
|---|---|---|---|
| Pearson r | 0.9723 | 0.9745 | 0.0023 |
| Spearman rho | 0.9396 | 0.9595 | 0.0199 |

The R-F relationship is remarkably stable across architectures with different dimensionalities (384 vs 768) and different training objectives. This rules out architecture-specific artifacts.

---

## Why This Contradicts the v1 Verdict

The v1 test found r = -0.13 (wrong sign). The v2 fixed test finds r = 0.97 (near-perfect). What changed?

1. **Cluster construction.** v1 used STS-B sentence pairs grouped by human similarity score into 20 clusters of 40 sentences. These clusters were artificial similarity bins, not natural topic clusters. The similarity scores varied within a narrow range per bin, creating clusters with little variance in embedding coherence. v2 uses 20 Newsgroups with natural topic clusters, random mixtures, and degraded mixtures -- a much wider range of coherence levels.

2. **Cluster size.** v1 used 40 sentences per cluster. v2 uses 200 documents per cluster. Larger clusters provide more stable estimates of both R and F. With n=40, the pairwise cosine similarity matrix (780 pairs) and the 384-dimensional covariance matrix are both noisier.

3. **Cluster count and variety.** v1 used 20 clusters varying only by similarity score. v2 uses 60 clusters spanning pure (single topic), mixed (random), and degraded (half-and-half), providing a much wider dynamic range in both R and F.

4. **Operational E remains the same.** Both v1 and v2 use compute_E from the shared formula (mean pairwise cosine similarity). The identity was not rescued by changing the E definition.

---

## Honest Assessment: Is log(R) = -F + const?

**The correlation is real but the identity is not exact.**

The identity check shows std(log(R) + F) = 13.2 with range(log(R)) = 1.6 and range(F) = 44.6. If log(R) = -F + const were exact, std(log(R) + F) would be zero. The residual standard deviation is 13.2, which is 30% of the range of F. This means log(R) and -F are strongly correlated but are not related by an affine identity.

More precisely: the data is consistent with log(R) being approximately proportional to -F (or some monotone transform of F), but with substantial scatter. The r = 0.97 Pearson correlation means R^2 = 0.94, so 94% of the variance in log(R) is explained by a linear model in -F. The remaining 6% is genuine noise or nonlinearity.

**What this means for the FEP claim:**

- R-maximization and surprise minimization are strongly coupled in the same direction: higher R consistently means lower F.
- But they are not identical operations. Maximizing R will usually reduce F, but the optimal R configuration need not be the optimal F configuration.
- The claim "log(R) = -F + const" is too strong. The honest claim is "log(R) correlates with -F at r > 0.97 across natural text embedding clusters."

---

## Verdict Logic

Pre-registered criteria:
- **CONFIRM** requires: corr > 0.5 in >= 2/3 architectures AND R-gating outperforms 1/variance by >= 10%.
- **FALSIFY** requires: corr < 0.3 in ALL architectures OR R-gating worse than 1/variance.

Results:
- Criterion 1 (correlation): **CONFIRM** (r > 0.97 in 3/3 architectures)
- Criterion 2 (gating): **UNINFORMATIVE** (all methods achieve F1 = 1.0, advantage = 0%)

Since Criterion 2 is at ceiling (uninformative, not falsified), and Criterion 1 overwhelmingly passes, the formal verdict is:

**INCONCLUSIVE** (cannot fully confirm because one criterion is uninformative)

But substantively:
- The strong correlation is **partially confirmed**: log(R) tracks -F with r > 0.97.
- The exact identity log(R) = -F + const is **not confirmed**: residual std = 13.2, far from zero.
- The practical advantage of R over simpler metrics is **not demonstrated**: all methods hit F1 ceiling.

---

## Reconciliation with v1

The v1 FALSIFIED verdict was correct for its methodology but its methodology was too limited:
- Small clusters (40), narrow dynamic range (STS-B similarity bins), single architecture.
- The v1 test was asking "does R correlate with F across clusters that differ only slightly in embedding coherence?" Answer: no, the signal-to-noise ratio was too low.

The v2 test asks "does R correlate with F across clusters that differ substantially in embedding coherence?" Answer: yes, extremely strongly.

This does not rehabilitate the v1 algebraic claim (E = exp(-z^2/2) is still reverse-engineered). It says something different: when you use operational E (cosine similarity) on real data with meaningful variation in cluster quality, R and F are empirically locked together.

---

## Data

- **Dataset:** 20 Newsgroups (sklearn, headers/footers/quotes removed), 5000 documents (250/category)
- **Encoders:** all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d), multi-qa-MiniLM-L6-cos-v1 (384d)
- **Clusters:** 60 per architecture (20 pure, 20 mixed, 20 degraded), 200 documents each
- **Random seed:** 42
- **Elapsed time:** 1342.4 seconds (~22 minutes)
- **Free energy:** Gaussian model with regularization 1e-4

## Limitations

1. **Gating test at ceiling.** The purity gap was too extreme for a meaningful comparison. A harder discrimination task (e.g., purity > 0.6 vs < 0.3) might differentiate R from 1/variance.

2. **Gaussian assumption for F.** The free energy was computed under a Gaussian generative model. Real text embeddings are not Gaussian. If F were computed under a more appropriate model (e.g., von Mises-Fisher for normalized embeddings), the correlation might change.

3. **Cluster construction confound.** Pure clusters are topically coherent (high E, low variance, high R) and have low F because their embeddings cluster tightly. Mixed clusters are incoherent (low E, high variance, low R) and have high F because they scatter widely. The correlation may simply reflect that both R and F respond to the same underlying property (cluster tightness) without log(R) being structurally equal to -F.

4. **Identity does not hold exactly.** std(log(R) + F) = 13.2 is far from zero. This is correlation, not identity.

5. **No causal test.** We show R and F co-vary. We do not show R-maximization produces the same updates as surprise minimization.
