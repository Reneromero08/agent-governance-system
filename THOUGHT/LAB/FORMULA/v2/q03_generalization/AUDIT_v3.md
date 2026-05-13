# AUDIT REPORT: Q03 v3 -- Generalization Proof (Round 2)

**Auditor:** Adversarial Audit Agent (Round 2)
**Date:** 2026-02-06
**Scope:** Code correctness, statistical validity, methodological soundness, verdict accuracy
**Previous audit:** AUDIT.md (v2) -- identified 3 CRITICAL, 5 MAJOR, 6 MINOR, 1 INFO
**v3 Verdict under review:** CONFIRMED (R beats E for text; fails financial)
**Severity scale:** CRITICAL (invalidates results), MAJOR (materially affects interpretation), MINOR (should be noted), INFO (observations)

---

## 0. Audit Fix Verification

The v2 audit identified 7 priority issues. Here is the status of each fix:

| v2 Issue | Severity | Fix Status | Verified? |
|----------|----------|------------|-----------|
| STAT-01: No significance test for R vs E | CRITICAL | Steiger's test added | YES -- see STAT-V1 below |
| STAT-05: Financial E-Sharpe tautology | CRITICAL | Replaced with sector purity | YES -- see FIN-V1 below |
| METH-01: E not changed per domain (reframing) | CRITICAL | Honest reframing in header | YES |
| STAT-03: Only 2 similar MiniLM models | MAJOR | 3 models (384d MiniLM, 768d MPNet, 384d QA-MiniLM) | PARTIAL -- see STAT-V2 |
| METH-02: Tri-modal purity distribution | MAJOR | Continuous 0-100% noise in 5% steps | YES |
| METH-05: Overlapping 60-day windows | MAJOR | Non-overlapping windows | YES |
| BUG-01: abs() in beats_E comparison | MAJOR | Signed rho comparison | YES |

**Summary: 6/7 fixes properly applied.** STAT-03 is partial (see below).

---

## 1. Steiger Test Implementation Audit (STAT-V1)

### 1.1 Is the implementation correct?

The Steiger test at lines 92-128 of `test_v3_q03.py` implements the Meng, Rosenthal & Rubin (1992) variant of Steiger (1980). Checking the formula:

```python
r_mean_sq = (r_xz ** 2 + r_yz ** 2) / 2.0
f_val = (1.0 - r_xy) / (2.0 * (1.0 - r_mean_sq))
f_val = min(f_val, 1.0)
h = (1.0 - f_val * r_mean_sq) / (1.0 - r_mean_sq)
z_xz = 0.5 * math.log((1 + r_xz) / max(1 - r_xz, 1e-10))
z_yz = 0.5 * math.log((1 + r_yz) / max(1 - r_yz, 1e-10))
denom = math.sqrt(2.0 * (1.0 - r_xy) / ((n - 3) * h))
z_stat = (z_xz - z_yz) / denom
p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
```

This matches the standard Meng-Rosenthal-Rubin formula. **The implementation is correct.**

### 1.2 Is p=0.0 (reported as "<1e-15") a computation artifact?

**YES, this is a floating-point underflow, not a computation error.**

For text_20ng (median model), the Steiger z = 16.61. The two-tailed p-value is:
  p = 2 * (1 - Phi(16.61))

Phi(16.61) is so close to 1.0 that 64-bit floating-point cannot distinguish
it from 1.0. In IEEE 754 double precision, the smallest representable
p-value from `stats.norm.cdf` underflows to 0.0 at approximately z > 8.3.

The actual p-value is approximately:
  p ~ 2 * exp(-16.61^2/2) / (16.61 * sqrt(2*pi)) ~ 5e-62

So "p < 1e-15" is a correct lower bound. The p-value is genuinely
astronomically small. **This is not an error.**

### 1.3 But WHY is the z-statistic so absurdly large?

This is the key question. A Steiger z of 16.6 is unusual in practice. Let me
trace through the inputs for text_20ng (median model = all-mpnet-base-v2):

- r_xz = rho(R_full, purity) = 0.9497 (R_full predicting purity)
- r_yz = rho(E, purity) = 0.9267 (E predicting purity)
- r_xy = rho(R_full, E) = ??? (the correlation between the two predictors)
- n = 65

The difference r_xz - r_yz = 0.023 is small. But the Steiger z depends
critically on r_xy -- the correlation between R_full and E. If R_full and E
are very highly correlated (say r_xy = 0.99+), then even a tiny improvement
is statistically significant because the test knows the predictors are
nearly identical, so any consistent improvement is unlikely by chance.

**I cannot directly verify r_xy from the results JSON because the R_full-vs-E
correlation is not saved in the output.** This is an audit gap. However, given
that R_full = (E/grad_S) * sigma^Df and E is a component of R_full, the
correlation between them is expected to be extremely high (likely > 0.99),
which would explain the large z.

**Finding STAT-V1a [MINOR]:** The correlation between R and E (r_xy in the
Steiger test) is not recorded in the results JSON, preventing independent
verification of the Steiger z. It should be logged for transparency.

### 1.4 Is Steiger using Spearman rho as input valid?

The code feeds Spearman rho values into a Steiger test designed for Pearson r.
Strictly, Steiger's test assumes bivariate normality, which Spearman violates.

However, this is a well-known and widely accepted practice in the literature.
The Fisher z-transformation of Spearman rho has approximately the same
asymptotic variance as for Pearson r (Fieller, Hartley & Pearson, 1957), so
using Steiger with Spearman rho is defensible for n=65. The test is slightly
anti-conservative (true p slightly higher than reported), but given p ~ 1e-62,
even an order of magnitude correction would not change the conclusion.

**Finding STAT-V1b [INFO]:** Steiger's test technically requires Pearson r, not
Spearman rho. Given the extreme significance levels, this has no practical impact.

---

## 2. Domain Diversity Problem (DOMAIN-01) [CRITICAL]

### 2.1 Both passing "domains" are the same dataset

This is the most serious problem with the v3 test.

| Domain | Dataset | Data Type | Model | Category seed |
|--------|---------|-----------|-------|---------------|
| text_20ng | 20 Newsgroups | Text embeddings | 3 models, median selected | seed=42, 5 random categories |
| text_alt | 20 Newsgroups | Text embeddings | all-mpnet-base-v2 | seed=99, 5 random categories |
| financial | Yahoo Finance | 60-day return windows | N/A | N/A |

The two "text domains" are:
1. The same dataset (20 Newsgroups)
2. The same data type (text embeddings)
3. The same ground truth metric (cluster purity)
4. The same clustering methodology (noise injection)
5. The only difference is which 5 of 20 categories are selected as base categories

**This is not two domains. This is the same domain tested twice with different random seeds.**

The pre-registration says: "CONFIRMED if R significantly outperforms E (Steiger p<0.05) in >=2/3 domains." It defines three domains as ["text", "text_secondary", "financial"]. The pre-registration describes text_secondary_ground_truth as "cluster purity on different corpus" -- but the actual implementation uses the SAME corpus (20 Newsgroups), just with a different random seed for category selection.

A "different corpus" would be, for example:
- Wikipedia articles
- IMDb reviews
- ArXiv abstracts
- Reuters news
- Any dataset that is NOT 20 Newsgroups

Using 20 Newsgroups with seed=42 and 20 Newsgroups with seed=99 is not a
"different corpus." It is the same corpus with different category subsets.

### 2.2 Impact on verdict

If text_20ng and text_alt are counted as ONE domain (which they are, scientifically),
then the scorecard becomes:

| Domain | Sig beats E? |
|--------|-------------|
| text (20NG, any seed) | YES |
| financial | NO |

That is 1/2 domains, not 2/3. The pre-registered rule says:
- CONFIRMED: >= 2/3
- FALSIFIED: 0/n
- INCONCLUSIVE: otherwise

With 1/2 passing, the verdict should be **INCONCLUSIVE**, not CONFIRMED.

### 2.3 The VERDICT_v3.md acknowledges this but does not downgrade

The VERDICT_v3.md states (lines 189-194):

> "The 2 passing domains are both text (same dataset, different categories).
> If the criterion is '2 genuinely different domain types,' the verdict
> would be INCONCLUSIVE."

This is honest. But the verdict then proceeds to use the CONFIRMED classification
anyway, arguing that the pre-registration said ">=2/3 domains" without requiring
different data modalities. This is technically correct but scientifically
misleading. The pre-registration listed "text_secondary" with ground truth
"cluster purity on different corpus" -- but the implementation does not use a
different corpus.

**The pre-registration was violated on its own terms: text_secondary was supposed
to be a "different corpus" but is in fact the same corpus.**

---

## 3. Financial Domain Analysis (FIN-V1)

### 3.1 Tautology fix verified

The v2 audit identified that using Sharpe ratio as ground truth was tautological
(E and Sharpe both measure return consistency). v3 correctly replaces Sharpe with
sector classification purity. This is a genuinely independent ground truth.
**Fix verified.**

### 3.2 Financial results are genuine null

The financial correlations are:
- R_simple vs sector purity: rho = 0.163, p = 0.365
- R_full vs sector purity: rho = 0.173, p = 0.336
- E vs sector purity: rho = 0.163, p = 0.365
- Steiger R_full vs E: z = 0.186, p = 0.853

None of these are significant. R_simple and E produce *identical* rho values
(0.163), which is suspicious and deserves investigation (see STAT-V3 below).

### 3.3 R_simple = E in financial domain?

**Finding STAT-V3 [MAJOR]:** In the financial domain, rho(R_simple, purity) =
rho(E, purity) = 0.16296 (identical to 5 decimal places). This means the Spearman
rank-ordering of R_simple values is identical to the Spearman rank-ordering of E
values across all 33 financial clusters. Since R_simple = E/grad_S, this implies
that grad_S is approximately constant across all financial clusters, or at least
that it does not change the rank order.

Looking at the financial cluster data:
- grad_S ranges from 0.137 to 0.248 across 33 clusters
- This is a ~1.8x range, which should affect rank order

However, E ranges from -0.007 to 0.097, while grad_S ranges from 0.137 to 0.248.
When E is small relative to grad_S (all |E| < 0.1 while grad_S > 0.13), the
division E/grad_S approximately preserves the rank order of E because grad_S acts
as a near-constant denominator relative to the scale of E variation.

This means **R_simple provides ZERO additional information over E in the financial
domain.** The Steiger z for R_simple vs E is exactly 0.0 (p=1.0), confirming this.

### 3.4 Non-overlapping windows fix verified

v3 uses strictly non-overlapping 60-day windows (n_windows = len(ret_series) // window_size),
producing ~8 windows per stock over 2 years. This eliminates the 98.3% overlap
pseudo-replication from v2. **Fix verified.**

### 3.5 Financial sample size concern

With 33 clusters and only 15 windows per cluster, the financial domain has limited
statistical power. However, n=33 is adequate for detecting moderate effects
(rho > 0.35 at alpha=0.05, 80% power). The null result (rho=0.16-0.17) is below
this detection threshold but is presented honestly. **No fix needed.**

### 3.6 Small cluster sizes in financial domain

**Finding FIN-V2 [MINOR]:** Every financial cluster has exactly n_windows=15. With
15 observations in 60 dimensions, the covariance matrix is rank-deficient (rank
at most 14). This means:
- sigma (participation ratio / ambient dim) is bounded above by 14/60 = 0.233
- Df computation from eigenvalue decay has at most 14 non-zero eigenvalues
- The formula components are operating in a degenerate regime

Looking at the actual Df values for financial clusters: they cluster tightly
around 0.116-0.118, with extremely low variance. This confirms the degenerate
regime -- the eigenvalue spectrum is dominated by the rank deficiency, not by
genuine fractal structure. R_full ~ R_simple * (sigma^Df) where sigma^Df is
approximately a constant scaling factor (~0.77), which explains why R_full and
R_simple have nearly identical Steiger results.

---

## 4. Model Diversity Assessment (STAT-V2)

### 4.1 Three models is better than two, but...

v3 uses three models:
1. all-MiniLM-L6-v2 (384d, MiniLM architecture)
2. all-mpnet-base-v2 (768d, MPNet architecture)
3. multi-qa-MiniLM-L6-cos-v1 (384d, QA-tuned MiniLM)

This is an improvement over v2's two 384d MiniLMs. The inclusion of a 768d MPNet
model adds genuine architectural diversity.

### 4.2 But two of three models are still MiniLM

Models 1 and 3 are both MiniLM-L6-based (same architecture, same layer count,
same hidden dimension). They differ only in training data (general vs QA-tuned).
This means:
- 2/3 models share the same architecture
- The "3 model" consistency is partially inflated by architectural homogeneity

**Finding STAT-V2 [MINOR]:** The model diversity fix is partial. A stronger
validation would include a model from a different family entirely (e.g., E5,
GTE, BGE, or a non-Sentence-Transformers model). This does not invalidate the
results but weakens the "consistent across models" claim slightly.

### 4.3 Results are very consistent across models

Despite the above caveat, the actual results show:
- R_simple rho: 0.948-0.953 (range = 0.005)
- R_full rho: 0.949-0.952 (range = 0.003)
- E rho: 0.927-0.941 (range = 0.014)
- R_simple Steiger z: 15.3-18.9
- R_full Steiger z: 10.5-16.6

The 768d MPNet (model 2) is the least favorable for R (Steiger z=10.5 for R_full),
which provides some evidence that the result is not architecture-dependent. But
all three models produce comfortably significant results. **Acceptable.**

---

## 5. Continuous Purity Fix Verification

### 5.1 Implementation is correct

v3 creates clusters with noise fractions: [0%, 5%, 10%, 15%, 20%, 30%, 40%, 50%,
60%, 70%, 80%, 90%, 100%] across 5 base categories, yielding 65 clusters. The
actual purity distribution spans from ~0.073 to 1.000.

### 5.2 Purity is NOT uniformly distributed

The purity distribution has a natural asymmetry: a noise fraction of X% produces
purity >= (100-X)% because the "noise" documents are drawn from random categories,
so some noise docs happen to match the dominant category. For example:
- 0% noise -> purity = 1.000
- 50% noise -> purity ~ 0.500-0.550 (some noise matches dominant)
- 100% noise -> purity ~ 0.073-0.150 (random baseline from 20 categories)

This is correct behavior, not a bug. The resulting distribution has good coverage
of the [0.07, 1.0] range, though it is denser at higher purities. This is a
genuine continuous test, unlike v2's tri-modal distribution. **Fix verified.**

### 5.3 But: same 5 categories always used (within a seed)

All 65 clusters for a given model/seed use the same 5 base categories. This means
the test measures R's ability to track purity within 5 specific categories, not
across all 20. The text_alt domain uses a different 5 categories (seed=99), which
provides some category diversity. **Acceptable.**

---

## 6. Code Review

### 6.1 R_simple = SNR identity

The SNR verification confirms that R_simple = SNR = mean(cos)/std(cos) across all
163 clusters with max |R_simple - SNR| = 0.0 (exact equality). This is correct
because compute_E returns mean(cos) and compute_grad_S returns std(cos).

### 6.2 Median model selection for adjudication

The code selects the median model by R_full rho for adjudication (line 871).
This is a reasonable aggregation strategy. The median model is all-mpnet-base-v2
(768d MPNet), which is the most architecturally diverse model. **Good choice.**

### 6.3 Cluster size is constant (150 docs for text, 15 windows for financial)

All text clusters have exactly 150 documents. All financial clusters have exactly
15 windows. This eliminates cluster-size confounds. **Good.**

### 6.4 No data leakage between train/test

There is no train/test split in this test -- all metrics are computed on the full
cluster. This is correct for correlation analysis (no prediction task). **Good.**

### 6.5 Adjudication logic

The adjudication (lines 645-763) correctly:
- Uses signed rho (no abs()) -- BUG-01 fix verified
- Selects the best R metric per domain
- Applies Steiger test for the corresponding R metric
- Requires both Steiger p < 0.05 AND R rho > E rho
- Uses ceil(2n/3) for the confirmation threshold

**Finding BUG-V1 [MINOR]:** The adjudication selects "best R metric" as whichever
of R_simple/R_full has the higher rho, then uses the corresponding Steiger test.
This is a form of cherry-picking: you get two chances to beat E (R_simple or
R_full). A stricter approach would pre-register which R metric to use. In
practice, R_simple and R_full are extremely similar (0.948 vs 0.950 for text),
so this has negligible impact. But it should be noted.

### 6.6 Import structure

All major imports (numpy, pandas, scipy) are at module level. sentence_transformers
is imported inside run_text_domain(), which is acceptable for a heavy optional
dependency. yfinance is imported inside run_financial_domain(), also acceptable.
**BUG-03 fix verified.**

### 6.7 Hardcoded absolute path

**Finding BUG-V2 [MINOR]:** The formula module is loaded via hardcoded absolute
path (line 46): `r"D:\CCC 2.0\AI\agent-governance-system\..."`. This will break
on any other machine or if the repo is moved. Should use `__file__`-relative path.

---

## 7. What Was NOT Tested

### 7.1 No truly different text corpus

The pre-registration listed "cluster purity on different corpus" for text_secondary.
This was not delivered. A proper second text domain would use Wikipedia, IMDb, ArXiv,
or any non-20NG dataset.

### 7.2 No non-text, non-financial domain

The tabular domain from v2 was dropped entirely. While it had problems, removing
it without replacement reduces the test's domain breadth. Image embeddings, audio
embeddings, or biological sequence embeddings would be stronger evidence.

### 7.3 No domain-specific E definitions

Q03 originally asked about changing the E definition per domain. v3 honestly
reframes this as "does cosine R work across data types" -- but this means the
original Q03 question remains unanswered.

---

## 8. The Core Question: Should the Verdict Be CONFIRMED?

### 8.1 Arguments for CONFIRMED (as stated)

1. The pre-registration defines 3 domains and requires 2/3 with Steiger p<0.05
2. text_20ng and text_alt are listed as separate domains in the pre-registration
3. Both text domains pass with extreme significance (z > 8)
4. Financial fails honestly
5. 2/3 = CONFIRMED per the pre-registered rule

### 8.2 Arguments for INCONCLUSIVE (my recommendation)

1. **text_20ng and text_alt are the same domain** -- same dataset, same data type,
   same ground truth, same methodology, different random seed. Counting them as
   two domains is like running the same experiment twice and claiming n=2.

2. **The pre-registration promised a "different corpus"** but delivered the same
   corpus with a different seed. This is a pre-registration violation.

3. **Scientifically, generalization means working across different data types.**
   R works on text embeddings and fails on financial data. That is 1 data type
   out of 2, which is INCONCLUSIVE by any reasonable reading.

4. **The improvement is genuine but small.** R beats E by +0.020 to +0.023 rho
   in text. This is statistically significant but practically marginal. E alone
   achieves rho = 0.88-0.93 with purity, and R improves this to 0.90-0.95.

5. **R_simple = SNR is well-understood.** The "generalization" is really just the
   observation that SNR of cosine similarities is a slightly better quality
   statistic than the mean alone. This is expected from basic statistics and does
   not constitute a novel finding that needs to "generalize."

### 8.3 My assessment

The v3 test is methodologically sound for the text domain. The Steiger test is
correctly implemented, the continuous purity range eliminates the tri-modal
confound, the non-overlapping windows fix the financial pseudo-replication, and
the sector purity ground truth eliminates the tautology. These are all genuine
improvements over v2.

However, the verdict of CONFIRMED rests entirely on counting text_20ng and text_alt
as separate domains. If they are counted as one domain (which they should be,
since they use the same dataset), the verdict is INCONCLUSIVE.

**Recommended verdict: INCONCLUSIVE**

Specifically: "R significantly outperforms E for predicting cluster purity in
text embedding spaces (1 data type). R fails to generalize to financial data.
Cross-domain generalization is not demonstrated."

---

## 9. Issues Requiring Resolution

### P0 (Must fix):

1. **DOMAIN-01 [CRITICAL]:** text_20ng and text_alt use the same corpus (20 Newsgroups).
   Either (a) replace text_alt with a genuinely different text corpus, or
   (b) downgrade the verdict to INCONCLUSIVE.

### P1 (Should fix):

2. **STAT-V1a [MINOR]:** Log the R-vs-E correlation (r_xy input to Steiger) in
   the results JSON for transparency.

3. **STAT-V3 [MAJOR]:** Investigate and document why R_simple and E produce
   identical Spearman rho (0.16296) in the financial domain. Document that
   R_simple provides zero additional discriminative power over E for financial data.

4. **BUG-V1 [MINOR]:** Pre-register which R metric (R_simple vs R_full) to use
   for adjudication rather than selecting the best per domain.

### P2 (Should note):

5. **STAT-V2 [MINOR]:** Note that 2/3 embedding models are MiniLM architecture.
6. **FIN-V2 [MINOR]:** Note that financial clusters operate in a degenerate regime
   (15 samples in 60 dimensions).
7. **BUG-V2 [MINOR]:** Replace hardcoded absolute path with __file__-relative.

---

## 10. Summary Scorecard

| ID | Finding | Severity | New/Carry |
|----|---------|----------|-----------|
| DOMAIN-01 | Two "text domains" are same dataset (20NG) | CRITICAL | NEW |
| STAT-V1a | R-vs-E correlation not logged | MINOR | NEW |
| STAT-V1b | Spearman fed to Pearson-based Steiger | INFO | NEW |
| STAT-V2 | 2/3 models are MiniLM architecture | MINOR | CARRY (partial fix) |
| STAT-V3 | R_simple = E rank-ordering in financial | MAJOR | NEW |
| FIN-V2 | Financial clusters in degenerate regime | MINOR | NEW |
| BUG-V1 | Cherry-picking R_simple vs R_full | MINOR | NEW |
| BUG-V2 | Hardcoded absolute path | MINOR | NEW |

**Counts: 1 CRITICAL, 1 MAJOR, 5 MINOR, 1 INFO**

This is a significant improvement from v2's 3 CRITICAL, 5 MAJOR.

---

## 11. Comparison: v2 vs v3

| Aspect | v2 | v3 | Improved? |
|--------|----|----|-----------|
| Significance testing | None | Steiger's test | YES |
| Financial ground truth | Sharpe (tautological) | Sector purity (independent) | YES |
| Honest framing | "R generalizes" | "Does cosine R work across types?" | YES |
| Model diversity | 2 MiniLMs (384d) | 2 MiniLMs + 1 MPNet (768d) | PARTIAL |
| Purity distribution | Tri-modal | Continuous 0-100% | YES |
| Overlapping windows | 98.3% overlap | Non-overlapping | YES |
| abs() bug | Present | Fixed | YES |
| Domain count | 3 (text, tabular, financial) | 3 (text, text_same_corpus, financial) | REGRESSED |
| Domain diversity | Text + tabular + financial | Text + text_again + financial | REGRESSED |
| Tabular domain | Included (failed) | Dropped | REGRESSED |

The methodological improvements are substantial. The domain diversity has regressed.

---

## 12. Final Assessment

The v3 test demonstrates genuine methodological rigor in the text domain. The
Steiger test implementation is correct, the continuous purity range is a proper
test, the model diversity is adequate, and the financial tautology is properly
fixed. The code is clean and the verdict document is honest about limitations.

The single critical problem is that the CONFIRMED verdict relies on counting
the same corpus with two different random seeds as two separate domains. This
is not scientifically defensible. Removing this inflation reduces the score
from 2/3 to 1/2, which triggers INCONCLUSIVE under the pre-registered criteria.

**What R_simple = E/grad_S = SNR genuinely shows:** In text embedding spaces,
dividing mean cosine similarity by its standard deviation provides a small but
statistically significant improvement over the mean alone for predicting cluster
quality. This is the signal-to-noise ratio of cosine similarities. It is a valid
and useful statistic in domains where cosine similarity is semantically meaningful.
It does not generalize to domains where cosine similarity is not meaningful.

**Bottom line:** The verdict should be **INCONCLUSIVE**, not CONFIRMED. The honest
summary is: "R provides a modest, statistically significant improvement over E in
text embedding spaces. Generalization to other data types is not demonstrated."
