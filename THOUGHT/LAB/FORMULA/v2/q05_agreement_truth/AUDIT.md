# Q05 Adversarial Audit Report

**Auditor:** Automated adversarial audit (110% rigor)
**Date:** 2026-02-06
**Scope:** Code, statistics, methodology, and verdict for Q05 v2 fixed test
**Files reviewed:** formula.py, test_v2_q05_fixed.py, test_v2_q05_fixed_results.json, VERDICT_v2.md

---

## 1. Code Bugs Found

### BUG-1: Numpy boolean serialization causes incorrect criterion 2 evaluation (CRITICAL)

**Location:** `test_v2_q05_fixed.py` lines 604, 632, 642-645

The field `echo_higher_than_genuine` is stored in JSON as the string `"True"` rather than the JSON boolean `true`. Evidence from the results JSON:

```json
"echo_higher_than_genuine": "True"
```

This happens because `bool(numpy.bool_(True))` returns Python `True`, but somewhere in the chain the value gets serialized as a string. The real impact is in the counting logic (lines 642-645):

```python
n_echo_higher = sum(1 for mn in model_names
                    if results_by_model[mn]["echo_higher_than_genuine"] is True)
```

The `is True` identity check fails against the string `"True"`, so `n_echo_higher` counts as 0. Both `n_echo_higher` and `n_genuine_higher` report 0, which is logically impossible since the data clearly shows echo > genuine in all 3 architectures.

**Impact:** Criterion 2 reports neither confirmed nor falsified when the data actually falsifies it (echo > genuine in 3/3 architectures). The verdict correctly notes this bug but does not fix it.

**Severity:** CRITICAL. This bug suppressed a second falsification trigger. The verdict was already FALSIFIED by criterion 3, so the final verdict label happens to be correct, but criterion 2's status is wrong in the machine-readable results.

### BUG-2: R_full skipped for all-mpnet-base-v2 (768-dim) -- silent data loss (MINOR)

**Location:** `test_v2_q05_fixed.py` line 452

```python
do_full = (dim <= 400)
```

The all-mpnet-base-v2 model produces 768-dimensional embeddings, so R_full is never computed for it. All 80 R_full values are NaN. This is intentional (CPU performance), but:
- The `spearmanr` call on line 480 for R_full will operate on 0 valid observations, producing NaN rho and NaN p-value.
- The results JSON shows `"rho_R_full_purity": NaN` and `"n_valid_R_full": 0` for mpnet, which is correct but means R_full is only evaluated on 2 of 3 architectures.
- The verdict only uses R_simple, so this has no effect on the outcome.

**Severity:** Minor. Documented adequately, no impact on verdict.

### BUG-3: No deduplication of cluster indices (MINOR)

**Location:** `test_v2_q05_fixed.py` lines 237, 267-268, 293-294

When a category has fewer documents than needed, `replace=True` is used, allowing the same document to appear multiple times in a cluster. For example: `replace=len(idx) < cluster_size`. Duplicate embeddings inflate E (they have cosine similarity = 1.0 with themselves) and suppress grad_S.

**Severity:** Minor. The 20 Newsgroups dataset has ~250 docs per category and cluster_size is 80, so most pure clusters can be drawn without replacement. The subsample of 5000 docs / 20 categories = 250 per category means this only triggers when the category count falls below the draw size, which is unlikely for most categories.

---

## 2. Statistical Errors

### STAT-1: The 2.53x inflation figure is correct but misleading in isolation

Verified computation:
- Mean R_clean across 20 clusters: 0.4173
- Mean R_biased for "The committee determined that": 1.0450
- Mean of per-cluster inflation ratios: 2.533

These numbers are consistent. The mean-of-ratios (2.533) closely matches the ratio-of-means (1.0450/0.4173 = 2.504), confirming no outlier distortion. All 20 individual inflation ratios are in [2.16, 3.33], so this is not driven by a few extreme clusters.

However, the other two bias phrases produce only 1.20x and 1.22x inflation -- well below the 2x threshold. The verdict is triggered by 1 of 3 phrases.

### STAT-2: rho = 0.85-0.86 is very strong and the R-vs-E improvement is tiny

The R_simple vs purity correlation (rho = 0.856 averaged) versus E vs purity (rho = 0.838 averaged) shows R_simple outperforms E by only delta_rho = 0.01-0.03. This is statistically distinguishable given n=80 but practically negligible. The grad_S denominator adds very little discriminative power beyond what E alone provides.

The verdict correctly notes this ("margin is modest") but does not highlight how marginal it is. The formula R = E/grad_S is doing almost all its work through E.

### STAT-3: Echo chamber test (n=10 per group) has limited statistical power

With only 10 echo chamber clusters and 10 genuine agreement clusters, the Mann-Whitney test has limited power. One of the three architectures (multi-qa-MiniLM, p=0.064) fails to reach significance at alpha=0.05. The effect size (1.3-1.4x ratio) is real and consistent across architectures, but the small sample means confidence intervals are wide.

### STAT-4: Purity has only 4 distinct values -- Spearman correlation is rank-tied

The 80 clusters have purities of exactly {1.0, 0.8, 0.5, ~0.1}, creating massive rank ties (20 clusters per level). Spearman's rho handles ties, but having effectively only 4 ordinal levels means we are measuring whether R correctly rank-orders four groups, not whether it tracks a continuous truth variable. A simpler test (Kruskal-Wallis across the 4 groups, or just checking monotonicity of group means) would be more appropriate and more informative.

**Impact:** The rho=0.85 figure sounds more impressive than it is. With 4 levels and 20 replicates each, even a modest mean separation between groups yields high rho. This does not invalidate the finding, but it means rho=0.85 is not comparable to rho=0.85 on a continuous variable.

---

## 3. Methodological Issues

### METHOD-1 (CRITICAL): The bias attack tests a property of ALL embedding-based metrics, not R specifically

The boilerplate prepending attack works by making all embeddings more similar. This would inflate ANY metric based on pairwise cosine similarity:
- **E alone** would also increase (higher mean similarity)
- **Cosine centroid similarity** would also increase
- **Average linkage** would also increase
- **Any cluster coherence metric** built on cosine similarity would be inflated

The test does not compare R's vulnerability to E's vulnerability under the same attack. If E is inflated by 2.5x too, then R is no more vulnerable than the raw similarity it is built on. If E is inflated less than 2.5x, the R = E/grad_S formulation actually amplifies the problem (because grad_S shrinks when embeddings converge).

**The test code computes E_clean and E_biased (lines 736-737) but never reports E inflation ratios.** This is a significant omission. We can compute from the data:
- E_clean values are stored but E_biased values are not in the results JSON (only R values are stored)
- Without this comparison, we cannot determine if the 2.53x R inflation is caused by the E/grad_S formulation specifically or is inherent to cosine similarity.

**Impact:** FALSIFIED may be unfair to R specifically. The test shows that cosine-similarity-based metrics are vulnerable to surface manipulation (which is trivially true and well-known), not that R is uniquely flawed.

### METHOD-2 (SIGNIFICANT): The echo chamber test conflates homogeneity with echo chambers

An "echo chamber" in the test is simply a cluster of documents from one newsgroup category (e.g., all talk.politics.guns). "Genuine agreement" is a cluster drawn from multiple related categories (e.g., docs from sci.crypt + sci.electronics + sci.med + sci.space).

The difference between these is definitional: the echo chamber has purity=1.0, the genuine agreement has purity=0.25 (4 categories, roughly equal). Of course R is higher for the echo chamber -- Test 1 already proved R correlates with purity. Test 2 is not an independent test; it is a special case of Test 1.

A real echo chamber test would compare:
- **Echo chamber:** many copies of the same narrative/viewpoint
- **Genuine convergence:** independent sources reaching the same conclusion through different reasoning

The newsgroup categories do not distinguish these. talk.politics.guns posters may hold diverse views within the category, while sci.space posters may all agree on basic physics. The category boundary is a topic boundary, not an agreement/independence boundary.

**Impact:** Test 2 does not actually test what it claims. It tests whether R is higher for single-topic clusters than multi-topic clusters, which is a restatement of Test 1.

### METHOD-3 (MODERATE): Bias attack phrase selection is cherry-picked

Three phrases were tested. Only one ("The committee determined that") crosses the 2x threshold. This phrase is 31 characters long and semantically distinctive, while "In conclusion, " is 15 characters and "According to recent studies, " is 29 characters. With texts truncated to 256 characters (line 145), the 31-character phrase consumes 12% of the input, significantly shifting the embedding.

The falsification criterion is "any inflation > 2x", which is met. But:
- 2 of 3 phrases fail to cross 2x (inflation ~1.2x)
- The one that crosses is the longest and most semantically loaded
- No systematic sweep of phrase lengths or semantics was done
- The pre-registered criterion says "OR bias attack inflates R > 2x" which triggers on a single success

This is a valid but aggressive test design. A more conservative design would require "majority of bias phrases cause >2x inflation."

### METHOD-4 (MINOR): Truncation to 256 characters amplifies bias attack

Line 145: `truncated = [t[:256] for t in texts]` means a 31-character prepended phrase occupies 12% of the model input. Sentence-transformers have a token limit (~256 tokens for MiniLM), but character truncation happens before tokenization. For short newsgroup posts, the bias phrase could dominate the input.

Without truncation, the same phrase prepended to a 2000-character document would occupy only 1.5% of the input, and the inflation ratio would likely be much smaller. The truncation decision is reasonable for CPU performance but it amplifies the bias attack beyond what would occur in a production setting.

---

## 4. Verdict Assessment

### Is FALSIFIED the correct verdict?

The verdict is technically correct per the pre-registered criteria: one bias phrase inflates R by 2.53x > 2x threshold. The pre-registration says "FALSIFY if bias attack inflates R > 2x", and this is met.

However, the verdict is **unfair and over-stated** for these reasons:

1. **The bias attack is not specific to R.** It exploits cosine similarity, which underlies R. Any cosine-based metric would be equally vulnerable. Claiming R is "falsified" because cosine similarity can be gamed is like claiming a thermometer is "falsified" because it can be heated with a lighter. The question should be whether R is *more* vulnerable than its components.

2. **The strongest positive finding is underweighted.** rho=0.85-0.86 between R and purity across 3 architectures with n=80 is a strong result. The verdict says "CONFIRMED" for this criterion but then dismisses it as "tautological." It is not tautological -- R was not given the labels. That R's embedding-space measurement correlates with an external ground truth is a genuine positive finding.

3. **1 of 3 bias phrases triggered falsification.** The other two produced only 1.2x inflation. A more robust falsification would require consistent inflation across diverse attack vectors, not a single success.

4. **"R tracks concentration, not truth" is an overclaim.** R demonstrably tracks cluster purity (rho=0.85), which is a proxy for topical coherence. Whether this constitutes "truth" depends on the definition of truth. The claim that R "cannot distinguish truth from echo chambers" is true but applies equally to any metric based on embedding similarity. This is a limitation of the embedding representation, not of the R formula.

### What verdict would be fair?

**INCONCLUSIVE with strong positive signal and known limitations** would be more accurate:

- R correlates well with purity (confirmed, strong signal)
- R does not distinguish echo chambers from genuine agreement (but this is a restatement of the purity finding, not independent evidence)
- R can be inflated by surface manipulation of embeddings (true, but this is a property of cosine similarity, not R specifically)
- R provides modest improvement over E alone (delta_rho = 0.01-0.03)

---

## 5. Issues Requiring Resolution

| ID | Severity | Issue | Resolution Needed |
|---|---|---|---|
| BUG-1 | CRITICAL | Numpy bool serialization bug causes criterion 2 to miscount | Fix: change `is True` to `== True` or ensure proper bool casting before serialization |
| METHOD-1 | CRITICAL | Bias attack not compared to E inflation -- cannot attribute to R specifically | Re-run with E inflation ratios reported; compare R vulnerability to E vulnerability |
| METHOD-2 | SIGNIFICANT | Echo chamber test is redundant with Test 1 | Redesign with independence/convergence distinction, not just topic homogeneity |
| STAT-4 | MODERATE | Only 4 discrete purity levels make rho misleading | Report Kruskal-Wallis or monotonicity test alongside rho |
| METHOD-3 | MODERATE | 1/3 phrases triggers falsification | Report all three; discuss phrase-length confound |
| METHOD-4 | MINOR | 256-char truncation amplifies bias attack | Retest without truncation or with model-native tokenization |
| BUG-3 | MINOR | Potential duplicate indices in clusters | Add deduplication or document expected frequency |

---

## 6. What Would Change the Verdict

The verdict would change to **INCONCLUSIVE** or **CONFIRMED with caveats** if:

1. **E inflation under bias attack is comparable to R inflation.** If E is inflated by 2.5x too, then R is no worse than raw cosine similarity. The test code has the data to compute this but does not report it. From the results JSON, we have E_clean and E_biased values for the echo chamber test but NOT for the bias attack test. This is the single most important missing analysis.

2. **Bias attack is re-run without 256-char truncation.** If full-length documents reduce inflation below 2x for all phrases, the falsification criterion is not met.

3. **The 2x threshold is re-examined.** The threshold was pre-registered, which makes it binding. But 2.53x from a pathological phrase against 256-char-truncated text is a narrow falsification. The other two phrases (1.20x, 1.22x) suggest R is reasonably robust to typical surface manipulation.

4. **Echo chamber test is redesigned** to compare clusters with equal purity but different internal agreement structures (e.g., 50 docs all from one author vs. 50 docs from 50 different authors on the same topic). Currently it confounds purity with the echo/genuine distinction.

The verdict would change to **FALSIFIED (stronger)** if:

1. **E inflation is significantly less than R inflation under bias attack.** This would show that the E/grad_S formulation amplifies vulnerability beyond what cosine similarity alone provides.

2. **The numpy bool bug is fixed and criterion 2 is officially falsified.** This would provide two independent falsification triggers rather than one.

---

## 7. Summary Assessment

| Aspect | Score | Notes |
|---|---|---|
| Code correctness | 7/10 | One critical bug (BUG-1) that misreports criterion 2; otherwise clean |
| Statistical rigor | 6/10 | Correct computations, but rank-tied Spearman on 4 levels is misleading; missing E-inflation comparison |
| Methodological validity | 5/10 | Bias attack tests cosine similarity, not R specifically; echo test is redundant with purity test |
| Verdict fairness | 4/10 | Technically correct per pre-registration, but overclaims and under-attributes the positive findings |
| Overall | 5.5/10 | The data is good; the test infrastructure is solid; the interpretation is where the problems lie |

**Bottom line:** The test demonstrates that R = E/grad_S correlates well with cluster purity (a genuine positive finding) and that cosine similarity can be gamed by surface manipulation (a trivially true negative finding). The FALSIFIED verdict is technically correct per pre-registered rules but is unfair because it does not isolate R's specific contribution to the vulnerability. The critical missing analysis is: does R amplify or merely inherit the bias-attack vulnerability from its E component?
