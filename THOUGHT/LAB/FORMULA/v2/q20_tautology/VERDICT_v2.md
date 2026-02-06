# Q20 v2 Verdict: Is R = (E/grad_S) * sigma^Df Tautological?

**Verdict: FALSIFIED**

R is not confirmed to have genuine explanatory power beyond its components. Specifically: R_full does not significantly outperform E alone in any architecture, and the 8e conservation law fails catastrophically (317.8% mean error). While R_full does correlate well with cluster purity, so does plain E (mean pairwise cosine similarity), nearly as well.

---

## Test Summary

### Environment
- Dataset: 20 Newsgroups (18,846 documents, 20 categories)
- 60 clusters per architecture: 20 pure (purity=1.0), 20 mixed (purity=0.5), 20 random (purity~0.08)
- Architectures: all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d), multi-qa-MiniLM-L6-cos-v1 (384d)
- Total runtime: 2,347.8 seconds
- Seed: 42
- Data hash: ed9911d480529fbe

### Test 1: Component Comparison (Spearman rho with cluster purity)

| Component   | MiniLM-L6 | mpnet-base | multi-qa  |
|-------------|-----------|------------|-----------|
| E           | 0.9007    | 0.8879     | 0.9040    |
| grad_S      | 0.6574    | 0.6988     | 0.6295    |
| sigma       | -0.5557   | -0.6116    | -0.5265   |
| Df          | -0.0772   | 0.2492     | -0.0926   |
| inv_grad_S  | -0.6574   | -0.6988    | -0.6295   |
| sigma^Df    | -0.5572   | -0.6148    | -0.5416   |
| R_simple    | 0.9198    | 0.9129     | **0.9217**|
| **R_full**  | **0.9202**| **0.9149** | 0.9213    |

**Key findings:**
- R_full has the highest |rho| in 2 of 3 architectures, but only by razor-thin margins over E alone (0.019-0.027)
- In the multi-qa architecture, R_simple (0.9217) actually beats R_full (0.9213), meaning the sigma^Df term HURTS performance
- The margin over E is never >= 0.05 in any architecture (required by pre-registered criteria)
- Bootstrap test: R_full vs E has p-values of 0.053, 0.053, and 0.029 -- borderline at best, non-significant in 2 of 3

**Interpretation:** E alone (mean pairwise cosine similarity) already captures ~90% of the variation in cluster purity. The division by grad_S adds ~2% absolute improvement. The sigma^Df fractal scaling term adds essentially nothing -- and sometimes slightly hurts.

### Test 2: 8e Conservation (PR * alpha)

| Sample size | MiniLM-L6      | mpnet-base      | multi-qa        |
|-------------|----------------|-----------------|-----------------|
| n=100       | 406.7 (1770%)  | 799.2 (3575%)   | 362.6 (1568%)   |
| n=200       | 64.0 (194%)    | 517.9 (2282%)   | 64.6 (197%)     |
| n=500       | 80.7 (271%)    | 106.1 (388%)    | 75.4 (247%)     |
| n=1000      | 77.8 (258%)    | 124.1 (471%)    | 72.9 (235%)     |
| n=2000      | 78.0 (259%)    | 123.5 (468%)    | 71.0 (227%)     |

Target: 8e = 21.746. Mean error at n=2000: **317.8%**

**Key findings:**
- PR * alpha ranges from 71 to 124 at n=2000, never anywhere close to 8e = 21.746
- The product scales with min(n, d) through PR, which itself grows with sample size
- At n=100, the product is absurdly high (362-799) due to the rank-deficient regime
- The "8e conservation law" does not hold. This is not a constant; it is a sample-size-dependent statistic

**Interpretation:** The 8e = 21.746 conservation law is definitively falsified. PR (participation ratio) is bounded by min(n, d) and grows with sample size, making PR * alpha a non-convergent quantity. The previously reported values near 8e were artifacts of specific n/d combinations.

### Test 3: Ablation (5 functional forms)

| Form    | MiniLM-L6 | mpnet-base | multi-qa  |
|---------|-----------|------------|-----------|
| R_full  | **0.9202**| **0.9149** | **0.9213**|
| R_sub   | 0.9110    | 0.9028     | 0.8886    |
| R_exp   | 0.9136    | 0.9036     | 0.9147    |
| R_log   | 0.8899    | 0.8795     | 0.9025    |
| R_add   | 0.8953    | 0.8829     | 0.9017    |

**R_full is the best ablation form in 3/3 architectures.** However, the margins are modest (0.0066-0.0327 over next best), and R_exp (exponential suppression) comes close in all architectures. This suggests the general signal-to-noise structure matters, but the specific E/grad_S division is not uniquely privileged.

### Test 4: Novel Predictions

| Prediction | MiniLM-L6 | mpnet-base | multi-qa |
|------------|-----------|------------|----------|
| P1: \|rho\| > 0.4 | PASS (0.92) | PASS (0.91) | PASS (0.92) |
| P2: Top quartile purity > 0.7 | PASS (1.00) | PASS (1.00) | PASS (1.00) |
| P3: R_full is best overall | PASS | PASS | **FAIL** (R_simple=0.9217 > R_full=0.9213) |

Predictions passed >= 2/3 in all architectures. But these are weak predictions: any reasonable signal-to-noise ratio of cosine similarities will correlate with purity at |rho| > 0.4 on clusters with purity ranging from 0.07 to 1.0. The predictions are tautologically easy.

---

## Pre-Registered Criteria Evaluation

**CONFIRM requires ALL of:**
- R_full outperforms all components by >= 0.05 margin: **0/3 FAIL** (margins: 0.019, 0.027, 0.017)
- Positive correlation: **3/3 PASS**
- Bootstrap significant for all components: **1/3 FAIL** (only multi-qa passes)
- 8e within 30%: **FAIL** (317.8% error)
- Best ablation form: **3/3 PASS**
- Novel predictions >= 2/3: **3/3 PASS**

**FALSIFY requires ANY of:**
- R_full does not outperform in any architecture: **TRUE** (0/3 with >= 0.05 margin)
- 8e error > 100%: **TRUE** (317.8%)

**Result: FALSIFIED** (both falsification criteria met)

---

## Honest Assessment

### What R_full does well:
1. It genuinely correlates with cluster purity (rho ~ 0.92 across architectures)
2. It is the best functional form among the ablation variants (by small margins)
3. The correlation sign is correct (positive: higher R = purer clusters)
4. It outperforms most individual components by large margins

### What R_full does NOT do:
1. It does not significantly outperform E alone (plain mean cosine similarity)
2. The sigma^Df term adds negligible value (sometimes negative value)
3. The 8e conservation law is completely wrong -- off by 3-5x consistently
4. The formula's apparent power comes almost entirely from E, which is just "how similar are these embeddings on average?"

### Root cause analysis:
R = (E / grad_S) * sigma^Df. In practice:
- E dominates the correlation (rho = 0.88-0.90)
- Dividing by grad_S adds a small boost (R_simple rho = 0.91-0.92)
- Multiplying by sigma^Df adds essentially nothing
- The formula is, functionally, a slightly dressed-up version of E/grad_S = mean/std of pairwise similarities
- This IS a signal-to-noise ratio. The "tautology concern" from the v1 audit is validated: R is a signal-to-noise ratio dressed in mathematical notation, and the fractal/eigenvalue components are decorative.

---

## Files

- Code: `THOUGHT/LAB/FORMULA/v2/q20_tautology/code/test_v2_q20_fixed.py`
- Results: `THOUGHT/LAB/FORMULA/v2/q20_tautology/results/test_v2_q20_fixed_results.json`
- Formula: `THOUGHT/LAB/FORMULA/v2/shared/formula.py`
