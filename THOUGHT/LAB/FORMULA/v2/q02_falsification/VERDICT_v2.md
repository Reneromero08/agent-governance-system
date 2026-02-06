# Q02 v2 Verdict (Fixed Methodology)

## Result: FALSIFIED

The formula R = (E / grad_S) * sigma^Df fails the pre-registered falsification criteria. Three of four component-level tests fail across all three embedding architectures. The falsification trigger is: **>= 3 components fail** (3/4 failed).

However, this verdict is more nuanced than v1's. The formula performs surprisingly well on the modus tollens test (avg violation rate = 10.0%), passes all adversarial tests with massive effect sizes (d > 4.5), and its core ratio E/grad_S ranks 1st or 2nd on functional form comparisons. The failure is specifically in **sigma and Df** -- the fractal-geometric components contribute no meaningful variation and can be removed without loss.

## Methodology Fixes From v1

| Problem in v1 | Fix in v2 |
|---------------|-----------|
| Pooled STS-B sentence pairs into clusters (cross-pair artifact) | Used 20 Newsgroups with natural topic clusters |
| n=20 bins for functional form, all p > 0.15 | n=60 clusters (pure/mixed-2/mixed-5), all p < 1e-13 |
| Modus tollens had only 6 groups above threshold | 100 train + 100 test clusters, 11-16 groups above threshold per model |
| Single embedding model | 3 architectures (all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, paraphrase-MiniLM-L3-v2) |
| No pre-registered criteria | All criteria stated before data loaded |

## Test 1: Component-Level Falsification

### E (mean pairwise cosine similarity) -- PASS (3/3 models)

| Model | Pure mean | Mixed mean | Cohen's d | U p-value |
|-------|-----------|------------|-----------|-----------|
| all-MiniLM-L6-v2 | 0.147 | 0.042 | 3.74 | 3.4e-08 |
| multi-qa-MiniLM-L6-cos-v1 | 0.144 | 0.041 | 3.71 | 3.4e-08 |
| paraphrase-MiniLM-L3-v2 | 0.161 | 0.060 | 5.45 | 3.4e-08 |

E strongly distinguishes topically pure clusters from random mixtures. Effect sizes are very large (d > 3.7) and all p-values are below 1e-07. This is the strongest component.

### grad_S (std of pairwise cosine) -- FAIL (0/3 models)

| Model | Pure mean | Mixed mean | Cohen's d | U p-value |
|-------|-----------|------------|-----------|-----------|
| all-MiniLM-L6-v2 | 0.124 | 0.094 | -3.63 | 1.00 |
| multi-qa-MiniLM-L6-cos-v1 | 0.127 | 0.103 | -2.75 | 1.00 |
| paraphrase-MiniLM-L3-v2 | 0.120 | 0.108 | -2.18 | 1.00 |

**Critical finding: grad_S is HIGHER for pure clusters than mixed clusters.** This is the opposite of the expected direction. The formula assumes grad_S measures noise/dispersion (high = bad), but topically coherent clusters have MORE variable pairwise similarities than random mixtures. This is because random mixtures converge to a narrow band of low similarity, while topic clusters have a wider spread (some pairs within the topic are very similar, others less so).

This means the denominator E/grad_S is dividing by a LARGER number for good clusters, partially canceling the signal from E. The formula works despite this, not because of it.

### sigma (compression ratio) -- FAIL (1/3 models)

| Model | Mean | Std | CV |
|-------|------|-----|----|
| all-MiniLM-L6-v2 | 0.113 | 0.008 | 0.074 |
| multi-qa-MiniLM-L6-cos-v1 | 0.109 | 0.009 | 0.086 |
| paraphrase-MiniLM-L3-v2 | 0.065 | 0.024 | 0.378 |

Sigma is nearly constant across categories for the 6-layer models (CV < 0.09, far below the 0.2 threshold). The 3-layer model passes (CV = 0.378) but this appears to be an artifact of the shallower model having more variable eigenvalue spectra, not a meaningful signal. Sigma does not vary enough to contribute discriminative information to R.

### Df (fractal dimension) -- FAIL (0/3 models)

| Model | Mean | Std | CV |
|-------|------|-----|----|
| all-MiniLM-L6-v2 | 0.136 | 0.001 | 0.007 |
| multi-qa-MiniLM-L6-cos-v1 | 0.136 | 0.001 | 0.005 |
| paraphrase-MiniLM-L3-v2 | 0.138 | 0.000 | 0.003 |

Df is effectively constant across all categories and all models (CV < 0.01). The eigenvalue decay exponent does not vary meaningfully between topics. This component contributes nothing to the formula -- sigma^Df is approximately a constant multiplier, making R_full proportional to R_simple.

### Component Summary

| Component | Pass/Total Models | Verdict | Interpretation |
|-----------|-------------------|---------|----------------|
| E | 3/3 | PASS | Strong signal, captures topical coherence |
| grad_S | 0/3 | FAIL | Direction is reversed; pure > mixed |
| sigma | 1/3 | FAIL | Near-constant across categories |
| Df | 0/3 | FAIL | Near-constant (CV < 0.01) |

**1 of 4 components pass.** Falsification criterion (>= 3 fail) is triggered.

## Test 2: Functional Form Comparison

### All correlations are strong and significant

Unlike v1 (where all correlations had p > 0.15), the fixed methodology with proper cluster construction yields highly significant correlations for all variants (all p < 1e-13). The question is not whether R0 correlates with purity, but whether it correlates better than alternatives.

### Rankings by |Spearman rho| with cluster purity (n=60)

**all-MiniLM-L6-v2:**
1. R0 = E/grad_S: rho = 0.879 (p = 2.5e-20) <-- FORMULA
2. R4 = log(E) - log(grad_S): rho = 0.879 (identical -- R4 is monotonic transform of R0)
3. R1 = E/grad_S^2: rho = 0.856
4. R2 = E/MAD: rho = 0.850
5. R5 = E alone: rho = 0.842
6. R3 = E * grad_S: rho = 0.793

**multi-qa-MiniLM-L6-cos-v1:**
1. R0 = E/grad_S: rho = 0.862 <-- FORMULA
2. R4 = log(E) - log(grad_S): rho = 0.862
3. R1 = E/grad_S^2: rho = 0.852
4. R2 = E/MAD: rho = 0.843
5. R5 = E alone: rho = 0.824
6. R3 = E * grad_S: rho = 0.785

**paraphrase-MiniLM-L3-v2:**
1. R5 = E alone: rho = 0.866
2. R0 = E/grad_S: rho = 0.858 <-- FORMULA
3. R4 = log(E) - log(grad_S): rho = 0.858
4. R2 = E/MAD: rho = 0.855
5. R3 = E * grad_S: rho = 0.848
6. R1 = E/grad_S^2: rho = 0.835

R0 ranks 1st on 2 of 3 models and 2nd on the third. However, the differences are small (< 0.04 in rho) and not statistically significant at p < 0.01 in bootstrap comparisons.

### Bootstrap pairwise results (2000 resamples)

R0 beats 0 of 5 alternatives at p < 0.01 on any model. R0 "loses" to R4 (log transform, which is mathematically equivalent to a monotonic transform of R0) on all models, and to R5 (E alone) on the paraphrase model.

**Key insight:** R0 consistently outperforms E alone (win rate 93-96%) on the two 6-layer models, suggesting the grad_S denominator does add modest value despite the reversed direction finding. But this advantage is not statistically significant at p < 0.01.

## Test 3: Modus Tollens

### Results

| Model | Threshold T | Q_min | n(R>T) | Violations | Rate | 95% CI | rho(R,purity) |
|-------|------------|-------|--------|------------|------|--------|---------------|
| all-MiniLM-L6-v2 | 1.132 | 1.000 | 11 | 1 | 9.1% | [1.6%, 37.7%] | 0.903 |
| multi-qa-MiniLM-L6-cos-v1 | 1.051 | 1.000 | 16 | 2 | 12.5% | [3.5%, 36.0%] | 0.884 |
| paraphrase-MiniLM-L3-v2 | 1.274 | 1.000 | 12 | 1 | 8.3% | [1.5%, 35.4%] | 0.834 |

Average violation rate: 10.0%. The 95% CIs are still wide (upper bounds ~36%) due to moderate n, but point estimates are near the 10% criterion. Two of three models pass.

**Critically different from v1:** The Spearman correlation between R and purity on held-out test data is rho = 0.83-0.90 (all p < 1e-27). In v1, this was rho = -0.06 (p = 0.68). The methodology fix (proper cluster construction, not STS-B pooling) completely reverses the modus tollens result.

### Interpretation

R_simple does have genuine predictive power for cluster purity. The conditional "high R implies high purity" holds at roughly the 10% violation level. This is a meaningful result: the formula, despite its component-level failures, produces a useful composite signal.

## Test 4: Adversarial

| Model | R_simple d | R_full d | E alone d |
|-------|-----------|----------|-----------|
| all-MiniLM-L6-v2 | 4.51 | 4.52 | 3.90 |
| multi-qa-MiniLM-L6-cos-v1 | 4.89 | 4.93 | 4.02 |
| paraphrase-MiniLM-L3-v2 | 5.42 | 4.41 | 5.38 |

All effect sizes are massive (d > 4.4). R_simple, R_full, and E alone all strongly distinguish pure from random clusters. R_simple has slightly larger effect sizes than E alone on 2 of 3 models.

## Applying Pre-Registered Criteria

| Criterion | Required | Observed | Triggered? |
|-----------|----------|----------|------------|
| CONFIRM: 4/4 components pass | 4 pass | 1 pass | No |
| CONFIRM: modus tollens < 10% | < 10% | 10.0% avg | Borderline |
| CONFIRM: R0 beats >= 4/5 alternatives | >= 4 wins | 0 wins | No |
| FALSIFY: >= 3 components fail | >= 3 fail | 3 fail | **Yes** |
| FALSIFY: modus tollens > 20% | > 20% | 10.0% | No |
| FALSIFY: R0 loses to >= 3 | >= 3 losses | 1.3 avg | No |

**Verdict: FALSIFIED** via the component-level criterion (3 of 4 components fail).

## Honest Assessment: What the Data Actually Says

The verdict is more complex than a simple "falsified."

### What works:
1. **R_simple = E/grad_S is a useful metric.** It correlates strongly with cluster purity (rho = 0.83-0.90) and passes the modus tollens test on 2 of 3 models. It has massive effect sizes (d > 4.5) distinguishing pure from random clusters.
2. **E is the dominant signal.** Mean pairwise cosine similarity is a strong proxy for topical coherence (d = 3.7-5.5 for pure vs mixed).
3. **The ratio form has modest value.** On 2 of 3 models, E/grad_S outperforms E alone (93-96% bootstrap win rate), though not at p < 0.01.

### What fails:
1. **grad_S is higher for good clusters, not lower.** The formula's intuition that grad_S measures noise is wrong for natural topic clusters. The formula works despite this, because E's signal is strong enough to overwhelm the denominator's contrary effect.
2. **sigma and Df are effectively constant.** The fractal-geometric terms contribute nothing. R_full approximately equals R_simple times a constant. The sigma^Df term can be removed without any loss of discriminative power.
3. **R0 is not significantly better than alternatives.** At the p < 0.01 level, no functional form is distinguishable from the others. The specific ratio E/grad_S is not privileged.

### What changed from v1:
The v1 test found rho = -0.06 between R and quality. The v2 test finds rho = 0.88. This is entirely due to methodology: v1 pooled STS-B sentence pairs into artificial "clusters" where E was dominated by cross-pair similarity, producing noise. V2 uses natural topic clusters where E directly measures within-cluster coherence.

### Bottom line:
The formula R = (E/grad_S) * sigma^Df is falsified as a mathematical entity with four meaningful components. In practice, it reduces to approximately R = E/grad_S * constant, and the grad_S denominator works in the opposite direction from its claimed interpretation. The useful signal is E (mean pairwise cosine similarity), which is a standard, well-understood metric. The formula adds modest but non-significant value beyond E alone.

## Data

- **Dataset:** 20 Newsgroups (18,846 documents, 20 categories, subsampled to 8,000)
- **Models:** all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, paraphrase-MiniLM-L3-v2
- **Cluster sizes:** 100 documents per cluster (4,950 pairwise comparisons)
- **Bootstrap:** 2,000 resamples per comparison
- **Seed:** 42
- **Runtime:** 457 seconds
- **Code:** `THOUGHT/LAB/FORMULA/v2/q02_falsification/code/test_v2_q02_fixed.py`
- **Results:** `THOUGHT/LAB/FORMULA/v2/q02_falsification/results/test_v2_q02_fixed_results.json`
