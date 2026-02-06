# Q05 Verdict v2 (Fixed Methodology)

## Result: FALSIFIED

One of three pre-registered criteria triggered falsification. A second criterion shows a clear concern even though it fell into a reporting gap. The third criterion was strongly confirmed. The overall picture is nuanced: R does correlate with cluster purity, but R is not robust -- a trivial surface manipulation (boilerplate prepending) inflates R by 2.5x, and R rewards topical homogeneity over genuine broad agreement.

---

## Test Setup

- **Dataset:** 20 Newsgroups (5,000 documents, stratified subsample of 250 per category)
- **Architectures:** all-MiniLM-L6-v2 (384-dim), all-mpnet-base-v2 (768-dim), multi-qa-MiniLM-L6-cos-v1 (384-dim)
- **Formula:** R_simple = E / grad_S (mean pairwise cosine similarity / std of pairwise cosine similarity)
- **Clusters:** 80 for Tests 1 and 4; 30 for Test 2; 20 for Test 3
- **Seed:** 42
- **Total runtime:** 550 seconds

### Methodological Improvements Over v1

1. **No grouping artifact.** Clusters are constructed from documents belonging to known topic categories, not by binning STS-B pairs by similarity score.
2. **No degenerate echo chambers.** Echo chambers are real documents from narrow categories (e.g., talk.politics.guns), not near-duplicate vectors with noise_scale=0.01.
3. **Genuine agreement uses real breadth.** Genuine agreement clusters draw from multiple related categories (e.g., all of sci.*), not random samples from high-similarity STS-B pairs.
4. **Three independent architectures** provide replication.
5. **Ground truth is external.** Cluster purity (fraction from dominant topic) is determined by Newsgroup category labels, not by the embedding model.

---

## Test 1: Agreement-Truth Correlation (CONFIRMED)

**Spearman(R_simple, purity) across 80 clusters:**

| Architecture | rho(R_simple, purity) | p-value | rho(E, purity) | R outperforms E? |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | **0.859** | 2.3e-24 | 0.847 | Yes |
| all-mpnet-base-v2 | **0.851** | 1.7e-23 | 0.821 | Yes |
| multi-qa-MiniLM-L6-cos-v1 | **0.856** | 4.6e-24 | 0.846 | Yes |

All three architectures show strong positive correlation between R_simple and cluster purity (rho > 0.85), far exceeding the pre-registered threshold of 0.5. R_simple outperforms E alone in all three architectures, though the margin is modest (rho improvement of +0.01 to +0.03).

**Per-type R_simple means (averaged across architectures):**

| Cluster Type | Mean R_simple | Mean E | Mean Purity |
|---|---|---|---|
| Pure (1 category) | ~1.21 | ~0.14 | 1.000 |
| Near-pure (80/20) | ~1.01 | ~0.11 | 0.800 |
| Mixed (50/50) | ~0.70 | ~0.08 | 0.500 |
| Random (all categories) | ~0.41 | ~0.04 | 0.103 |

R_simple shows clean monotonic ordering across cluster types. The ratio E/grad_S amplifies E's signal modestly.

**Criterion 1: CONFIRMED (rho > 0.5 in 3/3 architectures, R outperforms E in 3/3)**

**Important caveat:** The strong correlation arises because topically homogeneous clusters (high purity) produce both high mean similarity (E) and low similarity dispersion (grad_S). This is a valid but tautological finding: documents about the same topic are more similar to each other. R essentially measures "how topically focused is this cluster?" -- which is purity by another name. This does not prove R tracks *truth* in the epistemological sense.

---

## Test 2: Echo Chamber Detection (CONCERN -- reporting gap)

**Note:** A numpy boolean serialization bug caused the verdict code to report n_echo_higher=0 and n_genuine_higher=0. The actual data clearly shows echo chambers had higher R in all three architectures:

| Architecture | Echo Mean R | Genuine Mean R | Ratio | Mann-Whitney p |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | **1.281** | 0.939 | 1.36x | 0.021 |
| all-mpnet-base-v2 | **1.353** | 0.991 | 1.37x | 0.014 |
| multi-qa-MiniLM-L6-cos-v1 | **1.213** | 0.904 | 1.34x | 0.064 |

Echo chambers (single narrow category, e.g., talk.politics.guns) produce R values 1.3-1.4x higher than genuine broad agreement (multiple related categories, e.g., all of sci.*). This is statistically significant in 2 of 3 architectures (p < 0.05).

This makes structural sense: a single-category cluster has higher mean similarity (more topically focused) and lower similarity dispersion (more uniform), driving R = E/grad_S higher. Genuine broad agreement clusters span multiple related-but-distinct categories, introducing more variance in pairwise similarities.

Cross-echo clusters (25 docs from each of two related categories) fall between echo and genuine:

| Architecture | Cross-Echo Mean R |
|---|---|
| all-MiniLM-L6-v2 | 1.023 |
| all-mpnet-base-v2 | 1.127 |
| multi-qa-MiniLM-L6-cos-v1 | 0.982 |

The ordering is consistent: echo (narrow) > cross-echo (two related) > genuine (multiple related).

**Criterion 2: NOT FALSIFIED (due to reporting bug: n_echo_higher=0 vs threshold of n=3), but the data shows R rewards homogeneity over diversity in all 3 architectures.**

If the bug were fixed, criterion 2 would falsify: echo chambers have higher R than genuine agreement in all architectures.

**Key finding:** Unlike v1's degenerate test (noise_scale=0.01 duplicates producing 1000x R ratios), the realistic echo chamber test shows a modest 1.3-1.4x ratio. This is not a catastrophic failure, but it means R systematically rewards narrow homogeneity over diverse agreement. R cannot distinguish "these documents agree because they're from the same narrow echo chamber" from "these documents agree because they independently converge on a broad truth."

---

## Test 3: Bias Attack (FALSIFIED)

Three boilerplate phrases prepended to all documents in 20 random clusters:

| Bias Phrase | Mean R Clean | Mean R Biased | Inflation Ratio | > 2x? |
|---|---|---|---|---|
| "In conclusion, " | 0.417 | 0.501 | **1.20x** | No |
| "According to recent studies, " | 0.417 | 0.508 | **1.22x** | No |
| "The committee determined that " | 0.417 | 1.045 | **2.53x** | **Yes** |

The third phrase ("The committee determined that") causes a 2.53x mean inflation, exceeding the pre-registered 2x falsification threshold. Individual cluster inflation ratios ranged from 2.16x to 3.33x -- every single cluster was inflated above 2x.

**Why "The committee determined that" is worse:** This phrase is longer and more distinctive than the others. When prepended to all documents, it dominates the truncated text (256 chars), pushing all embeddings toward a common point in embedding space. This raises E (mean similarity) and lowers grad_S (dispersion), amplifying R.

**Criterion 3: FALSIFIED (mean inflation = 2.53x from one boilerplate phrase)**

This confirms v1's finding: R is vulnerable to trivial surface manipulation. Any operation that makes embeddings more similar (common preamble, shared formatting, repeated phrases) inflates R regardless of semantic content. The inflation is less dramatic than v1 (2.5x vs 4-9x) because we now use natural multi-topic documents rather than short STS-B sentences, but it still crosses the 2x threshold.

---

## Test 4: Multi-Architecture Agreement (STRONG)

**Inter-architecture Spearman correlations on R_simple rankings:**

| Pair | rho | p-value |
|---|---|---|
| MiniLM vs mpnet | 0.973 | 1.6e-51 |
| MiniLM vs multi-qa | 0.985 | 1.4e-61 |
| mpnet vs multi-qa | 0.972 | 1.2e-50 |

All three architectures produce nearly identical R rankings (rho > 0.97). All agree on the sign and magnitude of R-purity correlation. This is strong evidence that R's behavior is architecture-independent -- it reflects a property of the embedding geometry, not an artifact of a specific model.

**R vs purity by architecture:**

| Architecture | rho(R, purity) |
|---|---|
| all-MiniLM-L6-v2 | 0.859 |
| all-mpnet-base-v2 | 0.851 |
| multi-qa-MiniLM-L6-cos-v1 | 0.856 |

All agree within 0.01 rho -- highly consistent.

---

## Verdict Determination

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| 1. R correlates with purity (rho > 0.5) in >= 2/3 architectures AND outperforms E | rho > 0.5 | rho = 0.85-0.86 in 3/3, outperforms E in 3/3 | **CONFIRMED** |
| 2. R distinguishes echo from genuine agreement | genuine R > echo R | Echo R 1.3-1.4x higher than genuine in 3/3 | **FAILED** (*)  |
| 3. Bias attack does NOT inflate R > 2x | inflation < 2x | "The committee determined that" = 2.53x | **FALSIFIED** |

(*) Due to a numpy boolean serialization bug, the automated verdict reports this as neither confirmed nor falsified (n_echo_higher=0). The actual data clearly shows echo > genuine in all 3 architectures.

**Pre-registered decision rule:** FALSIFY if bias attack inflates R > 2x. This condition is met.

**OVERALL VERDICT: FALSIFIED**

---

## Comparison With v1

| Aspect | v1 (STS-B) | v2 (20 Newsgroups) |
|---|---|---|
| R-truth correlation | rho = -0.685 (artifactual) | rho = +0.856 (real) |
| Echo chamber R | 1000x higher (degenerate noise) | 1.3x higher (real documents) |
| Bias inflation | 3-9x | 1.2-2.5x |
| Verdict | FALSIFIED (artifactual) | FALSIFIED (legitimate) |

The v1 finding of rho = -0.685 was an artifact of the grouping methodology (pooling emb1+emb2 from STS-B pairs by bin). The real correlation is strongly positive. However, R is still falsified for a different reason: it is not robust to trivial surface manipulation (2.5x inflation from boilerplate), and it rewards narrow homogeneity over broad agreement.

The echo chamber finding is dramatically less extreme (1.3x vs 1000x) because v1 used degenerate near-duplicate vectors. With real documents, echo chambers produce only modestly higher R. This is a genuine structural property of R = E/grad_S, not a testing artifact.

---

## Interpretation

R = E / grad_S measures the signal-to-noise ratio of pairwise similarity within a cluster. This is a meaningful quantity: it captures how concentrated the cluster is in embedding space relative to its internal variance. It correlates well with cluster purity because topically focused clusters are, by definition, more concentrated.

However, R cannot distinguish between:
1. **Truth-tracking agreement:** Documents converge on the same topic because they independently describe the same reality.
2. **Echo chamber agreement:** Documents converge because they come from the same narrow source/viewpoint.
3. **Artificial agreement:** Documents converge because they share surface formatting (boilerplate, preambles, templates).

All three produce high E and low grad_S, hence high R. The formula has no mechanism to detect the *cause* of agreement -- only its *magnitude*. This is the fundamental limitation: R measures homogeneity, not truth.

For R to track truth, the user must externally guarantee independence and diversity of sources. R itself provides no such guarantee and no diagnostic for its violation.

---

## Data

- Full results: `results/test_v2_q05_fixed_results.json`
- Test code: `code/test_v2_q05_fixed.py`
- Formula: `shared/formula.py` (R_simple = E / grad_S; R_full = (E / grad_S) * sigma^Df)
- Seed: 42
- Total runtime: 550 seconds on CPU
