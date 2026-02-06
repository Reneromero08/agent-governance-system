# Q07: R Composes Across Scales

## Hypothesis

R = E(z)/sigma is an intensive measure that composes well across scales. There exist composition axioms C1 (Locality), C2 (Associativity), C3 (Functoriality), and C4 (Intensivity) that R uniquely satisfies. R does not grow or shrink with scale, and its value at one granularity level (words, sentences, paragraphs, documents) is comparable to its value at others. Furthermore, R exhibits renormalization group fixed-point behavior, meaning the beta function vanishes at the R fixed point.

## v1 Evidence Summary

A single themed corpus of 64 words, 20 sentences, 5 paragraphs, and 2 documents was used. R was computed at each level using SentenceTransformer embeddings:

- R values across scales: words=0.71, sentences=0.64, paragraphs=0.73, documents=0.96, giving CV=0.158.
- Five alternative operators (additive, multiplicative, max, linear avg, geometric avg) were shown to fail one or more axioms on this corpus.
- An adversarial gauntlet of 6 domains (shallow, deep, imbalanced, feedback, sparse, noisy) all passed with CV ranging from 0.049 to 0.158.
- Four negative controls (shuffled hierarchy, wrong aggregation, non-local injection, random R) all correctly failed.
- A percolation analysis reported critical exponents nu=0.3, beta=0.35, gamma=1.75, claimed as 3D percolation universality class.
- RG beta function analysis reported mean |beta|=0.307 across scale transitions.

## v1 Methodology Problems

The verification identified the following issues with the v1 tests:

1. **Single toy corpus (HIGH).** All "multi-scale" tests used a single hand-crafted nested corpus where paragraphs literally contain the sentences, which contain the words. These are not independent observations at different scales. The thematic coherence was imposed by construction, making R similarity across levels expected rather than informative.

2. **C4 (Intensivity) fails on synthetic data (CRITICAL).** The scale_transformation.py tests reported CV=0.499 and CV=1.851, far above the 0.2 threshold. R values across a scale sweep varied by 1000x (from 7.16 to 0.007). The PASS claim uses only the real-embedding results while ignoring these synthetic failures.

3. **Axioms are reverse-engineered (HIGH).** C1-C4 were designed to select R = E/sigma. The "uniqueness proof" does not actually prove uniqueness over any defined function space -- it merely enumerates 5 alternatives that fail. Infinitely many other functions were not tested.

4. **RG fixed-point claim contradicted by own data (HIGH).** Mean |beta|=0.307, which is 6x the stated threshold of 0.05. Despite this, the receipt reports is_fixed_point=true by using R values (similar by construction) rather than the beta function metric.

5. **Formula inconsistency (CRITICAL).** Two different R formulas exist in the code: multiscale_r.py uses sigma=std(errors), while real_embeddings.py uses sigma=mean(distances) with a concentration factor 1/(1+cv). These compute fundamentally different quantities.

6. **Negative L-correlation passes functoriality (MEDIUM).** Cross-scale L-correlations were words->sentences: -0.245, sentences->paragraphs: 0.354, paragraphs->documents: -1.000 (mean: -0.297). Negative correlation means anti-preservation, yet all three marked as passing C3.

7. **Threshold manipulation (MEDIUM).** CV threshold for intensivity appears as 0.3, 0.2, and 0.5 in different test locations. The most generous threshold is used where results require it.

8. **Phase transition analysis is from toy model (MEDIUM).** Percolation analysis used synthetic random data on a hierarchical tree, not R computed on real embeddings. Critical exponents do not match the claimed 3D percolation class (distance=0.585).

## v2 Test Plan

### Test 1: Multi-Corpus Intensivity (the Core Test)

Compute R at multiple granularity levels using genuinely independent corpora, not nested subsets:

1. Select 5+ publicly available corpora spanning different domains (news, biomedical, legal, fiction, technical).
2. For each corpus, compute R at token-level (sliding windows of 5-10 tokens), sentence-level, paragraph-level, and document-level using consistent methodology.
3. At each level, use independently sampled texts (not nested). For example, sentence-level samples should not be drawn from the same documents used for paragraph-level samples.
4. Fix a single, pre-registered R formula before any computation.
5. Measure CV across levels within each corpus and across corpora at each level.
6. Use at least 3 embedding models (e.g., all-MiniLM-L6-v2, all-mpnet-base-v2, bge-small-en-v1.5) to test model independence.

### Test 2: Uniqueness via Functional Form

Instead of enumerating alternatives, test the functional space more rigorously:

1. Define the space of composition operators as f(E_1, ..., E_n, sigma_1, ..., sigma_n) -> R_composed.
2. Use symbolic regression (e.g., PySR) on the multi-corpus data to discover what functional forms best predict an independent quality metric (e.g., semantic textual similarity benchmark scores).
3. Check whether R = E/sigma emerges as the best or near-best form, or whether other forms dominate.

### Test 3: Scale Covariance Under Controlled Coarsening

1. Start with a large corpus at fine granularity (token-level).
2. Apply controlled coarse-graining: merge tokens into sentences, sentences into paragraphs, paragraphs into sections, sections into documents.
3. At each level, compute R using the pre-registered formula.
4. Measure the beta function: beta(lambda) = dR/d(log lambda) where lambda is the coarsening scale.
5. Test whether beta approaches zero (fixed point) or remains nonzero.

### Test 4: Cross-Domain Generalization

1. Compute R at sentence-level on at least 10 distinct semantic domains.
2. Compare R distributions across domains. An intensive quantity should have similar distributions regardless of domain.
3. Use a Kolmogorov-Smirnov test to assess whether R distributions differ significantly across domains.

### Test 5: Negative Controls

1. Construct adversarial corpora where multi-scale composition should genuinely fail (e.g., shuffled tokens forming valid sentences, machine-translated text with semantic drift).
2. Verify R correctly detects the pathology at the appropriate scale.
3. Use corpora where scale independence is known to fail (e.g., highly domain-specific jargon at word level vs. general text at document level).

## Required Data

- **STS Benchmark** (Semantic Textual Similarity): https://ixa2.si.ehu.eus/stswiki/
- **WikiText-103**: Large-scale document corpus with natural multi-scale structure
- **PubMed abstracts**: Domain-specific biomedical text (available via NCBI)
- **MultiNLI**: Multi-genre natural language inference corpus
- **C4 (Colossal Clean Crawled Corpus)**: Subsets for web domain diversity
- **EUR-Lex**: Legal documents in multiple granularities

## Pre-Registered Criteria

- **Success (confirm):** CV of R across 4+ granularity levels < 0.15 on each of 5+ independent corpora, AND CV < 0.20 averaged across all corpora, AND beta function |beta| < 0.1 at the fixed point.
- **Failure (falsify):** CV > 0.30 on more than 2 corpora, OR R varies by more than 3x across levels on any corpus, OR symbolic regression finds a strictly superior composition operator.
- **Inconclusive:** CV in the range 0.15-0.30 on most corpora; beta function nonzero but small (0.1 < |beta| < 0.3).

## Baseline Comparisons

- **Raw E (evidence without normalization):** R must show lower cross-scale CV than unnormalized E.
- **Cosine similarity of centroids:** R must provide information beyond raw centroid similarity.
- **Random baseline:** R on shuffled data must show significantly higher CV than R on real data (p < 0.01).
- **Alternative normalizations:** R = E/std, R = E/mad, R = E/iqr must be compared head-to-head on the same data.

## Salvageable from v1

- **Adversarial test design structure:** `v1/questions/high_q07_1620/` contains a reasonable gauntlet framework (shallow, deep, sparse, noisy, feedback) that can be adapted with genuinely independent corpora.
- **Alternative operator comparison code:** The 5-alternative test framework is reusable with a larger operator space and proper corpora.
- **Negative control framework:** The 4 negative controls (shuffled hierarchy, wrong aggregation, non-local injection, random R) are well-conceived and can be reused directly.
