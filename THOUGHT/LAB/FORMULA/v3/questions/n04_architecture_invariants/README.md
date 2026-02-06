# N4: What Geometric Properties Are Architecture-Invariant?

## Why This Question Matters

Q34 found eigenvalue correlations across different models, but dismissed it as shared training data. The "Platonic representation hypothesis" (Huh et al., 2024) independently proposes that different neural networks converge on similar representations. The question is: which properties converge, and is it the data, the architecture, or something deeper?

## Hypothesis

**H0:** Some measurable geometric properties of embedding spaces are invariant across architectures trained on non-overlapping data.

**Specific sub-hypotheses:**
- H0a: Eigenvalue distribution shape (power-law exponent alpha) is architecture-invariant
- H0b: Intrinsic dimensionality is architecture-invariant
- H0c: Clustering structure (number of natural clusters, silhouette score) is architecture-invariant
- H0d: Isotropy (uniformity of direction usage) is architecture-invariant

**H1:** All geometric properties are determined by training data, not architecture. Models trained on non-overlapping data show no convergence.

## Pre-Registered Test Design

### Models (minimum 6, across 3+ architecture families)

| Model | Architecture | Training Data |
|-------|-------------|---------------|
| word2vec (Google News) | Shallow, skip-gram | Google News corpus |
| GloVe (Common Crawl) | Matrix factorization | Common Crawl |
| BERT-base | Transformer encoder | BookCorpus + Wikipedia |
| GPT-2 | Transformer decoder | WebText |
| all-MiniLM-L6-v2 | Distilled transformer | NLI + STS data |
| FastText (Wikipedia) | Shallow, subword | Wikipedia |

**Controlled comparison:** If possible, find or train models on genuinely non-overlapping data (e.g., medical-only vs legal-only corpora).

### Procedure

1. For a shared evaluation vocabulary (10,000 common English words present in all models):
   - Extract embeddings from each model
   - Compute: eigenvalue distribution, power-law exponent, intrinsic dimensionality (MLE), k-means silhouette for k=5,10,20,50, isotropy score
2. Compare properties across models using correlation and distance metrics
3. For the controlled comparison: repeat on non-overlapping domain-specific models

### Success Criteria

- **Invariant found:** At least one geometric property shows |r| > 0.7 across all model pairs, including non-overlapping-data models
- **Data-determined:** Properties correlate only when training data overlaps. Non-overlapping models show |r| < 0.3
- **Architecture-determined:** Properties cluster by architecture family regardless of data

### Implications

- If invariants exist: there's a genuine universal structure in how neural networks represent language
- If data-determined: convergence is an artifact of shared internet text, not a deep finding
- If architecture-determined: the finding is about transformers, not about language

## Dependencies

- None. Can run immediately.

## Related

- v2/Q34 (Platonic convergence -- the observation this follows up)
- v2/Q50 (Completing 8e -- cross-model Df*alpha, may reflect same phenomenon)
- N3 (Sigma determinants -- sigma variation may relate to architecture or domain)
