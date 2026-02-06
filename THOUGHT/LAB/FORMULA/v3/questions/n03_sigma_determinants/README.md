# N3: What Determines Sigma Per Domain?

## Why This Question Matters

Q25 falsified sigma universality -- sigma varies 15x across domains (1.92 to 39.44). v1 treated this as a failure. But the variation itself is data. If sigma is predictable from domain properties, that's a characterization of embedding space structure nobody else has published.

## Hypothesis

**H0:** Sigma is predictable from measurable domain properties.

**Specific sub-hypotheses:**
- H0a: Sigma correlates with vocabulary size / type-token ratio
- H0b: Sigma correlates with embedding dimensionality
- H0c: Sigma correlates with mean pairwise cosine similarity (i.e., with E itself)
- H0d: Sigma correlates with intrinsic dimensionality of the embedding subspace

**H1:** Sigma is effectively random across domains with no predictable pattern.

## Pre-Registered Test Design

### Datasets (minimum 20 domains)

Sample from diverse sources:
- Text: Wikipedia (multiple categories), news (AG-News), legal (SCOTUS), medical (PubMed), social media (tweets), literature (Project Gutenberg)
- Non-text (if applicable): image embeddings (CIFAR-10/100 via CLIP), audio embeddings (ESC-50)
- Multiple languages: English, Spanish, Chinese Wikipedia subsets

### Procedure

1. For each domain:
   - Sample n=1000 texts/items
   - Encode with `all-MiniLM-L6-v2` (and one other model for robustness)
   - Compute sigma = compression ratio as defined in v2/GLOSSARY.md
   - Compute domain features: vocabulary entropy, type-token ratio, mean E, std of E, intrinsic dimensionality (via MLE estimator), mean pairwise distance
2. Build regression model: sigma ~ domain features
3. Cross-validate (leave-one-domain-out)

### Success Criteria

- **Predictable:** Cross-validated R^2 > 0.5 -- sigma is more than half explained by domain features
- **Partially predictable:** R^2 between 0.2 and 0.5 -- some structure but high residual
- **Unpredictable:** R^2 < 0.2 -- sigma is effectively a free parameter per domain

### Implications

- If predictable: sigma can be estimated from domain properties, making R computable without fitting sigma per domain
- If unpredictable: sigma^Df is a 2-parameter fit per domain, raising questions about the formula's parsimony

## Dependencies

- None. Can run immediately.

## Related

- v2/Q25 (Sigma universality -- falsified, motivates this question)
- v2/Q22 (Domain-specific thresholds -- may share the same underlying domain structure)
- N5 (Threshold determinants -- parallel question about thresholds)
