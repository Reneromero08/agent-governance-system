# N2: What Does grad_S Actually Encode?

## Why This Question Matters

grad_S = std of pairwise cosine similarities. In the formula R = E/grad_S, this is the denominator. If grad_S encodes noise, dividing by it is signal-to-noise normalization (useful). If grad_S encodes information (topic diversity, semantic richness), dividing by it suppresses signal (harmful). v1 called it "entropy gradient" philosophically but never empirically characterized it.

## Hypothesis

**H0:** grad_S correlates with a known, interpretable property of the observation set.

**Specific sub-hypotheses (test all):**
- H0a: grad_S correlates with topic diversity (number of distinct topics in the text set)
- H0b: grad_S correlates with annotation disagreement (where human labels are available)
- H0c: grad_S correlates with embedding space local density (average k-NN distance)
- H0d: grad_S correlates with text quality metrics (perplexity, coherence scores)

**H1:** grad_S is noise with no interpretable correlate.

## Pre-Registered Test Design

### Datasets

| Dataset | What It Provides | Source |
|---------|-----------------|--------|
| SNLI | Human-labeled entailment with annotator agreement data | HuggingFace |
| STS-B | Continuous similarity scores (1-5) with inter-annotator variance | HuggingFace |
| 20 Newsgroups | Known topic labels (20 categories) | scikit-learn |
| MultiNLI | Genre-labeled text pairs | HuggingFace |

### Procedure

1. Encode all texts with `all-MiniLM-L6-v2`
2. For each natural group (document cluster, annotated pair set, topic group):
   - Compute grad_S
   - Compute topic diversity (entropy of topic distribution)
   - Compute annotator disagreement (where available)
   - Compute local density (mean cosine distance to 5 nearest neighbors)
   - Compute text perplexity (using GPT-2 as reference LM)
3. Run correlation analysis: grad_S vs each property
4. Run multiple regression: can grad_S be predicted from these properties?

### Success Criteria

- **Interpretable:** At least one property correlates with grad_S at |r| > 0.4 (p < 0.01) across 2+ datasets
- **Predictable:** Multiple regression R^2 > 0.3 for predicting grad_S from interpretable features
- **Noise:** No property correlates at |r| > 0.2 across datasets -- grad_S is uninformative

### Implications

- If grad_S = noise measure: dividing by it is justified (SNR normalization)
- If grad_S = information measure: dividing by it is harmful (explains Q10's E>R finding)
- If grad_S = diversity measure: the formula penalizes diverse evidence, which may be a feature or a bug depending on the application

## Dependencies

- N1 results inform interpretation (if E beats R, knowing what grad_S encodes explains why)
- Independent of all v2 Qs

## Related

- N1 (Does E/grad_S outperform bare E?)
- v2/Q1 (Why grad_S?)
- v2/Q10 (Where E outperformed R)
- v2/Q11 (Sensitivity analysis -- grad_S behavior under perturbation)
