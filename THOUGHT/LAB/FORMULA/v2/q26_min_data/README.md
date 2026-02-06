# Q26: Minimum Data Requirements Determinable

## Hypothesis
The smallest observation set that gives reliable R-gating is determinable and follows a predictable relationship. Specifically: N_min scales with log(dimensionality), i.e., N_min = c * log(D) + b. For D=384 (MiniLM): N_min ~ 20-50. For D=768 (BERT): N_min ~ 30-75.

## v1 Evidence Summary
- Original hypothesis FALSIFIED: N_min shows no dependence on D whatsoever (all R^2 = 0.000).
- N_min is constant at 3-5 regardless of dimensionality (D=50 to D=768).
- 7 model configurations tested (3 independent models + 4 PCA projections).
- 50 bootstrap trials per configuration, 200 texts.
- Semantic structure matters: diverse content converges at N=3, coherent/clustered content at N=5.
- CV < 0.10 threshold used to determine N_min.
- Practical recommendation of N=5-10 is supported.

## v1 Methodology Problems
1. **Overgeneralized "no scaling" claim**: PCA projections (D=50, 100, 200, 400) all derive from the same base model (all-mpnet-base-v2, D=768). PCA preserves intrinsic structure, so N_min stability across PCA dimensions is expected. Only 2 truly independent dimensionalities (384, 768) from 3 models were tested.
2. **Arbitrary CV threshold**: N_min = 3 depends on the CV < 0.10 threshold, which is not derived from theory or application requirements. If the threshold were 0.05, N_min would increase. If 0.20, N_min could be as low as 2.
3. **Post-hoc semantic structure finding**: The discovery that coherent content needs N=5 while diverse content needs N=3 was not pre-registered, was tested on a single model (MiniLM-L6-v2 only), and was not validated on held-out data.
4. **No theoretical sample complexity bound**: The original question asks for a sample complexity bound. The study provides only empirical bootstrap estimates, not a theoretical bound of the form N >= f(d, delta, epsilon).
5. **Missing connection to R formula**: N_min is measured for stable embedding statistics, but "reliable gating" is never quantified. How much does the gate decision change with N? What is the gate decision error rate at N=3 vs N=10?
6. **Single embedding model family**: All models are sentence-transformers. No testing with fundamentally different architectures (OpenAI embeddings, count-based methods, etc.).

## v2 Test Plan

### Test 1: Gate Decision Stability (Not Just Statistics)
- For each N in {2, 3, 5, 7, 10, 15, 20, 50}, compute R from N randomly sampled observations and make a gate decision (R > threshold).
- Repeat 1000 times with different random samples.
- Report the gate decision flip rate: how often does the gate decision change between samples at each N?
- This directly measures "reliable gating" rather than CV of embedding statistics.

### Test 2: Genuinely Independent Dimensionalities
- Use models with genuinely different native dimensionalities:
  (a) all-MiniLM-L6-v2 (D=384)
  (b) all-mpnet-base-v2 (D=768)
  (c) text-embedding-3-small from OpenAI API or equivalent (D=1536) -- or use open alternatives like Instructor-XL (D=768) and GTE-large (D=1024)
  (d) GloVe 50d, 100d, 200d, 300d (genuinely different training, not PCA of same model)
- Report N_min for each genuinely independent model/dimensionality.
- Only then test whether N_min scales with D.

### Test 3: CV Threshold Sensitivity
- For each model, compute N_min at thresholds CV < {0.01, 0.02, 0.05, 0.10, 0.15, 0.20}.
- Report a full N_min(CV_threshold) curve.
- Identify whether there is a natural elbow or discontinuity that justifies a specific threshold.

### Test 4: Pre-Registered Semantic Structure Replication
- Pre-register the hypothesis: "Semantically coherent content requires higher N_min than semantically diverse content."
- Test on at least 3 models (not just MiniLM-L6-v2).
- Use at least 4 corpus types: coherent, diverse, contradictory, random.
- Use at least 500 texts per corpus type (up from 200).
- Report confidence intervals on the N_min difference.

### Test 5: Theoretical Bound Derivation
- Derive a concentration inequality for R = E/sigma using Hoeffding or McDiarmid bounds.
- For cosine similarities in [c_min, c_max], bound P(|R_N - R_inf| > epsilon) as a function of N.
- Compare the theoretical N_min from the bound with the empirical N_min.
- If the theoretical bound is loose, report the tightness ratio.

### Test 6: Practical Gate Accuracy vs. N
- For a labeled dataset (e.g., SNLI entailment vs. contradiction), compute gate decisions at each N.
- Report gate accuracy (agreement with the N=100 "ground truth" gate decision) as a function of N.
- Find the N at which gate accuracy reaches 95%, 99%.

## Required Data
- **STS Benchmark** (Semantic Textual Similarity, ~8K pairs)
- **SNLI** (~570K sentence pairs)
- **Wikipedia random articles** (for diverse content)
- **20 Newsgroups** (for topic-coherent content)
- **GloVe pre-trained vectors** (50d, 100d, 200d, 300d, Stanford NLP)

## Pre-Registered Criteria
- **Success (confirm):** N_min for 95% gate decision stability is determinable and consistent (CV < 0.25) across at least 4 genuinely independent models, AND the practical recommendation (specific N value) holds across models.
- **Failure (falsify):** N_min varies by more than 3x across models at the same gate accuracy threshold, OR N_min depends strongly on the CV threshold with no natural elbow (meaning the "minimum" is purely a convention).
- **Inconclusive:** N_min is consistent across sentence-transformer models but untested on other architectures, or gate decision stability differs qualitatively from embedding statistic stability.

## Baseline Comparisons
- N_min from classical statistics (Cochran's formula for sample size)
- N_min from concentration inequalities (theoretical bound)
- Empirical N_min using simple mean cosine similarity instead of R
- Empirical N_min using max cosine similarity instead of R

## Salvageable from v1
- `q26_scaling_test.py`: The multi-model bootstrap infrastructure is well-designed and reusable.
- `q26_semantic_structure_test.py`: The corpus type comparison methodology is sound; needs pre-registration and multi-model extension.
- The practical finding that N=5-10 is sufficient for sentence-transformers is a reasonable starting point for the v2 investigation.
- The bootstrap methodology (50 trials per configuration) is adequate and can be reused.
