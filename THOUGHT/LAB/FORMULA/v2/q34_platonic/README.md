# Q34: Embedding Models Converge to Shared Geometry

## Hypothesis

Independent embedding models trained on potentially different corpora converge to geometrically equivalent semantic structure (up to isomorphism). Specifically, that a "Platonic form" exists -- a unique spectral/geometric structure that all sufficiently trained models discover, regardless of architecture, training objective, or language.

## v1 Evidence Summary

Testing across multiple model families produced these results:

- **Cross-model eigenvalue correlation:** Spearman = 1.0 for all 19 model pairs (8 base models from Microsoft, HuggingFace, BAAI, Alibaba). Held-out generalization: 0.52 trained vs 0.00 random.
- **Sentence transformers (4 models):** Mean cross-model eigenvalue correlation = 0.989, min = 0.980. Df mean = 57.6 +/- 5.9.
- **Cross-architecture (5 architectures -- GloVe, Word2Vec, FastText, BERT, SentenceTransformer):** Mean correlation = 0.971. GloVe-Word2Vec = 0.995, GloVe-BERT = 0.940.
- **Cross-lingual (EN vs ZH):** Mean correlation = 0.914. Multilingual SentenceTransformer EN-ZH = 0.9964. Monolingual EN-BERT vs ZH-BERT = 0.7795.
- **Cumulative variance curve:** Identified as the "Platonic invariant" with mean correlation = 0.9944 across 6 models.
- **Df by training objective:** MLM ~25, similarity ~51, count-based ~44. Df is NOT universal but spectral shape converges.

## v1 Methodology Problems

The Phase 4 verdict identified substantial issues:

1. **Shared training data confound:** All tested models were trained on overlapping web text corpora (Wikipedia appears in nearly all). Spectral convergence may reflect shared Zipfian input statistics, not a "Platonic form."
2. **Tiny word set:** All tests used 68-96 hand-selected concrete nouns. These are high-frequency, imageable words that are the easiest semantic category. The covariance matrix is rank-limited to at most 68, which may dominate the spectral structure.
3. **Fabricated null baseline:** The statistical rigor test (`test_q34_statistical_rigor.py`) uses hardcoded null correlations (0.38 to 0.61) typed into source code, not computed from actual random embeddings. The resulting p-value and Cohen's d are meaningless.
4. **Invariant identification contradicts code:** The saved results file identifies "Decay Rate" (score 1.229) as the best invariant, not "Cumulative Variance" (0.994). The scoring function has a bug (negative CV produces score > 1). Cumulative variance was selected post-hoc for narrative appeal.
5. **"Theorem" is not a theorem:** The "Spectral Convergence Theorem" has no proof, no well-defined conditions ("same underlying reality" is not mathematical), and its claimed bound (corr > 0.99) is contradicted by the data (cross-lingual monolingual = 0.78, cross-architecture = 0.971).
6. **Cross-lingual overclaim:** The strongest cross-lingual result (0.9964) uses a multilingual model trained explicitly to align languages. Monolingual cross-lingual (0.78) is much weaker with 5x Df difference (EN Df = 10.9, ZH Df = 2.26).
7. **R is disconnected:** R is never computed anywhere in Q34. The R=1510 score has no connection to any computation performed.

## v2 Test Plan

### Experiment 1: Convergence with Independent Training Data

Test spectral convergence between models trained on genuinely non-overlapping corpora.

- **Data:** Train or source embedding models on: (a) pre-2000 text only, (b) medical/biomedical text only (PubMed), (c) legal text only, (d) code (GitHub), (e) standard web text baseline
- **Method:** Compute eigenvalue spectra of distance matrices for 10,000 randomly sampled terms from each domain. Compare spectral correlations.
- **Analysis:** If convergence holds between medical-only and legal-only models (no shared training data), the confound is addressed. If convergence drops, shared data was driving it.
- **Key question:** Does spectral convergence survive when the shared training data confound is removed?

### Experiment 2: Large Random Vocabulary Test

Replicate convergence with large, randomly sampled vocabulary instead of hand-picked words.

- **Data:** 10,000 words randomly sampled from frequency-balanced lists (include rare words, function words, abstract terms, not just concrete nouns)
- **Method:** Compute eigenvalue correlations and cumulative variance curves across 5+ models
- **Analysis:** Compare to the v1 results with 68-96 hand-picked words. Test whether the rank-limitation artifact changes results.
- **Key question:** Does convergence hold for large, unbiased vocabulary samples?

### Experiment 3: Proper Null Distribution

Compute actual null spectral correlations from random embeddings.

- **Method:** Generate 10,000 random embedding matrices matching each model's dimensionality and word count. Compute eigenvalue correlations between pairs of random matrices. Repeat 1,000 times to build the null distribution.
- **Analysis:** Compare observed cross-model correlations to this null. Compute proper p-values and effect sizes.
- **Key question:** How much spectral correlation is expected by chance for matrices of these dimensions?

### Experiment 4: Non-Language Domains

Test whether convergence is language-specific or appears in any high-dimensional data.

- **Data:** Protein embeddings (ESM-2), molecular embeddings (ChemBERTa), image embeddings (CLIP, DINOv2), audio embeddings (CLAP)
- **Method:** Compute spectral correlations within each modality (do protein models converge?) and across modalities (do protein and language spectra correlate?)
- **Analysis:** If non-language models also converge but their spectra differ from language models, this supports domain-specific but real convergence. If random data also converges, the effect is trivial.

### Experiment 5: Zipf's Law as Alternative Explanation

Test whether Zipfian frequency statistics alone explain the observed spectral convergence.

- **Method:** Generate synthetic corpora with varying Zipf exponents (s = 0.5, 0.75, 1.0, 1.25, 1.5). Train simple embedding models on each. Compare spectral structures.
- **Analysis:** If spectral structure tracks Zipf exponent rather than "underlying reality," the Platonic interpretation is weakened.

## Required Data

- Pre-trained models: sentence-transformers (5+ models), GloVe, Word2Vec, FastText, BERT variants
- Domain-specific models: BioBERT, LegalBERT, CodeBERT (or equivalent)
- Non-language models: ESM-2 (protein), CLIP (vision-language), DINOv2 (vision)
- Frequency-balanced word lists: 10,000+ words with uniform frequency sampling
- Cross-lingual: separate monolingual models for EN, ZH, ES, AR (no shared pretraining)

## Pre-Registered Criteria

- **Success (confirm):** Spectral correlation > 0.90 between models trained on genuinely non-overlapping data AND correlation > 0.90 for 10,000 randomly sampled words AND observed correlation exceeds the properly computed null by > 3 standard deviations
- **Failure (falsify):** Correlation drops below 0.80 when training data overlap is removed OR correlation for random vocabulary differs by > 15% from hand-picked vocabulary OR proper null correlation is within 2 SD of observed
- **Inconclusive:** Correlation partially drops but remains > 0.85; null is ambiguous; domain-specific models show mixed results

## Baseline Comparisons

- **Proper random null:** Eigenvalue correlations between random matrices of matching dimensions (not hardcoded values)
- **Zipf-matched synthetic:** Correlations between embeddings trained on synthetic Zipfian text
- **Within-model resampling:** Bootstrap spectral correlations from resampling the same model's vocabulary (measures sampling noise)

## Salvageable from v1

- The empirical observation that eigenvalue spectra correlate across models (0.85-0.99) is genuine
- The cross-architecture comparison framework is well-designed
- The finding that training objective matters more than architecture is valuable
- The cumulative variance curve as a candidate invariant is a reasonable starting point
- Test scripts: `test_q34_cross_architecture.py`, `test_q34_cross_lingual.py`, `test_q34_invariant.py`
- Cross-lingual anchor word pairs (68 bilingual pairs)
