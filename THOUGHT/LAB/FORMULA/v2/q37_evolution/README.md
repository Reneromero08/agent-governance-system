# Q37: Embedding Statistics Evolve with Detectable Patterns

## Hypothesis
Meanings evolve over time on the embedding manifold with measurable dynamics. Specifically: (a) semantic drift rates are approximately universal across common words, (b) R-stability through history predicts word survival, (c) cross-lingual convergence occurs for core concepts even across language isolates, (d) phylogenetic structure is recoverable from embedding statistics, and (e) a conservation law (Df * alpha = constant) persists across time, languages, and semantic categories.

## v1 Evidence Summary
- 15/15 tests passed using real data (no simulations).
- Tier 1 (Historical): Drift rate CV=18.5% across 200 words; 97% maintain viable R; changed words drift 1.10x faster than stable words.
- Tier 3 (Cross-Lingual): Translation equivalents cluster at 2.05x distance ratio; language family phylogeny recoverable with FMI=0.60; isolate language convergence p < 1e-11.
- Tier 4 (Phylogenetic): Hierarchy preserved with Spearman r=0.165; hyponymy predictable at 47.3% Precision@10; ancestral reconstruction at 2.04x signal ratio.
- Tier 9 (Conservation): Df * alpha CV=7.1% across 1850-1990; CV=11.8% across 10 languages; CV=1.0% across 4 semantic categories.
- Tier 10 (Multi-Model): Df * alpha CV=2.0% across 5 models; hierarchy mean r=0.135; phylogeny FMI=0.47.
- Data sources: HistWords (Stanford, 1800-1990), WordNet 3.0, multilingual embeddings (13 languages), 5 embedding models.

## v1 Methodology Problems
1. **No symmetry breaking demonstrated**: The title promises "Semiotic Evolution / Symmetry Breaking" but no symmetry group is identified, no order parameter defined, no symmetric-to-asymmetric transition shown. The "symmetry breaking" framing is marketing.
2. **Cross-lingual convergence is architecturally imposed**: Tiers 3.1-3.3 use multilingual models (paraphrase-multilingual-MiniLM-L12-v2) trained on parallel corpora. The "convergence" is a training objective, not emergent. The isolate language test (p < 1e-11) uses a model explicitly trained on multilingual paraphrase data including those languages.
3. **Post-hoc word selection in Tier 1.3**: "Changed" words (gay, awful, nice) were selected because they are known to have changed meaning. This guarantees the result (drift ratio > 1.05). The threshold of 1.05 (5% effect size) is extremely permissive.
4. **Extremely weak effect sizes celebrated**: Spearman r=0.165 with WordNet hierarchy explains approximately 2.7% of variance. Mean r=0.135 across models is even weaker. These are celebrated as "PASS" against thresholds of 0.1 (1% explained variance).
5. **Conservation law discrepancy**: Df * alpha is approximately 58 for HistWords but approximately 22 for modern transformers -- a 2.6x discrepancy. A conservation law that varies by 2.6x between measurement systems is not a conservation law.
6. **Multi-model "universality" tests related models**: The 5 sentence-transformer models share architecture families, training procedures, and training data. Low CV (2.0%) is expected for derivatives of the same family. True universality would require fundamentally different approaches (count-based vs. neural, independently trained monolingual models).
7. **"M field" framing is metaphorical**: No dynamics on any field are demonstrated. What is shown is that embeddings change across decades (a dataset property) and that certain statistics are moderately stable.

## v2 Test Plan

### Test 1: Pre-Registered Semantic Drift Detection
- Pre-register a list of 50 words predicted to have changed meaning between 1900-1990 and 50 predicted to have stayed stable, BEFORE looking at HistWords data.
- Measure embedding drift for all 100 words.
- Report: (a) drift rate distribution for each group, (b) effect size (Cohen's d) between groups, (c) classification accuracy (can drift rate distinguish changed from stable words?).
- Threshold: Cohen's d > 0.5 (medium effect) for confirmation, not 1.05x ratio.

### Test 2: Cross-Lingual Convergence with Monolingual Models
- Test convergence using independently trained monolingual models (not multilingual models):
  (a) English word2vec or GloVe
  (b) Spanish fastText (trained on Spanish Wikipedia only)
  (c) German fastText (trained on German Wikipedia only)
  (d) Japanese fastText (trained on Japanese Wikipedia only)
- Align embedding spaces post-hoc using Procrustes alignment on a small seed dictionary (100 word pairs).
- Measure: (a) translation equivalent clustering ratio, (b) language family phylogeny FMI.
- This tests whether convergence is genuine or an artifact of shared multilingual training.

### Test 3: Conservation Law Stress Test
- Compute Df * alpha across:
  (a) HistWords decades (1850-1990, 15 time points)
  (b) 5 sentence-transformer models
  (c) GloVe (50d, 100d, 200d, 300d) -- genuinely different dimensionalities
  (d) word2vec (Google News, 300d) -- different training corpus
  (e) fastText (multiple languages, independently trained)
- Report: full distribution of Df * alpha values, CV, and range.
- Explicitly address the 2.6x discrepancy between HistWords (~58) and transformers (~22).
- Test whether Df * alpha = constant within each model family but differs across families (suggesting it is an architecture property, not a universal law).

### Test 4: Phylogenetic Reconstruction Benchmark
- Use established computational phylogenetics methods as baselines:
  (a) Bayesian phylogenetic reconstruction (BEAST or RevBayes) on Swadesh list cognate data
  (b) Embedding-based phylogeny (from v1 method)
  (c) Lexicostatistical distance (standard comparative linguistics)
- Compare each method's tree against the established linguistic phylogeny (Ethnologue / Glottolog).
- Report: normalized Robinson-Foulds distance, FMI, and branch score distance for each method.
- The question is whether embeddings add anything beyond established comparative methods.

### Test 5: Hierarchy Preservation with Proper Baselines
- Measure correlation between embedding distance and WordNet distance.
- Report Spearman r for at least 5 embedding models and compare against:
  (a) Random embeddings (negative control)
  (b) WordNet-trained embeddings (Poincare embeddings, positive control)
  (c) BERT (not a sentence-transformer, different architecture)
- Set the success threshold at r > 0.3 (approximately 9% explained variance, still modest but not trivial).

### Test 6: Temporal Prediction (Not Post-Hoc Detection)
- Using HistWords data from 1850-1950, train a model to predict which words will change meaning between 1950-1990.
- Features: drift rate, neighborhood instability, frequency change, polysemy count.
- Evaluate on held-out 1950-1990 data.
- Report: precision, recall, F1 for predicting semantic change.
- This tests genuine predictive power, not post-hoc detection of known changes.

## Required Data
- **HistWords** (Stanford, historical word embeddings, 1800-1990, freely available)
- **WordNet 3.0** (Princeton, 117K synsets, freely available)
- **GloVe** (Stanford NLP, 6B tokens, 50d/100d/200d/300d)
- **word2vec** (Google News, 300d, 3M words)
- **fastText** (Facebook/Meta, pre-trained for 157 languages, CC + Wikipedia)
- **Ethnologue / Glottolog** (reference linguistic phylogenies)
- **Swadesh list** (standard 100/200 word comparative linguistics list)

## Pre-Registered Criteria
- **Success (confirm):** (a) Pre-registered semantic drift detection achieves Cohen's d > 0.5 AND classification AUC > 0.7, AND (b) cross-lingual convergence holds with independently trained monolingual models (clustering ratio > 1.5 after Procrustes alignment), AND (c) conservation law CV < 0.15 within each model family.
- **Failure (falsify):** (a) Drift detection Cohen's d < 0.2 (negligible effect), OR (b) cross-lingual convergence disappears with monolingual models (clustering ratio < 1.2 after alignment), OR (c) conservation law CV > 0.30 within model families (not a conservation).
- **Inconclusive:** Mixed results across criteria, or conservation law holds within families but the constant differs by > 2x across families (an architecture property, not a universal law).

## Baseline Comparisons
- Random embeddings (negative control for all tests)
- Poincare embeddings trained on WordNet (positive control for hierarchy)
- Standard lexicostatistical methods (baseline for phylogenetic reconstruction)
- Frequency-based semantic change detection (baseline for drift -- just track word frequency)
- Cosine distance between bag-of-contexts vectors (classical distributional semantics baseline)

## Salvageable from v1
- `test_q37_historical.py`, `test_q37_crosslingual.py`, `test_q37_phylogeny.py`, `test_q37_conservation.py`, `test_q37_multimodel.py`: All test infrastructure is well-structured and uses real data. The code for loading HistWords, computing drift, and running bootstrap is reusable.
- `q37_evolution_utils.py`: Utility functions for embedding analysis are reusable.
- `run_all_q37_tests.py`: Test runner can be adapted.
- The HistWords data loading and decade-specific analysis pipeline is genuinely useful.
- The critical bug fixes (zero embedding detection, early decades exclusion) should be preserved.
- The FMI=0.60 phylogeny result is legitimate and worth reproducing with expanded baselines.
