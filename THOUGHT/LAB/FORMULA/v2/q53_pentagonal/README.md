# Q53: Pentagonal/Phi Geometry Exists in Embeddings

## Hypothesis

Embedding spaces exhibit pentagonal (5-fold) symmetry and phi (golden ratio, 1.618) geometry. Specifically: eigenvalue ratios approximate phi, angular distributions show 5-fold (72-degree) symmetry, golden angle (137.5 degrees) preferences exist, and icosahedral structure is present in high-dimensional embeddings.

## v1 Evidence Summary

Five tests were conducted across 5 embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2, and random baselines):

- **72-degree clustering:** Trained embeddings cluster at 72.85-81.14 degrees (model-dependent). Random baselines at 90 degrees. Trained: 66% at 72-degree window, Random: 0%. This is the ONLY test that passes with discriminative power.
- **Phi spectrum:** 0/5 models have ANY eigenvalue ratios near phi (1.618). 0 out of 77 consecutive eigenvalue ratios. FAILED.
- **5-fold PCA symmetry:** Random baselines ALSO pass (CV_5fold < CV_6fold for random vectors too). Test has zero discriminative power. INCONCLUSIVE.
- **Golden angle (137.5 degrees):** 0/5 models have ANY counts at 137.5 degrees. FAILED.
- **Icosahedral angles:** Counts at icosahedral angles are BELOW uniform baseline expectation. FAILED.

The original verdict was "SUPPORTED" but was corrected to "PARTIAL" after confirmation bias audit. Four independent audits (DEEP, OPUS, ULTRA_DEEP, VERIFY) all recommend FALSIFIED.

## v1 Methodology Problems

The Phase 6C verdict was decisive:

1. **72-degree clustering is arccos(0.3), not 360/5:** Trained embeddings for semantically related words have typical cosine similarity ~0.3, and arccos(0.3) = 72.5 degrees. The 72-degree clustering is a direct consequence of corpus composition and training objective, not pentagonal geometry. Model-dependence (72.85 to 81.14 degrees) rules out a geometric invariant.
2. **Original verdict was confirmation bias:** The ~72 degree mean was interpreted as "pentagonal" when it is actually 73-81 degrees depending on model. The original test code still outputs "SUPPORTED" due to flawed verdict logic that counts non-discriminative tests.
3. **Every specific pentagonal/phi prediction failed:** 0 phi ratios in eigenspectra, 0 golden angle counts, icosahedral counts below baseline, 5-fold PCA indistinguishable from random.
4. **Origin was confirmation bias from Q36:** The pentagonal hypothesis came from pattern-matching on Q36 angle distributions (~72 degrees -> "pentagonal!"). This is a case study in how confirmation bias propagates through a research program.
5. **Status inconsistency:** Main file says PARTIAL, all four independent audits say FALSIFIED.

## v2 Test Plan

### Experiment 1: Systematic Angle Distribution Characterization

Properly characterize the angular distribution of trained embeddings and identify its true cause.

- **Method:** For 10+ embedding models, compute pairwise angle distributions. Fit the distribution to candidate models: (a) arccos of similarity distribution, (b) null distribution for high-dimensional spheres (sin^(d-2)(theta) concentrated at 90 degrees), (c) mixture of semantic-cluster angles and random angles.
- **Data:** 10,000 randomly sampled words per model, 5+ models of varying architecture
- **Analysis:** Test whether the observed angle distribution is fully explained by arccos(mean_cosine_similarity) without any pentagonal assumption. Compute the mean cosine similarity and predict the angle peak. Compare predicted vs observed peak.
- **Key question:** Is the ~72-degree clustering fully explained by the mean cosine similarity of semantic clusters?

### Experiment 2: Exhaustive Golden Ratio Search

Comprehensively test for phi (1.618) in all measurable quantities of embedding spaces, not just eigenvalue ratios.

- **Method:** For 5+ models, compute: (a) all consecutive eigenvalue ratios, (b) all non-consecutive eigenvalue ratios (lambda_k / lambda_{k+n} for n = 1..20), (c) ratios of geometric quantities (Df, alpha, mean angle, std angle), (d) spacing ratios of k-nearest-neighbor distances
- **Analysis:** For each set of ratios, compare the observed distribution near 1.618 to the null distribution. Apply proper multiple comparison correction (Bonferroni or FDR) across all tested ratios.
- **Key question:** Does phi appear ANYWHERE in embedding geometry at a rate above chance?

### Experiment 3: Symmetry Group Detection

Test for the ACTUAL symmetry of embedding distributions, not just 5-fold symmetry.

- **Method:** Apply the angular power spectrum method: project embeddings to successive 2D planes and compute the Fourier transform of the angular distribution. Test for peaks at frequencies 2, 3, 4, 5, 6, 7, 8 (corresponding to 2-fold through 8-fold symmetry).
- **Data:** 5,000+ embeddings per model, 5+ models
- **Analysis:** Report which symmetry orders (if any) show significant peaks above noise. Compare trained vs random embeddings. If no symmetry peak is significant, the distribution is approximately rotationally symmetric.
- **Key question:** Does any discrete symmetry exist in embedding angular distributions?

### Experiment 4: Quasicrystal Structure Test

Test for quasicrystalline (aperiodic) order, which would be the rigorous version of the pentagonal hypothesis.

- **Method:** Compute the diffraction pattern (structure factor) of the embedding point cloud using the method of Baake & Grimm (2013) for detecting quasicrystalline order. Test for sharp Bragg peaks at irrational positions (which indicate quasicrystalline rather than crystalline order).
- **Data:** 10,000 embeddings per model, projected to various dimensionalities (2D through 10D)
- **Analysis:** Compare to diffraction patterns of known quasicrystals (Penrose tilings) and random point clouds. If Bragg peaks exist at positions consistent with 5-fold symmetry, there is genuine quasicrystalline structure.

## Required Data

- Pre-trained models: 10+ models spanning sentence-transformers, GloVe, Word2Vec, FastText, BERT variants
- Large vocabulary: 10,000+ randomly sampled words (not category-curated)
- Random embedding baselines matched to each model's dimensionality
- Reference diffraction patterns for known quasicrystals

## Pre-Registered Criteria

- **Success (confirm):** Phi appears in eigenvalue or distance ratios at > 3x the null rate (Bonferroni-corrected p < 0.01) AND angular power spectrum shows 5-fold peak > 3 sigma above noise for >= 3 models AND diffraction pattern shows Bragg peaks consistent with quasicrystalline order
- **Failure (falsify):** Phi ratio frequency is indistinguishable from null (p > 0.05 after correction) AND no angular symmetry peak exceeds noise AND no diffraction peaks detected
- **Inconclusive:** Marginal phi excess (1-3x null rate); weak symmetry peaks (1-3 sigma); diffraction pattern ambiguous

## Baseline Comparisons

- **Null phi distribution:** For random vectors in R^d, compute the expected frequency of ratios near 1.618 in eigenvalue spectra and distance matrices
- **High-dimensional sphere null:** sin^(d-2)(theta) distribution for angles between random unit vectors (the true null for angular analysis)
- **arccos(similarity) prediction:** Predicted angle peak from the measured mean cosine similarity (no free parameters)
- **Known quasicrystal diffraction:** Penrose tiling diffraction pattern as positive control

## Salvageable from v1

- The finding that trained embeddings cluster at ~72-81 degrees (vs 90 degrees for random) is a genuine empirical observation about semantic similarity structure
- The falsification of phi in eigenvalue ratios (0/77 ratios) is a clean negative result
- The falsification of golden angle (137.5 degrees) and icosahedral structure are clean negatives
- The ULTRA_DEEP_ANALYSIS audit is one of the best falsification documents in the project
- The multi-model comparison framework (5 models) is reusable
- Test script: `test_q53_pentagonal.py`
- Results: `q53_results.json`
