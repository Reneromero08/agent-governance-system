# Q51: Embedding Spaces Have Intrinsic Complex Structure

## Hypothesis

Real-valued embedding spaces are shadows (projections) of a fundamentally complex-valued semiotic space. Specifically: the 8 PCA octants correspond to 8th roots of unity in a complex plane, semantic analogies obey phase arithmetic (theta_b - theta_a = theta_d - theta_c), and closed semantic loops accumulate quantized Berry holonomy. The complex structure is intrinsic to the embedding geometry, not an artifact of the projection method.

## v1 Evidence Summary

Four primary tests were conducted across 3-5 embedding models with extensive retesting (Try2 on 19 models, Try3 stress tests):

- **Zero Signature (Test 1):** Octant phases (assigned as 8th roots of unity) sum to near zero. |S|/n = 0.0206 mean across 5 models (threshold: < 0.1). However, Try2 with 19 models found: mean |S|/n = 0.0904, roots-of-unity evidence = 0/19 models.
- **Pinwheel (Test 2):** Octant-to-phase-sector mapping. Cramer's V = 0.27 (threshold was > 0.5), diagonal rate = 13.0% (threshold > 50%, random expectation = 12.5%). FAILED.
- **Phase Arithmetic (Test 3):** For word analogies, theta_b - theta_a ~ theta_d - theta_c in the PCA 2D plane. 90.9% pass rate at pi/4 threshold across 3 models, 4.98x separation ratio vs non-analogies. Try2 on 19 models: 68-91% pass rates. But Try3 Experiment 1 (random 2D bases): 0/19 models pass -- phase arithmetic requires PCA specifically, not any 2D projection.
- **Berry Holonomy (Test 4):** Q-score = 1.0000 (perfect quantization) for all loops and all models. However, this is the 2D winding number (not Berry phase, which is identically zero for real vectors per Q43). 2D winding numbers are integers by topological necessity.
- **V6 Report Test 6 (Method Consistency):** Phase structure in PC1-2 does NOT extend to PC3-4. Cross-PC correlation = 0.03-0.17 (random data: 0.06). FALSIFIED by the project's own assessment.
- **Try3 Stress Test:** Exp1 (random bases) = 0/19 pass; Exp2 (shared lambda) = 0/19 pass; Exp3 (winding stability) = 19/19 pass (trivially expected); Exp4 (complex probe) = 12/19 pass.

## v1 Methodology Problems

The Phase 4 verdict identified fatal methodology problems:

1. **Complex structure is imposed, not discovered:** The test projects real embeddings to 2D via PCA, declares PC1 = real axis and PC2 = imaginary axis, then finds "complex" properties. Any 2D projection of any data can be interpreted as a complex plane. Try3 Exp1 proved this is PCA-specific (0/19 pass on random bases).
2. **Berry phase is mathematically zero:** Q43 rigorously proved that Berry phase = 0 for real vectors (<psi|d|psi> = 0). Q51's "Berry phase" is actually a 2D winding number, which is an integer by topological definition. Perfect Q-score = 1.0 is guaranteed by the mathematics, not discovered in the data.
3. **Zero signature is circular:** Octant phases are ASSIGNED by the researcher (theta_k = (k+0.5)*pi/4), not measured. The test checks whether these assigned phases, weighted by octant populations, cancel. This requires only roughly balanced octant populations, not roots-of-unity structure. Try2 found 0/19 models with actual roots-of-unity evidence.
4. **Pinwheel test decisively failed:** 13.0% diagonal rate vs 12.5% random expectation is indistinguishable from chance. This directly falsifies the octant-to-phase-sector mapping.
5. **Phase arithmetic is the parallelogram rule:** Word analogies create parallelogram structures in embedding space (Mikolov 2013). PCA projects these parallelograms to 2D, where angle-difference consistency follows geometrically. This is not complex multiplication.
6. **Status never updated:** Main file claims ANSWERED/CONFIRMED despite v6 report showing UNDER INVESTIGATION with a falsified test, Try2 showing 0/19 roots-of-unity evidence, and Try3 showing 0/19 on the two critical experiments.

## v2 Test Plan

### Experiment 1: Intrinsic vs Projection Complex Structure

Definitively test whether "complex" properties are intrinsic or PCA artifacts.

- **Method:** For each of 5+ models, compute phase arithmetic pass rates on: (a) PCA 2D projection, (b) 100 random 2D projections, (c) ICA 2D projection, (d) random rotation of PCA basis, (e) t-SNE/UMAP 2D projection
- **Data:** Google analogy dataset (19,544 analogies) across 5 models
- **Analysis:** If phase arithmetic passes ONLY for PCA and not for random projections, the "complex structure" is a PCA artifact. If it passes for any well-chosen 2D projection, there may be genuine 2D structure worth characterizing.
- **Key question:** Does the phase structure exist in the data, or only in the PCA projection?

### Experiment 2: Complex-Valued Embedding Training

Train actual complex-valued embedding models and compare to real-valued baselines.

- **Method:** Train Word2Vec or GloVe variants with complex-valued weights (using existing complex neural network libraries). Compare: (a) downstream task performance (analogy, similarity, classification), (b) eigenvalue structure (do complex eigenvalues reveal structure real eigenvalues miss?), (c) phase recovery (are the imaginary parts semantically meaningful?)
- **Data:** Standard training corpora (Wikipedia), standard evaluation benchmarks
- **Analysis:** If complex embeddings outperform real ones or reveal meaningful phase structure, the hypothesis gains support. If performance is equivalent, the complex structure adds nothing.
- **Key question:** Does complex-valued training reveal hidden phase information?

### Experiment 3: Parallelogram Rule Sufficiency Test

Test whether the parallelogram rule in embedding space fully explains the observed "phase arithmetic."

- **Method:** For each analogy (a:b :: c:d), compute the parallelogram score (||b - a - d + c||) and the phase arithmetic score (|theta_b - theta_a - theta_d + theta_c|). Compute the correlation between these two scores.
- **Data:** Google analogy dataset, BATS dataset
- **Analysis:** If correlation > 0.9, the parallelogram rule fully explains phase arithmetic and no complex structure is needed. If correlation is low, there may be angular structure beyond what parallelograms predict.

### Experiment 4: Octant Structure Significance

Test whether the 8-octant structure (signs of PC1-3) carries semantic information beyond what 3 principal components provide.

- **Method:** Compare the 8-octant classification to: (a) k-means clustering with k=8, (b) spectral clustering with k=8, (c) random 8-way partition. Evaluate each partition on semantic coherence (within-cluster similarity, cluster purity for known categories).
- **Data:** 5,000 words with known semantic categories
- **Analysis:** If 8-octant partition performs no better than k-means or random partition, octants have no special semantic significance. If octant partition outperforms alternatives, there is genuine sign-structure information.

### Experiment 5: Higher-Dimensional Phase Structure

Test whether phase structure extends beyond 2D (addressing V6 Test 6 falsification).

- **Method:** Compute phase arithmetic in PC1-2, PC3-4, PC5-6, ..., PC(2k-1)-2k planes. Test each pair independently.
- **Data:** Google analogy dataset across 5 models
- **Analysis:** If phase arithmetic works in PC1-2 but not in any other PC pair, the phenomenon is confined to the dominant variance plane and is not intrinsic complex structure. If it works across multiple PC pairs, there may be genuine multi-dimensional complex structure.

## Required Data

- Google word analogy dataset (19,544 analogies in 14 categories)
- BATS balanced analogy test set (99,200 analogies)
- Pre-trained models: 5+ sentence-transformers, GloVe, Word2Vec
- Complex neural network training library (e.g., complexPyTorch)
- Standard training corpus (Wikipedia) for complex embedding training

## Pre-Registered Criteria

- **Success (confirm):** Phase arithmetic passes for >= 3 non-PCA projection methods at >= 60% pass rate AND complex-valued embeddings outperform real-valued by >= 5% on downstream tasks AND phase structure extends to at least PC3-4 plane (pass rate > 50%)
- **Failure (falsify):** Phase arithmetic passes ONLY for PCA projection (< 40% for all others) AND complex embeddings perform equivalently to real AND phase structure confined to PC1-2 only
- **Inconclusive:** Mixed results across projection methods; complex embeddings show marginal improvement (1-5%); phase structure partially extends to PC3-4

## Baseline Comparisons

- **Parallelogram score:** For each analogy, compare phase arithmetic to the standard parallelogram embedding score (3CosAdd, 3CosMul)
- **Random 2D projections:** 100 random orthogonal 2D projections as null distribution
- **Real-valued embedding baselines:** Standard performance on analogy/similarity benchmarks
- **ICA as alternative to PCA:** Independent Component Analysis may reveal different structure

## Salvageable from v1

- The empirical observation that PCA 2D phase arithmetic works at 80-90% pass rates is genuine and worth understanding
- The Try3 stress test framework (testing across 19 models) is well-designed
- The Try2 pre-registration approach is good methodology
- The finding that phase arithmetic fails on random bases (0/19) is a key falsification result that constrains interpretation
- The V6 report's honest falsification of PC3-4 structure is valuable
- Code: `qgt_phase.py`, `test_q51_phase_arithmetic.py`, `test_q51_zero_signature.py`
