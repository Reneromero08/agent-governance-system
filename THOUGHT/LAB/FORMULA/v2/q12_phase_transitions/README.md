# Q12: R Shows Phase Transition During Training

## Hypothesis

There is a critical threshold for agreement (like a percolation threshold) during model training, and truth "crystallizes" suddenly rather than gradually. Specifically: as a language model is trained, there exists a critical training fraction alpha_c near which semantic structure (as measured by R, generalization ability, or fractal dimension Df) undergoes a sharp phase transition. This transition belongs to a well-defined universality class (claimed: 3D Ising), characterized by specific critical exponents (nu, beta, gamma). The transition is universal across model architectures.

## v1 Evidence Summary

The primary experiment interpolated between untrained and trained BERT weights using weights = alpha * trained + (1 - alpha) * untrained for alpha in {0%, 50%, 75%, 90%, 100%}:

- Generalization scores: 0.02 (0%), 0.33 (50%), 0.19 (75%), 0.58 (90%), 1.00 (100%).
- Largest generalization jump (+0.424) occurred between alpha=0.90 and alpha=1.00.
- Df trajectory: 62.5, 22.8, 1.6, 22.5, 17.3 (non-monotonic, with collapse at alpha=0.75).
- 12/12 "gold-standard physics tests" reported as passing, including finite-size scaling, Binder cumulant crossing, and universality class matching to 3D Ising (nu=0.67, beta=0.34, gamma=1.24).
- Cross-architecture universality claimed for BERT, GloVe, and Word2Vec (CV < 2%).
- Tests 8 and 9 initially failed (10/12), then were redesigned until they passed.

## v1 Methodology Problems

The verification identified the following issues with the v1 tests:

1. **Weight interpolation is not training (CRITICAL).** Linearly interpolating between untrained and trained weights does not reproduce the actual training trajectory. Neural network loss landscapes are highly non-convex; intermediate interpolated weights can land in pathological regions (as evidenced by the alpha=0.75 anomaly where Df collapses to 1.6 and generalization drops to 0.19). The document itself admits "Real training checkpoints: Still open."

2. **Tests 8 and 9 reverse-engineered (CRITICAL).** The Binder cumulant fix explicitly constructed a "direct parametric model" that DEFINES U(alpha, L) to cross at the desired point rather than measuring from independent data. The crossing spread improved from 0.23 to 0.005 not from better data but from engineering the model. Test 8 switched from measuring power-law correlations to generating synthetic fractional Gaussian noise with the desired Hurst exponent.

3. **3D Ising claim is a 3-parameter fit (HIGH).** Matching 3 critical exponents (nu, beta, gamma) to 3 parameters of a known universality class is curve fitting, not discovery. Embedding spaces are not lattice spin systems -- there is no physical Hamiltonian, no partition function, no thermal fluctuations.

4. **Only 5 data points (HIGH).** Five alpha values (0%, 50%, 75%, 90%, 100%) are insufficient to characterize a phase transition. The "largest jump" between 90% and 100% could be interpolation nonlinearity near the trained endpoint.

5. **Cross-architecture test uses same interpolation method (MEDIUM).** BERT, GloVe, and Word2Vec all tested with the same interpolation approach. If interpolation itself creates the transition artifact, all architectures would trivially show the same behavior.

6. **The alpha=0.75 anomaly (MEDIUM).** The drop in generalization at alpha=0.75 (0.19, worse than alpha=0.50 at 0.33) is most parsimoniously explained by linear interpolation producing garbage in non-convex landscapes, not by "unstable intermediate states" in a physical sense.

## v2 Test Plan

### Test 1: Real Training Checkpoints (the Core Test)

Replace weight interpolation with actual partially-trained models:

1. Train a transformer model (e.g., BERT-base or GPT-2-small) from scratch on a standard corpus (WikiText-103 or similar).
2. Save checkpoints at regular intervals (every 1% of total training, or at least 50 checkpoints).
3. At each checkpoint, compute R on a held-out evaluation set using a fixed, pre-registered formula.
4. Also compute: generalization accuracy on downstream task (e.g., SST-2 sentiment), fractal dimension Df of the embedding space, eigenspectrum shape (alpha exponent).
5. Plot all metrics vs. training fraction. Look for sharp discontinuities vs. gradual improvement.
6. If a transition exists, locate alpha_c by fitting a sigmoid and identifying the inflection point.
7. Repeat with at least 3 random seeds to assess reproducibility.

### Test 2: Finite-Size Scaling from Real Data

If Test 1 reveals a candidate transition:

1. Train models of different sizes (e.g., 4-layer, 6-layer, 8-layer, 12-layer transformers) on the same data.
2. Compute the candidate order parameter (generalization, R, or Df) at each checkpoint for each model size.
3. Test finite-size scaling: do the transition curves for different sizes collapse onto a single curve under the scaling ansatz f((alpha - alpha_c) * L^(1/nu))?
4. Extract critical exponents from the collapse. Do NOT pre-specify a target universality class.
5. If collapse fails, the transition is not a genuine phase transition in the statistical mechanics sense.

### Test 3: Order Parameter Identification

1. Candidate order parameters: generalization score, R value, principal eigenvalue gap, cosine similarity to fully-trained embeddings, mutual information between layers.
2. For each candidate, compute susceptibility (variance near the transition) and correlation length (spatial extent of fluctuations in the parameter landscape).
3. The true order parameter should show diverging susceptibility at alpha_c.
4. Check whether the order parameter is extensive or intensive.

### Test 4: Universality Across Architectures

1. Train at least 3 fundamentally different architectures from scratch: a transformer, an LSTM, and a CNN-based text model (e.g., TextCNN).
2. Use the same corpus and evaluation protocol for all.
3. Compare transition locations (alpha_c) and shapes across architectures.
4. True universality means the same critical exponents regardless of architecture; architecture-dependence means the transition is a training artifact.

### Test 5: Null Model

1. Generate a "training trajectory" from random weight perturbations (not gradient descent) applied at the same schedule.
2. Compute the same metrics at each step.
3. The null model should NOT show a phase transition. If it does, the transition is an artifact of the measurement, not of learning.

## Required Data

- **WikiText-103:** Standard language modeling training corpus
- **SST-2:** Sentiment classification downstream evaluation
- **STS-B:** Semantic similarity downstream evaluation
- **Pre-trained model weights:** BERT-base-uncased, GPT-2 (for comparison to known endpoints)
- **Compute budget:** Training from scratch requires significant GPU time (estimate 4-8 V100-days per architecture)

## Pre-Registered Criteria

- **Success (confirm):** A sharp transition in at least one order parameter (sigmoid fit steepness k > 5) is observed at a consistent alpha_c (CV < 0.10 across random seeds), AND finite-size scaling collapse succeeds (R^2 > 0.90), AND the transition is absent in the null model, AND at least 2/3 architectures show the transition at compatible alpha_c.
- **Failure (falsify):** All metrics show gradual, monotonic improvement with no detectable transition (sigmoid k < 2), OR finite-size scaling collapse fails (R^2 < 0.50), OR the null model shows equivalent "transitions," OR different architectures show transitions at incompatible alpha_c values.
- **Inconclusive:** Weak transition detected (sigmoid k 2-5) but finite-size scaling is ambiguous, or only 1/3 architectures show a clear transition.

## Baseline Comparisons

- **Linear interpolation null model:** The v1 interpolation approach applied to real checkpoints, to directly compare interpolation vs. actual training trajectories.
- **Loss curve:** Compare R-based transition detection to simply monitoring training loss. If loss shows the same transition, R adds nothing.
- **Perplexity:** Standard language model metric -- does it show a transition at the same alpha_c?
- **Random weight perturbation:** Null model where weights are randomly perturbed instead of trained.

## Salvageable from v1

- **Interpolation analysis framework:** `v1/questions/high_q12_1520/` contains the alpha-sweep methodology that can serve as a comparison condition (interpolation vs. real checkpoints).
- **Critical exponent extraction code:** The finite-size scaling and Binder cumulant analysis code can be reused IF applied to real training data rather than engineered simulations.
- **Df computation pipeline:** The fractal dimension estimation code from the eigenspectrum is a valid measurement tool.
- **Partial training results as reference:** The interpolation results (alpha=0%, 50%, 75%, 90%, 100%) provide a baseline to compare against real checkpoint trajectories, specifically to test whether interpolation creates artifacts.
