# Q46: Geometric Properties Are Stable Under Perturbation

## Hypothesis

The geometric properties of semantic embedding spaces (effective dimensionality Df, eigenvalue spectrum shape, spectral decay exponent alpha, gating thresholds) are stable under perturbation -- including noise injection, model variation, domain shift, and training data changes. Specifically, R-based gating decisions are robust to realistic perturbations of the embedding space.

## v1 Evidence Summary

The main Q46 document (23 lines) has status OPEN with "No dedicated research yet." However, a report file exists that describes "Three Laws of Geometric Stability" derived from the Feral Resident experiments:

- **Law 1 (Mass Accumulation):** 1/N averaging for memory updates prevents identity drift. Implemented in `geometric_memory.py`.
- **Law 2 (Critical Resonance):** 1/(2*pi) as the percolation threshold between signal and noise.
- **Law 3 (Nucleation Dynamics):** Threshold theta(N) = (1/(2*pi)) / (1 + 1/sqrt(N)) allows bootstrapping from cold start.

The report claims these derive from the Living Formula R = (E / grad_S) * sigma^Df, but provides no quantitative stability analysis.

## v1 Methodology Problems

The Phase 6D verdict noted:

1. **No actual stability analysis exists:** The main file correctly states OPEN. The report file describes engineering heuristics (1/N averaging, 1/(2*pi) threshold) but does not measure stability of geometric properties under perturbation.
2. **Critical gap identified:** sigma^Df is exponentially sensitive to sigma (a 3.7% change in sigma produces a 4.4x change in sigma^22; at Df=43.5, a 11% change produces 1000x). This is the most urgent numerical stability problem in the framework, and Q46 is the natural home for addressing it.
3. **No perturbation experiments:** No noise injection, no adversarial perturbation, no domain shift testing, no training data ablation.
4. **The "three laws" are engineering heuristics:** 1/N averaging is a running mean. 1/(2*pi) is an arbitrary threshold. These are useful engineering choices but not "invariant laws derived from the Living Formula."
5. **The question is well-posed and practically important:** The Phase 6D reviewer noted this is "potentially one of the most practically important" open questions in the framework.

## v2 Test Plan

### Experiment 1: Sigma Sensitivity Analysis

Quantify how sensitive R, Df, and alpha are to perturbations of sigma.

- **Method:** For 5 embedding models, compute sigma on a standard word set. Then perturb sigma by +/- 1%, 2%, 5%, 10%, 20%. Measure the resulting change in R, in sigma^Df, and in gating decisions (R > threshold?).
- **Analysis:** Plot |delta R / R| vs |delta sigma / sigma| as a function of Df. Derive the condition number of R with respect to sigma: kappa = |Df * sigma^(Df-1)|. Report the sigma range where R is practically usable (condition number < 100).
- **Key question:** For what range of sigma and Df is R stable enough for binary gating decisions?

### Experiment 2: Noise Injection Stability

Measure how geometric properties (alpha, Df, eigenvalue spectrum) change under additive Gaussian noise.

- **Method:** For 5 embedding models, inject Gaussian noise at SNR levels of 40dB, 30dB, 20dB, 10dB, 0dB. At each level, recompute alpha, Df, and the top-20 eigenvalue spectrum.
- **Data:** 5,000 word embeddings per model
- **Analysis:** Plot each geometric quantity vs SNR. Identify the SNR threshold below which geometric properties break down. Compare stability across models.
- **Key question:** How much noise can embedding geometric properties tolerate?

### Experiment 3: Domain Shift Stability

Measure how geometric properties change when the word set shifts between domains.

- **Method:** Compute alpha, Df, and eigenvalue spectrum for domain-specific word sets: (a) general English, (b) medical terminology, (c) legal terminology, (d) technical/code terms, (e) informal/slang
- **Data:** 1,000 domain-specific words per category, same embedding model for all
- **Analysis:** Compare geometric properties across domains. Compute the coefficient of variation of each property across domains.
- **Key question:** Are geometric properties stable across semantic domains, or do they vary with content?

### Experiment 4: Cross-Model Stability

Measure how geometric properties vary when the same text is embedded by different models.

- **Method:** Embed identical word sets using 10+ models of varying architecture and training objective. Compute alpha, Df, and eigenvalue spectrum for each.
- **Data:** 5,000 shared words across all models
- **Analysis:** Report CV of alpha, Df, and top eigenvalue ratios across models. Determine which properties are model-invariant and which are model-dependent.
- **Key question:** Which geometric properties are universal vs model-specific?

### Experiment 5: Gating Decision Robustness

Test whether binary gating decisions (R > threshold) are robust to realistic perturbations.

- **Method:** For 1,000 text pairs, compute R using the formula. Then perturb the embeddings (noise, model switch, vocabulary change) and recompute R. Measure the fraction of pairs where the gating decision flips.
- **Analysis:** Report the "flip rate" as a function of perturbation magnitude. Identify the perturbation level at which > 5% of gating decisions change.
- **Key question:** Is R-based gating robust enough for practical use?

## Required Data

- Pre-trained models: 10+ models spanning sentence-transformers, BERT variants, GloVe, Word2Vec
- Domain-specific word lists: 1,000 words each from medical, legal, technical, informal domains
- General vocabulary: 5,000 frequency-balanced words
- Noise generation: Gaussian noise at calibrated SNR levels

## Pre-Registered Criteria

- **Success (confirm stability):** Alpha varies by < 10% CV across domains AND < 15% CV across models AND gating decisions flip for < 5% of pairs at SNR > 20dB AND sigma condition number is < 100 for standard operating ranges
- **Failure (falsify stability):** Alpha varies by > 25% CV across domains OR gating flip rate > 20% at SNR = 20dB OR sigma condition number > 1000 for typical Df values (20-60)
- **Inconclusive:** Intermediate values; some properties stable and others not; stability depends on specific model or domain

## Baseline Comparisons

- **Random embedding stability:** Apply the same perturbations to random embeddings and measure how their geometric properties change (establishes whether trained embeddings are MORE stable than random, not just stable in absolute terms)
- **Cosine similarity stability:** Compare stability of R-based gating to simple cosine similarity threshold gating under the same perturbations
- **Theoretical bounds:** For Gaussian noise, derive analytical sensitivity of eigenvalue decay exponent alpha to noise level (Marchenko-Pastur theory provides this)

## Salvageable from v1

- The 1/N averaging heuristic for memory updates is a reasonable engineering baseline
- The observation that 1/(2*pi) ~ 0.159 functions as a reasonable threshold is worth testing systematically
- The nucleation dynamics idea (threshold varies with N) is an interesting hypothesis worth formalizing
- The identification of sigma^Df sensitivity as the critical issue (from Q29/Phase 3 verdicts) provides clear motivation
- Code: `geometric_memory.py` (1/N averaging), `feral_daemon.py` (dynamic threshold)
