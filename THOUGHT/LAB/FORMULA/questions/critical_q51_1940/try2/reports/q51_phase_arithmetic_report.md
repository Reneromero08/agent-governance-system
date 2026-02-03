# Q51 Try2 Report: Phase Arithmetic Validity

## Scope
This report addresses the prompt “Phase Arithmetic Validity” using external data and a global PCA projection to the first two PCs with phase defined as `theta = atan2(PC2, PC1)`.

## Formal Assumptions Required
1. Global PCA is fit once on a fixed vocabulary, producing a shared 2D basis for all words.
2. Analogy structure is evaluated only in this shared 2D coordinate system.
3. Phase is well-defined for all words used, meaning projected points are not near the origin.
4. Phase differences are computed with correct circular wrapping in [-π, π].
5. The dataset provides independent analogy ground truth, not derived from phase values.
6. No parameter tuning after observing results.

## What Would Be Implied vs Empirical
The construction itself only guarantees a linear projection. It does not logically imply phase-difference preservation. The phase relation is therefore empirical under these assumptions and must be tested.

## Hidden Assumptions / Coordinate Dependencies
1. PCA basis depends on the global vocabulary distribution.
2. Mean-centering defines the origin; phase is not translation-invariant.
3. Near-zero projections destabilize phase.
4. Any change in corpus or PCA fit changes all phases simultaneously.

## Method Summary
1. External dataset: Google Analogy Test Set `questions-words.txt`.
2. Train/test split: 80/20 with fixed seed.
3. Global PCA fit on training vocabulary only.
4. Phase computed for all words in the shared 2D space.
5. Error metric: `| (θ_b − θ_a) − (θ_d − θ_c) |` wrapped to [-π, π].
6. Pass threshold: π/4.

## Results Summary
Results are recorded in:
- `THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/reports/report.md`
- `THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/results/results.json`

All 19 evaluated models passed the predefined thresholds in this run.

## Conclusion
The phase-difference relation is not mathematically implied by PCA construction. It is supported empirically in this external-data run under the stated assumptions. The evidence is repeatable only within the fixed PCA basis and dataset choices.

## Limitations
1. PCA basis depends on vocabulary choice.
2. Phase is unstable for low-norm projections.
3. Results are specific to the analogy dataset and model set used.
