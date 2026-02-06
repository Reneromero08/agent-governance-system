# Q4: The Formula Makes Novel Falsifiable Predictions

## Hypothesis

The Living Formula R = (E / grad_S) * sigma^Df makes novel, testable predictions that (a) are not already known from basic statistics or standard signal-to-noise ratio theory, (b) are specific to the formula rather than any generic noise metric, (c) can be stated before testing, and (d) can be validated on real external data.

Concretely: there exists at least one prediction derivable from R that a simpler metric (e.g., 1/std, raw cosine similarity, standard SNR) does NOT make, and this prediction is confirmed on held-out real-world data.

## v1 Evidence Summary

Four predictions were tested, all on fully synthetic data:

| Prediction | Result | Numbers |
|------------|--------|---------|
| Low R predicts need for more context | WEAK SUPPORT | r = -0.11 |
| High R = faster convergence | CONFIRMED | 5.0 vs 12.4 samples |
| R threshold transfers across domains | CONFIRMED | Works on 2 synthetic noise distributions |
| R-gating improves decisions | CONFIRMED | 83.8% -> 97.2% accuracy |

One external-data test was also conducted (brain-stimulus Df matching on THINGS-EEG): NO SIGNIFICANT CORRELATION (max |r| = 0.109, p = 0.266).

A post-hoc "Riemann prediction" (alpha ~ 1/2) was added from Q48-Q50, claiming mean alpha = 0.5053 across 5 models.

## v1 Methodology Problems

The Phase 5 verification found severe problems across all claimed predictions:

1. **All four "confirmed" predictions are restatements of basic statistics.** The test code defines R as approximately 1/std^2 (lines 20-26 of q4_novel_predictions_test.py). Every "prediction" follows trivially from this: low noise -> faster convergence (CLT, known since 1733), filtering by quality improves quality (selection effect tautology), std-based metrics transfer across symmetric distributions.

2. **Test R does not match the actual formula.** The test uses E = 1/(1+std) and grad_S = std(observations), with sigma=0.5 and Df=1.0 hardcoded. This bears no resemblance to the GLOSSARY-defined R using cosine similarity and eigenvalue spectra.

3. **Zero pre-registration.** All predictions and tests appear in the same file with no evidence that predictions preceded data collection. Thresholds were set at the weakest possible level (e.g., r < -0.1 for Prediction 1, which barely cleared at r = -0.11).

4. **The only real-data test failed.** The brain-stimulus Df matching on THINGS-EEG produced a null result, then the report offered 5 post-hoc excuses for the failure without acknowledging it as negative evidence.

5. **The Riemann prediction is post-hoc.** The document itself states alpha ~ 1/2 "was NOT expected from the original formula," making it an observation attributed retroactively, not a prediction. Phase 4 independently rated Q48-Q50 as EXPLORATORY.

6. **No baseline comparison.** None of the four tests compared R to simpler alternatives (1/std, raw E, standard SNR). It is entirely possible that 1/std outperforms R on every test.

Verdict recommended downgrade from PARTIAL to EXPLORATORY, R from 1700 to 400-600.

## v2 Test Plan

### Phase 1: Identify Candidate Novel Predictions

Before any testing, derive at least 3 specific, falsifiable predictions from R = (E / grad_S) * sigma^Df (using the GLOSSARY-defined formula with actual cosine similarity, not a proxy) that are NOT trivially derivable from:
- Standard SNR (mean/std)
- Raw mean cosine similarity (bare E)
- Sample variance alone

Each prediction must be documented with:
- The mathematical derivation from the formula
- Why the prediction is specific to R (why bare E or 1/std would not make the same prediction)
- The precise numerical criterion for confirmation/falsification

### Phase 2: Pre-Registered Testing on Real Data

For each candidate prediction:

1. **State the prediction precisely** (including direction, effect size threshold, and dataset) BEFORE running any code
2. **Select a real external dataset** appropriate for the prediction (see Required Data below)
3. **Implement the test** using GLOSSARY-defined R (cosine similarity E, proper grad_S)
4. **Run the identical test** using bare E, 1/std, and a random baseline
5. **Compare:** R must outperform all baselines to count as a novel prediction

### Phase 3: Cross-Domain Transfer Test

If any prediction survives Phase 2:
- Calibrate on Domain A, freeze all parameters
- Apply unchanged to Domain B (genuinely different domain, not just different noise shape)
- R must still outperform baselines on Domain B

## Required Data

- **STS-B** (Semantic Textual Similarity Benchmark) -- for similarity prediction tasks
- **SNLI / ANLI R3** -- for entailment/contradiction prediction tasks
- **THINGS-EEG** (OpenNeuro ds003825) -- for cross-modal prediction (brain-stimulus)
- **HistWords** (historical word embeddings) -- for temporal drift prediction
- **SHP** (Stanford Human Preferences) -- for preference prediction tasks
- At least 2 genuinely different domains (e.g., text + financial, or text + genomic)

## Pre-Registered Criteria

- **Success (confirm):** At least ONE prediction satisfies ALL of: (a) pre-registered before testing, (b) tested on real external data, (c) R outperforms bare E with Cohen's d > 0.5, (d) R outperforms 1/std with Cohen's d > 0.5, (e) effect replicates on a second dataset
- **Failure (falsify):** Zero predictions meet all five criteria above after testing at least 3 candidates across at least 2 datasets
- **Inconclusive:** At least one prediction meets criteria (a)-(d) but fails replication (e), or effect sizes are 0.3 < d < 0.5

## Baseline Comparisons

Every prediction must be tested against:
1. **Bare E** (mean pairwise cosine similarity, no normalization)
2. **1/grad_S** (inverse standard deviation alone)
3. **Standard SNR** (E / grad_S without sigma^Df)
4. **Random baseline** (shuffled embeddings)

R must outperform ALL baselines to count as a genuinely novel prediction of the formula.

## Salvageable from v1

- **Brain-stimulus test methodology** from `v1/questions/critical_q04_1700/q4_novel_predictions/brain_stimulus_df_REPORT.md` -- the approach of testing against THINGS-EEG is sound, just needs proper execution with more subjects and corrected Df measures
- **The question itself** is well-posed and important -- "What does the formula predict that we don't already know?" is exactly the right question
- **v1 test code** at `v1/questions/critical_q04_1700/q4_novel_predictions/q4_novel_predictions_test.py` should be reviewed for structure only; the R implementation must be replaced with the GLOSSARY-defined formula
