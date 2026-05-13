# VERDICT: Q03 v4 -- Cross-Domain Generalization

**Version:** v4
**Date:** 2026-02-06
**Verdict:** INCONCLUSIVE
**Test file:** `code/test_v4_q03.py`
**Results:** `results/test_v4_q03_results.json`

---

## Question

Does R = (E / grad_S) * sigma^Df generalize across domains?

Reframed (v3+): Since v2 uses a single E definition (mean pairwise cosine
similarity), the honest question is: "Does R (using cosine E) correlate with
domain-appropriate quality metrics across different data types, and does R
outperform E alone?"

## v4 Change from v3

**DOMAIN-01 [CRITICAL] from AUDIT_v3:** The v3 test counted two runs of
20 Newsgroups (with different random seeds) as two separate "text domains."
The audit correctly identified this as the same domain tested twice.

v4 replaces the second 20 Newsgroups run with **AG News** (HuggingFace
datasets), a genuinely different text corpus with 4 classes (World, Sports,
Business, Sci/Tech) and 120,000 training documents. This is NOT 20 Newsgroups
in any form.

## Pre-Registered Decision Rule

- **CONFIRMED:** R significantly outperforms E (Steiger p<0.05) in >= 2/3 domains
- **FALSIFIED:** R does not significantly outperform E in any domain
- **INCONCLUSIVE:** otherwise

## Domain Results

### Domain 1: 20 Newsgroups (text_20ng)

| Metric | Rho vs Purity | p-value |
|--------|--------------|---------|
| R_full | 0.9497 | 1.77e-33 |
| R_simple | 0.9484 | 3.86e-33 |
| E | 0.9267 | 1.83e-28 |

- Steiger R_full vs E: z=16.61, p<1e-15
- R_full - E = +0.023
- **R significantly beats E: YES**
- Tested across 3 embedding models; result consistent (z > 10 for all)

### Domain 2: AG News (text_agnews)

| Metric | Rho vs Purity | p-value |
|--------|--------------|---------|
| R_full | 0.9083 | 1.48e-20 |
| R_simple | 0.9065 | 2.35e-20 |
| E | 0.9306 | 1.78e-23 |

- Steiger R_full vs E: z=-12.16, p<1e-15
- R_full - E = **-0.022**
- **R significantly beats E: NO (E beats R)**

This is the critical finding. On AG News, E alone is a *better* predictor of
cluster purity than R. The Steiger test is significant but in the WRONG
direction -- E outperforms R, not the other way around.

This means the v3 "CONFIRMED" verdict was an artifact of testing the same
dataset twice. When a genuinely different text corpus is used, R does not
consistently beat E.

### Domain 3: Financial (sector purity)

| Metric | Rho vs Purity | p-value |
|--------|--------------|---------|
| R_full | 0.173 | 0.336 |
| R_simple | 0.163 | 0.365 |
| E | 0.163 | 0.365 |

- Steiger R_full vs E: z=0.19, p=0.853
- R_full - E = +0.010
- **R significantly beats E: NO**
- Neither R nor E meaningfully predict sector purity (rho ~ 0.17, not significant)

## Scorecard

| Domain | Dataset | R beats E? | Steiger p<0.05? |
|--------|---------|-----------|-----------------|
| text_20ng | 20 Newsgroups | YES (+0.023) | YES (z=16.6) |
| text_agnews | AG News | NO (-0.022) | YES but WRONG direction |
| financial | Yahoo Finance | NO (+0.010) | NO (p=0.85) |

Domains where R significantly outperforms E: **1/3**
Required for CONFIRMED: 2/3

## Verdict: INCONCLUSIVE

Applying the pre-registered decision rule:
- 1/3 domains show R significantly beating E
- This is neither >= 2/3 (CONFIRMED) nor 0/3 (FALSIFIED)
- Therefore: **INCONCLUSIVE**

## Interpretation

R = E/grad_S = mean(cos)/std(cos) = SNR of cosine similarities.

On 20 Newsgroups, dividing by std(cos) improves purity prediction by a small
but statistically significant margin (+0.023 rho). On AG News, dividing by
std(cos) actually *hurts* purity prediction (-0.022 rho). On financial data,
neither metric predicts sector purity.

The AG News result is informative: AG News has 4 balanced classes (30k docs
each) compared to 20 Newsgroups' 20 classes of varying size. In a balanced
4-class setting, the noise injection procedure produces clusters where E
(mean similarity) is already an excellent predictor of purity. Dividing by
std(cos) introduces unnecessary noise from the variance estimate, degrading
prediction.

This suggests R's advantage over E is dataset-specific, not a general property
of embedding spaces. The generalization claim is not supported.

## SNR Verification

R_simple = SNR = mean(cos)/std(cos) confirmed across all 150 clusters
(max |R_simple - SNR| = 0.0).

## Elapsed Time

302.8 seconds
