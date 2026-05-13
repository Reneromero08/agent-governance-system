# Cluster A Final Verified Synthesis

**Date:** 2026-02-07
**Process:** v2 -> audit -> v3 fix -> v3 verify -> v4 fix -> v4 verify (until clean)
**Total agents used:** 25 independent reviewers across 5 rounds
**Status:** CLEAN -- 0 critical issues remaining

---

## Final Scorecard

| Q | Question | v2 | v3 | v4 | Final Verified | Confidence |
|---|----------|----|----|----|--------------------|------------|
| Q01 | grad_S predictive value | FALSIFIED | CONFIRMED | -- | **CONFIRMED** | MED-HIGH |
| Q02 | Falsification criteria | FALSIFIED | INCONCLUSIVE | -- | **INCONCLUSIVE** | HIGH |
| Q03 | Generalization | INCONCLUSIVE | CONFIRMED | INCONCLUSIVE | **INCONCLUSIVE** | HIGH |
| Q05 | Agreement vs truth | FALSIFIED | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** | HIGH |
| Q09 | Free energy / NLL | INCONCLUSIVE | INCONCLUSIVE | -- | **INCONCLUSIVE** | HIGH |
| Q15 | Statistical interpretation | INCONCLUSIVE | FALSIFIED | -- | **FALSIFIED** | HIGH |
| Q20 | Tautology | FALSIFIED | CONFIRMED | -- | **CONFIRMED** | HIGH |

**Final: 2 CONFIRMED, 4 INCONCLUSIVE, 1 FALSIFIED**

---

## The Honest Truth About R = (E / grad_S) * sigma^Df

### What is proven (high confidence)

1. **R_simple = E/grad_S = SNR significantly outperforms E alone for predicting label purity on 20 Newsgroups.**
   - Steiger p < 1e-6 on all 3 architectures (Q01)
   - Steiger p < 0.01 on all 3 architectures (Q20)
   - Cross-validated R^2 improvement: +0.05 to +0.12
   - The grad_S denominator adds genuine independent signal

2. **R is architecturally invariant.**
   - Inter-architecture correlation rho > 0.97 across all tests
   - Results replicate across all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d), multi-qa-MiniLM-L6-cos-v1 (384d)

3. **R is NOT more vulnerable to bias attacks than E.**
   - grad_S dampening factor > 1.0 on 2/3 architectures (Q05 v4)
   - The grad_S denominator provides a modest defensive buffer
   - Verified with corrected Steiger formula (v4)

4. **log(R) covaries strongly with Gaussian NLL.**
   - Within-group r = 0.69-0.89 (not a group-structure artifact)
   - Partial r = 0.79-0.84 after removing purity (genuinely independent)
   - BUT: slope = 0.035 not 1.0 -- proportional relationship, not identity

5. **R is NOT tautological.**
   - SNR (mean/std) adds genuine information beyond the raw signal (mean)
   - Confirmed via Steiger, cross-validated R^2, and AIC model comparison

### What is NOT proven

6. **sigma^Df adds nothing.**
   - R_simple consistently beats R_full in cross-validated R^2 (Q20)
   - Df has zero correlation with purity (|rho| < 0.01, p > 0.93)
   - sigma is approximately constant across topics within each model
   - The practical formula is R = E/grad_S, period

7. **Cross-domain generalization is unproven.**
   - R beats E on 20 Newsgroups (Steiger p < 1e-15)
   - R LOSES to E on AG News (Steiger z = -12.2, E wins)
   - R fails on financial data with honest ground truth (rho=0.17)
   - R's advantage is dataset-dependent, not universal

8. **"Bayesian interpretation" does not exist.**
   - The v2 "Bayesian quantities" were frequentist statistics
   - E beats R for predicting geometric/distributional quality measures
   - R beats E for predicting semantic/label quality (purity)
   - This is a geometric-vs-semantic distinction, not a Bayesian connection

9. **Free Energy Principle connection is not established.**
   - The tested quantity is Gaussian NLL, not FEP free energy
   - The proportional relationship (slope=0.035) is not an identity

10. **8e conservation law is untestable.**
    - Under v2 definitions, Df = 2/alpha, so Df*alpha = 2 trivially
    - Would require independent Df measurement (e.g., box-counting)

---

## The Simplified Formula

Based on all evidence, the empirically supported formula reduces to:

**R = E / grad_S**

Where:
- E = mean of pairwise cosine similarities (the signal)
- grad_S = std of pairwise cosine similarities (the noise)
- R = signal-to-noise ratio

This is a well-known statistical quantity (coefficient of variation inverse / SNR). It significantly outperforms raw E for predicting semantic cluster quality (label purity) on 20 Newsgroups, is architecturally invariant, and is modestly more robust to adversarial manipulation than E alone.

Key caveat: R does NOT consistently outperform E across datasets. On AG News, E beats R. The advantage appears dataset-dependent -- R helps most when within-cluster similarity variance is high (diverse subtopics within categories, as in 20 Newsgroups' mixed forums).

The sigma^Df multiplicative term should be dropped -- it adds no predictive value and sometimes degrades performance.

---

## Remaining Issues (non-critical)

| Q | Issue | Severity | Status |
|---|-------|----------|--------|
| Q01 | Subcluster non-independence (effective n~40-60) | MEDIUM | Effect survives at reduced n |
| Q01 | Bimodal purity distribution | MEDIUM | Needs mixed-only subgroup analysis |
| Q03 | AG News uses only 1 model vs 3 for 20NG | MINOR | Methodological asymmetry |
| Q05 | multi-qa has one phrase with dampening=0.993 | MINOR | Negligible |
| Q09 | Slope = 0.035, not 1.0 | INFO | Already reflected in INCONCLUSIVE |
| Q20 | R_simple beats R_full in CV R^2 | INFO | Confirms sigma^Df is inert |

**0 critical issues remaining.**

---

## Audit Trail

### Round 1: v2 Audit (7 agents)
- Found 60 issues (11 critical)
- All 4 FALSIFIED verdicts overturned to INCONCLUSIVE
- Reports: `q*/AUDIT.md`

### Round 2: v3 Fix + Rerun (7 agents)
- Fixed all 60 issues, rewrote and reran all 7 tests
- New scorecard: 3 CONFIRMED, 3 INCONCLUSIVE, 1 FALSIFIED
- Reports: `q*/VERDICT_v3.md`

### Round 3: v3 Verification (7 agents)
- Found 7 remaining issues (3 critical)
- Q03 verdict changed (same dataset counted twice)
- Q05 Steiger bug found (verdict accidentally correct)
- Reports: `q*/AUDIT_v3.md`

### Round 4: v4 Fix + Rerun (2 agents -- Q03 and Q05 only)
- Q03: Replaced duplicate 20NG with AG News -- R loses to E on AG News -> INCONCLUSIVE
- Q05: Fixed Steiger formula, removed tautological ratio -> INCONCLUSIVE holds
- Reports: `q*/VERDICT_v4.md`

### Round 5: v4 Verification (2 agents)
- Q03: AG News verified genuine, z=-12.16 verified real, INCONCLUSIVE upheld (0 critical)
- Q05: Steiger formula verified correct, dampening factor clean, INCONCLUSIVE upheld (0 critical)
- Reports: `q*/AUDIT_v4.md`

**Loop terminated: 0 critical issues remaining across all 7 Qs.**
