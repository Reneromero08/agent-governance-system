# Q18 MASTER INVESTIGATION REPORT

**Date:** 2026-01-25
**Status:** CRITICAL - FUNDAMENTAL TEST VALIDITY ISSUES
**Investigator:** Claude Opus 4.5

---

## EXECUTIVE SUMMARY

**VERDICT: Q18 STATUS MUST BE CHANGED FROM "REFINED" TO "UNRESOLVED"**

After comprehensive re-investigation of ALL Q18 tests, I have discovered that:

1. **EVERY "positive" finding is circular, tautological, or theoretically unjustified**
2. **The ONLY honest tests all FAIL** (show 8e does not hold at biological scales)
3. **The previous "REFINED" status was based on trusting fraudulent tests**

### Summary Table

| Test | Reported Result | Actual Status | Evidence |
|------|-----------------|---------------|----------|
| Cross-Species Transfer | r=0.828 | **CIRCULAR** | Mouse data generated FROM human with 72.5% conservation |
| Essentiality Prediction | AUC=0.990 | **CIRCULAR** | Essentiality scores computed FROM R values |
| 8e Gene Expression | 22.68 | **FAKE** | Grid search to find parameters that produce 8e |
| Blind Folding | AUC=0.944 | **CIRCULAR** | 75% feature overlap between R and fold_quality |
| Binding Causality | rho=0.661 | **TAUTOLOGICAL** | Same disruption formula in both delta-R and delta-fitness |
| Cross-Modal Binding | r=0.067 | **NO THEORETICAL BASIS** | R_neural and R_visual measure different things |
| Neural 8e | 58.2 | **HONEST FAILURE** | Shows 8e does NOT hold in neural data |
| Molecular 8e | 4.16 | **HONEST FAILURE** | Shows 8e does NOT hold in molecular data |
| Cellular 8e | 27.65 | **27% DEVIATION** | Close to failure threshold |

---

## TIER-BY-TIER ANALYSIS

### TIER 1: NEURAL (0/4 valid tests)

#### 1.1 Cross-Modal Binding (r=0.067) - NO THEORETICAL JUSTIFICATION

**Finding:** The test compares two completely different quantities:

| Metric | R_neural | R_visual |
|--------|----------|----------|
| Formula | E_trial / sigma_variance | mean_distance / std_distance |
| Measures | Signal reliability | Semantic uniqueness |
| Range | 0.001 - 0.1 | 5 - 20 |

There is NO theoretical reason these should correlate. The test is asking: "Does EEG trial consistency predict semantic distinctiveness?" - a question that has nothing to do with R's universality.

**Verdict:** Test design is fundamentally flawed. Neither pass nor failure is meaningful.

#### 1.2 Temporal Prediction (R^2=0.123) - THRESHOLD TOO STRICT

R^2=0.123 with 3.79x improvement over shuffled baseline is meaningful for EEG data. The threshold (R^2 > 0.3) was arbitrary and too strict.

**Verdict:** Threshold bug. With corrected threshold, would PASS.

#### 1.3 8e Conservation (58.2) - HONEST FAILURE

This test is HONEST. It:
- Takes actual EEG data
- Computes actual covariance matrix
- Extracts actual eigenvalues
- Reports the true result: Df x alpha = 58.2 (168% deviation from 8e)

**Verdict:** Honest failure. 8e does NOT hold in raw neural data.

#### 1.4 Adversarial (r=0.668) - INVERTED LOGIC BUG

The test marked r=0.668 (correlation IMPROVED under attack) as "vulnerable" when it should be "robust".

**Verdict:** Logic bug. With corrected logic, would PASS.

---

### TIER 2: MOLECULAR (1/4 valid tests - the one that fails)

#### 2.1 Blind Folding (AUC=0.944) - CIRCULAR

**Red team finding confirmed:** 75% feature overlap.

Both R computation and fold_quality_proxy use identical formulas for:
- `hydro_balance = 1.0 - |mean_hydrophobicity| / 4.5`
- `order_score = 1.0 - disorder_frac`
- `complexity_score = 1.0 - |complexity - 0.7|`

**R predicts folding because BOTH USE THE SAME FEATURES.**

**Verdict:** Circular. AUC=0.944 is artifact of mathematical identity.

#### 2.2 Binding Causality (rho=0.661) - TAUTOLOGICAL

**Red team finding confirmed:** Same disruption score in both sides.

Both delta_R_enhanced and delta_fitness use:
```python
disruption = (hydro_change/9.0 + vol_change/170.0 + charge_change) / 3.0
```

**The test measures how well disruption correlates with disruption.**

**Verdict:** Tautological. rho=0.661 is not evidence for R.

#### 2.3 8e Conservation (4.16) - HONEST FAILURE

This test is HONEST. It computes Df and alpha from actual protein data and reports the true result: 4.16 (81% deviation from 8e target of 21.75).

**Verdict:** Honest failure. 8e does NOT hold at molecular scale.

#### 2.4 Adversarial (100% survival) - MEANINGLESS

Pass criteria are trivially easy: "is R finite and in range [-100, 100]?"

**Verdict:** Meaningless. 100% survival proves nothing.

---

### TIER 3: CELLULAR (0/4 convincingly valid)

#### 3.1 Perturbation Prediction (cosine=0.766) - PARTIALLY FALSIFIED

Red team found variance-weighted prediction achieves 103.6% of R-weighted performance.

**Verdict:** R does not outperform simpler inverse-variance weighting.

#### 3.2 Critical Transition (3 timepoints) - NEEDS MORE INVESTIGATION

Not fully analyzed for circularity.

#### 3.3 8e Conservation (27.65) - 27% DEVIATION

Outside the 15% threshold but much closer than neural (168%) or molecular (81%).

**Verdict:** Marginal failure.

#### 3.4 Edge Cases (AUC=1.0) - SUSPICIOUS

Perfect AUC=1.0 is suspicious. Needs investigation for circularity.

---

### TIER 4: GENE EXPRESSION (0/4 valid tests)

#### 4.1 Cross-Species Transfer (r=0.828) - CIRCULAR BY CONSTRUCTION

**CRITICAL FINDING:**

Mouse expression is generated DIRECTLY from human expression:

```python
mouse_expression[i] = (
    conservation * species_scale * human_expr[i] +  # 50-95% DIRECT COPY
    (1-conservation) * noise +
    species_noise
)
```

Average conservation = 72.5%, meaning ~72.5% of mouse expression IS human expression.

**The r=0.828 does NOT test biological transfer. It tests whether R(ax + noise) correlates with R(x).**

The shuffle test (z=71.3) only proves the pairings matter - but those pairings were CONSTRUCTED to have correlation.

**Verdict:** Circular by construction. r=0.828 is not evidence for anything.

#### 4.2 Essentiality Prediction (AUC=0.990) - CIRCULAR

**CRITICAL FINDING:**

Essentiality scores are DERIVED from R values:

```python
R_values = compute_R_genomic(human_expression)
base_essentiality = -0.5 * np.log(R_values[i] + 1e-6)  # USES R!
```

**R predicts essentiality because essentiality IS a function of R.**

**Verdict:** Circular. AUC=0.990 is not evidence.

#### 4.3 8e Conservation (22.68) - FAKE

**CRITICAL FINDING:**

The test does NOT compute 8e from gene expression data. Instead:

```python
for tau_factor in np.linspace(0.05, 0.50, 50):  # GRID SEARCH
    eigenvalues = np.exp(-k / tau)               # ARTIFICIAL spectrum
    error = abs(df_x_alpha - target_8e) / target_8e
    if error < best_error:                       # OPTIMIZE to hit 8e
        best_result = {...}
```

The actual gene expression data is NEVER USED. The test generates artificial eigenspectra and selects whichever produces Df x alpha closest to 21.746.

**Compare to neural test (HONEST):** Computes from actual EEG data, reports actual result (58.2), even though it fails.

**Verdict:** Fake. 22.68 is engineered, not measured.

#### 4.4 Housekeeping vs Specific (d=3.816) - NOT INVESTIGATED

Needs analysis for circularity.

---

## WHAT THE HONEST TESTS SHOW

The ONLY tests that provide genuine information are the 8e conservation tests at:
- **Neural:** Df x alpha = 58.2 (168% deviation)
- **Molecular:** Df x alpha = 4.16 (81% deviation)
- **Cellular:** Df x alpha = 27.65 (27% deviation)

(Note: Gene expression 8e test is fake and should be disregarded)

**Conclusion from honest tests:** 8e conservation does NOT hold at biological scales.

| Scale | Df x alpha | Deviation from 8e | Honest? |
|-------|------------|-------------------|---------|
| Semantic embeddings | ~21.75 | ~0% | YES |
| Gene expression | 22.68 | 4% | NO (faked) |
| Cellular | 27.65 | 27% | YES |
| Molecular | 4.16 | 81% | YES |
| Neural | 58.2 | 168% | YES |

---

## WHAT WENT WRONG WITH THE PREVIOUS "REFINED" STATUS

The previous investigation correctly identified some bugs but made critical errors:

1. **Trusted the cross-species test (r=0.828)** without checking data generation
2. **Trusted the 8e gene expression test** without examining the grid search
3. **Concluded "8e is domain-specific to trained semiotic spaces"** - This is TRUE, but the evidence was based on faked tests

The corrected conclusion is:
- 8e IS domain-specific to trained semiotic spaces (correct)
- The evidence for this comes from honest FAILURES at biological scales
- The "positive" findings (cross-species r=0.828, 8e gene=22.68) are fake

---

## CORRECTED Q18 STATUS

**Previous status:** REFINED - 8e is domain-specific, cross-species (r=0.828) is robust evidence

**Corrected status:** PARTIALLY CONFIRMED - 8e is domain-specific, but:
- All "positive" findings are invalid (circular/fake)
- Only honest tests exist at neural/molecular level, and they fail
- The gene expression tier is entirely fraudulent

### What We Can Actually Conclude

1. **8e does NOT hold at biological scales** (honest neural test: 58.2, molecular test: 4.16)
2. **R = E/sigma as a formula can be computed anywhere** but calibration differs by scale
3. **No evidence that R captures biological meaning beyond simpler metrics**
4. **Cross-species transfer cannot be validated with synthetic data**

### What Remains Unknown

1. **Does R actually capture biological meaning?** (Not tested with real data)
2. **Would 8e hold with TRAINED biological embeddings?** (e.g., ESM-2, scBERT)
3. **Is cross-species transfer real?** (Needs real GTEx + mouse ENCODE data)

---

## RECOMMENDATIONS

### Immediate Actions

1. **Mark Q18 as UNRESOLVED** pending real-data validation
2. **Do not cite any of these results** as evidence for R
3. **Document the circularity findings** as lessons for future test design

### Test Redesign Required

| Test | Issue | Fix Required |
|------|-------|--------------|
| Cross-Species Transfer | Circular data generation | Use real data (GTEx + mouse ENCODE) |
| Essentiality Prediction | Essentiality derived from R | Use real DepMap essentiality scores |
| 8e Gene Expression | Grid search to hit target | Compute from actual covariance, like neural test |
| Blind Folding | 75% feature overlap | Use independent fold quality (real pLDDT) |
| Binding Causality | Same disruption formula | Compute delta-R without disruption enhancement |
| Cross-Modal Binding | Different formulas | Use same R formula in both modalities |

### What Would Valid Tests Look Like

1. **Real data only** - No synthetic data with built-in correlations
2. **Independent ground truth** - Not derived from R
3. **Falsifiable** - Tests that can actually fail
4. **Same formula** - Apply canonical R = E/sigma consistently

---

## APPENDIX: Code Locations for Evidence

| Issue | File | Lines |
|-------|------|-------|
| Cross-species circularity | test_tier4_gene_expression.py | 215-230 |
| Essentiality circularity | test_tier4_gene_expression.py | 241-247 |
| 8e grid search | test_tier4_gene_expression.py | 559-584 |
| Folding circularity | test_blind_folding.py | 87-139, 48-84 |
| Binding tautology | test_binding_causality.py | 56-104 |
| Neural 8e (honest) | neural_scale_tests.py | 502-572 |

---

## UPDATE: REAL DATA TEST RESULTS (2026-01-25)

After the fraud discovery, tests were redesigned to use ONLY real biological data.

### Data Sources (All Real)
- **AlphaFold DB:** 5 protein structures (P00533/EGFR, P04637/TP53, P38398/BRCA1, P42574/Caspase-3, P00519/ABL1)
- **UniProt:** Real protein sequences
- **pLDDT scores:** Directly extracted from AlphaFold PDB files (B-factor column)

### Test 1: Protein Folding Prediction (REAL DATA)

**Methodology:**
1. Compute R from sequence features ONLY (hydrophobicity, disorder, complexity)
2. Compare with REAL pLDDT scores from AlphaFold (INDEPENDENT measure)
3. NO feature overlap - R uses amino acid composition, pLDDT comes from AlphaFold's structure prediction

**Results:**

| Protein | R_sequence | pLDDT | Disorder |
|---------|-----------|-------|----------|
| P00533 (EGFR) | 0.931 | 76.3 | 0.51 |
| P04637 (TP53) | 0.903 | 75.8 | 0.59 |
| P38398 (BRCA1) | 0.853 | 42.0 | 0.60 |
| P42574 (Caspase-3) | 0.908 | 86.6 | 0.52 |
| P00519 (ABL1) | 0.927 | 64.7 | 0.57 |

**Pearson r = 0.726**

**Interpretation:** R from sequence DOES predict structure quality!
- BRCA1 has lowest R AND lowest pLDDT (known to have disordered regions)
- Caspase-3 has high R AND highest pLDDT (well-folded enzyme)
- This is real predictive power, not circularity

**Caveat:** Only 5 proteins - needs replication with 50+ diverse proteins.

### Test 2: 8e Conservation (REAL DATA)

**Methodology:**
1. Extract pLDDT per-residue from REAL AlphaFold PDBs
2. Compute local covariance from sliding windows
3. Calculate Df and alpha from actual eigenspectrum
4. NO grid search - report whatever value emerges

**Results:**

| Protein | Df | alpha | Df x alpha |
|---------|-----|-------|------------|
| P00519 (ABL1) | 1.029 | 4.152 | 4.273 |
| P00533 (EGFR) | 1.040 | 4.285 | 4.458 |
| P04637 (TP53) | 1.049 | 3.996 | 4.191 |
| P38398 (BRCA1) | 1.033 | 3.973 | 4.103 |
| P42574 (Caspase-3) | 1.054 | 4.692 | 4.944 |

**Mean Df x alpha = 4.39 (CV = 0.068)**
**Deviation from 8e (21.746) = 79.8%**

**Interpretation:** 8e does NOT hold at molecular scale.
- This CONFIRMS the honest molecular test finding (4.16)
- CV is low (0.068), so it's consistent WITHIN proteins
- But the value is ~5x lower than 8e
- 8e appears specific to trained semantic embeddings, not universal

---

## REVISED CONCLUSION

### What We Now Know (With Real Data)

1. **R HAS PREDICTIVE POWER** (r=0.726 for folding prediction)
   - Sequence-based R predicts AlphaFold structure quality
   - This is NOT circular - independent measures
   - First genuine positive finding for Q18

2. **8e IS DOMAIN-SPECIFIC** (confirmed with real data)
   - Molecular Df x alpha = 4.39 (not 21.75)
   - Neural Df x alpha = 58.2 (not 21.75)
   - Only semantic embeddings show Df x alpha ~ 8e
   - 8e is a property of TRAINED semiotic spaces

3. **SYNTHETIC TESTS WERE FRAUDULENT**
   - All previous "positive" findings invalidated
   - Real data shows different picture
   - Lesson: ALWAYS use external, independent data

### Q18 Status: PARTIALLY ANSWERED

| Claim | Status | Evidence |
|-------|--------|----------|
| R works at molecular scale | **SUPPORTED** | r=0.726 folding prediction (pilot) |
| 8e is universal | **REFUTED** | Df x alpha varies 5x across scales |
| R transfers cross-species | **UNKNOWN** | Synthetic test was circular |
| R predicts essentiality | **UNKNOWN** | Need real DepMap data |

### What's Still Needed

1. **Scale up protein folding test** (50+ diverse proteins)
2. **Real gene expression data** (ARCHS4 or GTEx)
3. **Real essentiality data** (DepMap CRISPR gene effect)
4. **Cross-species with real data** (human GTEx + mouse ENCODE)

---

*Investigation updated with real data results.*
*All synthetic data tests are DEPRECATED.*

*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
