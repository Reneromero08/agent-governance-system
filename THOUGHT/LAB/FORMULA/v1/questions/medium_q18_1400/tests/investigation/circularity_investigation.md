# Q18 Circularity Investigation Report

**Date:** 2026-01-25
**Investigator:** Claude Opus 4.5
**Subject:** Validation of Red Team Circularity Findings

## Executive Summary

This investigation analyzes the adversarial red team's claims that multiple Q18 tests suffer from circularity. After detailed code analysis, I find:

| Finding | Red Team Verdict | Investigation Verdict | Rationale |
|---------|------------------|----------------------|-----------|
| Essentiality Test | FALSIFIED | **VALID - TRUE CIRCULARITY** | Essentiality scores generated using R values |
| Folding Test | FALSIFIED | **PARTIALLY VALID** | Feature overlap exists but is scientifically expected |
| Mutation Test | FALSIFIED | **VALID - TRUE CIRCULARITY** | Both metrics derived from same disruption score |
| Cross-Species | ROBUST | **CONFIRMED ROBUST** | Shuffling test properly implemented |
| Perturbation | PARTIALLY_FALSIFIED | **VALID** | Variance weighting performs equivalently |

**Overall Assessment:** The red team findings are largely valid. Three of five tests have fundamental methodological flaws. The cross-species transfer test remains the only robust evidence for R's biological relevance.

---

## 1. Essentiality Test Analysis

### Red Team Claim
> "ESSENTIALITY TEST: Circular by construction - essentiality scores are GENERATED using R values"

### Code Evidence

From `test_tier4_gene_expression.py`, lines 236-261:

```python
# Generate essentiality scores
# Essential genes tend to have high R (tightly regulated, consistent expression)
# DepMap convention: lower score = more essential

essentiality_scores = np.zeros(n_genes)

# Compute R for essentiality correlation
R_values = compute_R_genomic(human_expression)

# Essentiality inversely correlated with R (high R -> low essentiality score -> essential)
# Add biological noise
for i in range(n_genes):
    # Base essentiality from R (negative correlation)
    base_essentiality = -0.5 * np.log(R_values[i] + 1e-6)  # <-- DIRECT USE OF R

    # Housekeeping genes more likely to be essential
    if i in housekeeping_genes:
        base_essentiality -= rng.uniform(0.5, 1.5)

    # Tissue-specific genes less likely to be essential (for organism survival)
    if i in tissue_specific_genes:
        base_essentiality += rng.uniform(0.5, 1.5)

    # Add noise
    essentiality_scores[i] = base_essentiality + rng.normal(0, 0.3)
```

### Investigation Finding: **TRUE CIRCULARITY CONFIRMED**

The circularity is unambiguous:

1. **R values are computed first** from expression data
2. **Essentiality scores are then generated** using the formula: `base_essentiality = -0.5 * np.log(R_values[i] + 1e-6)`
3. **The test then measures** whether R predicts essentiality

This is a textbook case of circular reasoning. The test achieves AUC=0.990 because essentiality was *defined* to correlate with R. The noise terms (housekeeping adjustment, random noise) partially obscure this but do not eliminate it.

### What Would a Non-Circular Test Look Like?

A valid essentiality test would require:

1. **Independent ground truth**: Use real essentiality data from sources like:
   - DepMap CRISPR knockout screens
   - Mouse knockout lethality databases
   - Human genetics (essential gene annotations from OMIM)

2. **Separate data sources**: Expression data from one source, essentiality labels from a completely independent experiment

3. **No formula overlap**: Ensure essentiality labels were determined without any knowledge of expression variability patterns

### Verdict: **RED TEAM CLAIM IS VALID**

---

## 2. Folding Test Analysis

### Red Team Claim
> "75% of fold quality factors (hydro_balance, order_score, complexity_score) are DIRECTLY used in R computation"

### Code Evidence

**From `test_blind_folding.py` - compute_fold_quality_proxy() (lines 48-84):**

```python
def compute_fold_quality_proxy(sequence: str, noise: float = 0.1, seed: int = 42) -> float:
    features = extract_protein_features(sequence)

    # Factor 1: Hydrophobic balance (optimal around 0)
    hydro_balance = 1.0 - abs(features.hydrophobicity_mean) / 4.5

    # Factor 2: Low disorder propensity
    disorder_aa = set("DEKRSPQGN")
    disorder_frac = sum(1 for aa in sequence if aa in disorder_aa) / len(sequence)
    order_score = 1.0 - disorder_frac

    # Factor 3: Moderate complexity (optimal around 0.7)
    complexity_score = 1.0 - abs(features.complexity - 0.7) * 2

    # Factor 4: Secondary structure propensity
    struct_score = min(1.0, helix_frac + sheet_frac + 0.3)

    # Combine factors
    quality = 0.3 * hydro_balance + 0.3 * order_score + 0.2 * complexity_score + 0.2 * struct_score
```

**From `test_blind_folding.py` - compute_R_from_sequence_family() (lines 87-139):**

```python
def compute_R_from_sequence_family(sequences: List[str], k: int = 5) -> float:
    embeddings = np.array([compute_sequence_embedding(seq) for seq in sequences])
    R_base, E, sigma = compute_R_molecular(embeddings, k=k)

    features_list = [extract_protein_features(seq) for seq in sequences]

    # 1. Conservation score
    conservation = 1.0 / (1.0 + embedding_variance)

    # 2. Order propensity (lower disorder = better folding)
    order_score = np.mean(disorder_scores)  # <-- SHARED

    # 3. Hydrophobic balance
    hydro_balance = 1.0 - abs(np.mean(hydro_means)) / 4.5  # <-- SHARED

    # 4. Complexity score
    complexity_score = 1.0 - abs(np.mean(complexities) - 0.7)  # <-- SHARED

    R_enhanced = (
        R_base * 0.3 +
        conservation * 0.25 +
        order_score * 0.25 +      # <-- SHARED
        hydro_balance * 0.1 +      # <-- SHARED
        complexity_score * 0.1     # <-- SHARED
    )
```

### Investigation Finding: **PARTIALLY VALID - BUT CONTEXT MATTERS**

The feature overlap is real:
- **hydro_balance**: Used in both (weight 0.3 in fold_quality, 0.1 in R_enhanced)
- **order_score**: Used in both (weight 0.3 in fold_quality, 0.25 in R_enhanced)
- **complexity_score**: Used in both (weight 0.2 in fold_quality, 0.1 in R_enhanced)

**However, this is a nuanced situation:**

#### Is Feature Overlap Always Circularity?

No. The key question is: *Do these features genuinely relate to both fold quality AND regulatory importance?*

**Arguments that this is NOT true circularity:**

1. **Both metrics capture real physics**: Hydrophobicity, disorder, and complexity genuinely affect both protein folding (biophysics) and the evolutionary signature captured by R (conservation patterns). Real proteins that fold well DO tend to have balanced hydrophobicity.

2. **R includes non-shared factors**: R_base (30%) and conservation (25%) are not in fold_quality_proxy. If the test were purely circular, we would expect these to add noise, not signal.

3. **fold_quality has a unique factor**: struct_score (helix/sheet propensity, 20%) is not in R computation.

**Arguments that this IS problematic:**

1. **The test design guarantees some correlation**: By construction, at least 45% of R's variance comes from the same factors as 80% of fold_quality. This inflates measured correlation.

2. **The inflation is not quantified**: We cannot know what portion of the AUC=0.944 comes from shared features vs. genuine predictive power.

3. **A cleaner test would separate these**: Use ONLY R_base and conservation for prediction, NOT the physicochemical features.

### What Would a Non-Circular Test Look Like?

Option A: **Use different features**
- Compute R using ONLY embedding-space metrics (E/sigma from kNN distances)
- Use fold_quality from external sources (PDB resolution, B-factors, disorder predictors)

Option B: **Use real data**
- R computed from sequence families
- Fold quality from actual experimental structure quality metrics

### Verdict: **RED TEAM CLAIM IS PARTIALLY VALID**

The circularity is real but the situation is more nuanced than "completely circular." The shared features reflect genuine biophysical relationships, but the test design makes it impossible to disentangle this from artificial correlation. The test should be redesigned.

---

## 3. Mutation Test Analysis

### Red Team Claim
> "The correlation rho=0.507 is TAUTOLOGICAL. Both delta-R and delta-fitness are computed from the same physicochemical disruption score"

### Code Evidence

**From `molecular_utils.py` - generate_dms_benchmark() (lines 355-412):**

```python
def generate_dms_benchmark(sequence: str, n_mutations: int = 100, seed: Optional[int] = None):
    # ...
    for _ in range(n_mutations):
        # Compute fitness effect
        # 1. BLOSUM penalty
        blosum_penalty = (blosum_wt - blosum_mut) / 10.0

        # 2. Physicochemical disruption
        hydro_change = abs(HYDROPHOBICITY.get(wt_aa, 0) - HYDROPHOBICITY.get(mut_aa, 0))
        vol_change = abs(VOLUMES.get(wt_aa, 100) - VOLUMES.get(mut_aa, 100))
        charge_change = abs(CHARGE.get(wt_aa, 0) - CHARGE.get(mut_aa, 0))

        phys_penalty = (hydro_change / 9.0 + vol_change / 170.0 + charge_change) / 3.0

        # Total fitness effect
        delta_fitness = -(blosum_penalty + phys_penalty + active_penalty + noise)
```

**From `test_binding_causality.py` - compute_delta_R_for_mutation() (lines 56-104):**

```python
def compute_delta_R_for_mutation(wild_type: str, mutation: MutationEffect, ...):
    # ... (compute base delta_r from embeddings) ...

    # Enhance with local physicochemical disruption metric
    hydro_change = abs(HYDROPHOBICITY.get(wt_aa, 0) - HYDROPHOBICITY.get(mut_aa, 0)) / 9.0
    vol_change = abs(VOLUMES.get(wt_aa, 100) - VOLUMES.get(mut_aa, 100)) / 170.0
    charge_change = abs(CHARGE.get(wt_aa, 0) - CHARGE.get(mut_aa, 0))

    # Disruption score (higher = more disruptive)
    disruption = (hydro_change + vol_change + charge_change) / 3.0

    # delta-R is more negative for disruptive mutations
    delta_r_enhanced = delta_r - disruption * 0.1  # <-- ADDS DISRUPTION TO PREDICTION
```

### Investigation Finding: **TRUE CIRCULARITY CONFIRMED**

The tautology is clear:

1. **delta_fitness** is computed as:
   ```
   delta_fitness = -(blosum_penalty + phys_penalty + ...)
   where phys_penalty = (hydro_change + vol_change + charge_change) / 3.0
   ```

2. **delta_r_enhanced** is computed as:
   ```
   delta_r_enhanced = delta_r - disruption * 0.1
   where disruption = (hydro_change + vol_change + charge_change) / 3.0
   ```

Both metrics include the **exact same disruption score** with the exact same formula. The correlation is guaranteed by construction.

**Why the red team found rho=0.728 for disruption alone:** Because disruption IS the core signal in delta_fitness (after noise). Delta-R only adds marginal information.

### What Would a Non-Circular Test Look Like?

1. **Use real DMS data**: Public DMS datasets (e.g., Fowler lab, ProteinGym) provide actual fitness measurements
2. **Remove disruption from delta_R**: Use ONLY the embedding-space delta_R without the physicochemical "enhancement"
3. **Blind prediction**: Compute delta_R without knowledge of the mutation's physicochemical properties

### Verdict: **RED TEAM CLAIM IS VALID**

---

## 4. Cross-Species Transfer Test Analysis

### Red Team Claim
> "ROBUST - Original r=0.828 is 71.3 standard deviations above random shuffles"

### Code Evidence

**From `red_team_analysis.py` - attack_cross_species_transfer() (lines 57-216):**

The shuffling test is correctly implemented:

```python
# ATTACK: Shuffle ortholog mapping randomly
n_shuffles = 100
shuffled_correlations = []

for i in range(n_shuffles):
    # Random permutation of ortholog mapping
    shuffled_mapping = rng.permutation(n_genes)[:n_orthologs]
    shuffled_human_R = human_R[shuffled_mapping]
    r_shuffled, _ = stats.pearsonr(shuffled_human_R, mouse_R)
    shuffled_correlations.append(r_shuffled)
```

### Investigation Finding: **ROBUST CONFIRMATION IS VALID**

The test correctly demonstrates:

1. **Random shuffling destroys correlation**: shuffled_mean = 0.0009, shuffled_max = 0.025
2. **Original correlation is extreme**: r = 0.828, z-score = 71.3
3. **P-value is essentially zero**: permutation_p = 0.0

**Why this test is NOT circular:**

1. Human R and mouse R are computed **independently** from different expression matrices
2. The ortholog mapping is **external information** (which genes are orthologs)
3. The shuffling control properly tests whether gene identity matters

**Potential concerns (minor):**

1. The data generation does include conservation-based correlation between species, which is realistic but synthetic
2. A stronger test would use real ortholog pairs with real expression data

### Verdict: **RED TEAM CONCLUSION IS VALID - This is the strongest evidence**

---

## 5. Perturbation Test Analysis

### Red Team Claim
> "Variance-weighted prediction achieves 0.061 cosine similarity, which is 103.6% of R-weighted (0.059)"

### Investigation Finding: **VALID - R provides no unique value**

The red team correctly identified that simpler baseline (inverse-variance weighting) performs equivalently or better than R-weighting. This does not indicate circularity per se, but rather that R may not provide unique predictive value beyond simpler metrics.

### Verdict: **RED TEAM CONCLUSION IS VALID**

---

## Summary of Circularity Types Detected

### Type 1: Direct Circular Definition (TRUE CIRCULARITY)
**Essentiality Test**: Ground truth labels were computed from the predictor variable.

### Type 2: Shared Feature Tautology (TRUE CIRCULARITY)
**Mutation Test**: Both predictor and target are computed from the same intermediate variable (disruption score).

### Type 3: Feature Overlap (PARTIAL CIRCULARITY)
**Folding Test**: Predictor and target share features, but both features genuinely relate to the phenomena. This is problematic but not as severe as Types 1-2.

### Type 4: Not Circularity (VALID)
**Cross-Species Test**: Predictor and target are computed independently; control test verifies gene identity matters.

---

## Recommendations

### 1. Redesign Essentiality Test
- Use real DepMap essentiality scores
- Ensure expression data and essentiality labels come from independent sources
- Consider using cell lines not in the expression training set

### 2. Redesign Folding Test
- Remove physicochemical features from R computation OR fold_quality
- Use external fold quality metrics (PDB statistics, AlphaFold confidence)
- Report contribution of R_base alone vs enhanced R

### 3. Redesign Mutation Test
- Remove disruption score from delta_R computation
- Use real DMS datasets (ProteinGym, etc.)
- Report base delta_R performance separately

### 4. Keep Cross-Species Test
- This is the strongest evidence for R
- Consider validation with real ortholog data
- Add phylogenetically distant species pairs

### 5. Additional Controls
For any test claiming R captures biological meaning:
- Report simpler baseline performance (1/CV, mean expression)
- Use permutation tests to establish null distribution
- Ensure no feature overlap between predictor and ground truth

---

## Conclusion

The adversarial red team's analysis is **largely correct**. Their methodology was sound: they traced through the code to identify shared variables and circular dependencies.

**Key findings:**

1. **3 of 5 tests have genuine circularity problems** (Essentiality, Mutation, Folding)
2. **1 test is robust** (Cross-Species Transfer)
3. **1 test shows R has no unique value** (Perturbation)

**Overall verdict on Q18 findings:**
The Q18 positive results are **NOT ROBUST** as a whole. The only credible evidence for R's biological relevance is the cross-species transfer finding. All other positive results are artifacts of test design.

**This does not mean R = E/sigma is wrong**, only that the current tests do not provide valid evidence for its correctness. Properly designed tests with independent ground truth are needed.

---

*Investigation completed: 2026-01-25*
*Investigator: Claude Opus 4.5*
