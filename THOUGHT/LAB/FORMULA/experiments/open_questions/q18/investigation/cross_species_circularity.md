# Q18 Cross-Species Transfer Test Circularity Investigation

**Date:** 2026-01-25
**Investigator:** Claude Opus 4.5
**File Analyzed:** `test_tier4_gene_expression.py` (lines 131-271, 278-323)

---

## EXECUTIVE SUMMARY

**VERDICT: YES - THE TEST IS CIRCULAR BY CONSTRUCTION**

The r=0.828 cross-species correlation does NOT test whether R captures biology. It tests whether synthetic data generated with built-in human-mouse correlation shows that correlation when measured. This is a tautology, not a validation.

---

## DETAILED ANALYSIS

### Question 1: How is mouse_expression generated?

**Source code (lines 215-230):**

```python
for mouse_idx, human_idx in enumerate(ortholog_mapping):
    # Conservation coefficient varies by gene
    conservation = rng.uniform(0.5, 0.95)  # <-- KEY: 50-95% conservation

    # Species-specific scaling
    species_scale = rng.uniform(0.7, 1.3)

    # Mouse expression = conserved component + species noise
    human_expr = human_expression[human_idx, :]
    species_noise = rng.normal(0, 0.3, n_samples)

    mouse_expression[mouse_idx, :] = (
        conservation * species_scale * human_expr +  # <-- DIRECT COPY of human
        (1 - conservation) * rng.normal(np.mean(human_expr), np.std(human_expr), n_samples) +
        species_noise
    )
```

**Critical Finding:**
- Mouse expression is **directly derived from human expression**
- The "conservation" factor (0.5-0.95) controls how much of human_expression is copied
- Average conservation = 0.725 (midpoint), meaning ~72.5% of mouse expression is human expression

**Mathematical breakdown:**
```
mouse_expression[i] =
    conservation * species_scale * human_expression[i]    # 50-95% direct copy
  + (1-conservation) * N(mu_human, sigma_human)           # 5-50% sampled noise
  + N(0, 0.3)                                             # species noise
```

### Question 2: Is the correlation BUILT INTO the synthetic data?

**YES, ABSOLUTELY.**

Let's trace the logic:

1. **Human expression** is generated with specific CV patterns (lines 176-204):
   - Housekeeping genes: low noise (CV 0.1-0.3)
   - Tissue-specific genes: high noise (bimodal pattern)
   - Other genes: moderate noise (CV 0.3-1.0)

2. **Human R** is computed as:
   ```
   R_human = E_human / sigma_human
   ```
   where E and sigma both derive from human_expression

3. **Mouse expression** is generated as:
   ```
   mouse = conservation * human + independent_noise
   ```

4. **Mouse R** is computed as:
   ```
   R_mouse = E_mouse / sigma_mouse
   ```

5. **The critical flaw:** Because mouse_expression is a noisy copy of human_expression:
   - If human_expression has low CV, mouse_expression will also have low CV (plus some noise)
   - If human_expression has high CV, mouse_expression will also have high CV (plus some noise)
   - Therefore, **R_mouse will inherently correlate with R_human** because they measure the same underlying signal

**This is not testing biological transfer. This is testing whether:**
```
R(f(x)) correlates with R(x)  where f(x) = ax + noise
```

The answer is trivially YES for any reasonable function R.

### Question 3: Is the red team shuffle test valid?

**NO, the shuffle test does NOT validate biological relevance.**

The shuffle test (if present) would:
1. Shuffle ortholog pairings randomly
2. Show that correlation drops to ~0
3. Claim this proves the correlation is "real"

**Why this is misleading:**

The shuffle test only proves that the **specific pairing** matters. But that pairing was **constructed** to have correlation by design:

```
mouse[i] = f(human[i])  # Built-in correlation
```

When you shuffle:
```
mouse[i] = f(human[j])  where j != i  # Pairing broken
```

Of course the correlation drops! But this doesn't prove R captures biology. It proves that your synthetic data generation created paired data.

**Analogy:** If I create pairs (x, 2x+noise), then measure correlation between transformed(x) and transformed(2x+noise), and show shuffling destroys correlation - I haven't proven anything about the transform. I've proven I generated paired data.

### Question 4: What would happen with INDEPENDENT data?

**Expected result: r approximately 0**

If we generated:
```python
human_expression = generate_random_expression(seed=42)
mouse_expression = generate_random_expression(seed=99)  # COMPLETELY INDEPENDENT
```

Then:
- R_human would be determined by human_expression's random CV patterns
- R_mouse would be determined by mouse_expression's random CV patterns
- These would be **statistically independent**
- Pearson r would be approximately 0 (within sampling noise)

**This would be the actual test:** Does R capture something intrinsic about "geneness" that transfers across species when measured on genuinely independent biological samples?

---

## STRUCTURAL DIAGNOSIS

### The Circular Logic Chain

```
1. Generate human_expression with specific patterns
2. Generate mouse_expression = f(human_expression) with 72.5% average conservation
3. Compute R_human = g(human_expression)
4. Compute R_mouse = g(mouse_expression) = g(f(human_expression))
5. Measure correlation(R_human, R_mouse)
6. Find high correlation (r=0.828)
7. CLAIM: "R transfers across species!"
```

**The fallacy:** Step 2 guarantees step 6. The correlation is STRUCTURAL, not EMPIRICAL.

### What The Test Actually Validates

The test validates that:
```
corr(g(x), g(ax + noise)) > 0  when a ~ 0.725
```

This is a mathematical property of the data generation, NOT evidence that R captures biology.

---

## WHAT A NON-CIRCULAR TEST WOULD LOOK LIKE

### Option 1: Real Data
Use actual human and mouse RNA-seq datasets:
- GTEx (human)
- Mouse ENCODE or similar

Compute R on each independently collected dataset and test correlation.

### Option 2: Independent Synthetic Data
```python
# Generate human expression with biologically-motivated parameters
human_expression = generate_expression(
    gene_categories=human_GO_annotations,
    tissue_specificity=human_atlas_patterns,
    seed=42
)

# Generate mouse expression with INDEPENDENT biological parameters
mouse_expression = generate_expression(
    gene_categories=mouse_GO_annotations,  # Different source
    tissue_specificity=mouse_atlas_patterns,  # Different source
    seed=99  # Different seed
)

# NOW test if R correlates for ortholog pairs
# If it does, that's evidence R captures something universal
```

### Option 3: Permutation-Based Null Model
If using synthetic data, the null should be:
```python
# Null: orthologs share no regulatory structure
mouse_null = generate_expression_independent_of_human()

# Alternative: orthologs share regulatory structure
mouse_alt = generate_expression_with_shared_regulatory_logic()

# Test should distinguish these, NOT compare to shuffled pairings
```

---

## FINAL VERDICT

| Question | Answer |
|----------|--------|
| Is the cross-species test CIRCULAR by construction? | **YES** |
| Does r=0.828 test biological transfer? | **NO** |
| What does r=0.828 actually test? | Whether noisy copies correlate with originals |
| Is the shuffle test (z=71.3) valid? | **NO** - it only confirms data generation structure |
| What would a non-circular test show? | Unknown - would require independent data |

---

## RECOMMENDATIONS

1. **Do not cite the r=0.828 result** as evidence of biological validity
2. **Rerun with real data** (GTEx + Mouse ENCODE) to get actual cross-species validation
3. **Acknowledge the limitation** in any publications: "Synthetic data validation; biological validation pending"
4. **Redesign the test** to use independent data sources for human and mouse
5. **Consider the essentiality test (Test 4.2)** has the same circularity problem:
   - Essentiality scores are DERIVED from R values (line 247): `base_essentiality = -0.5 * np.log(R_values[i] + 1e-6)`
   - Then R is used to predict essentiality
   - This is also circular by construction

---

## CODE EVIDENCE

The smoking gun is on lines 226-229:
```python
mouse_expression[mouse_idx, :] = (
    conservation * species_scale * human_expr +  # <-- Circular dependency
    (1 - conservation) * rng.normal(...) +
    species_noise
)
```

And the essentiality circularity on lines 241-247:
```python
R_values = compute_R_genomic(human_expression)  # First compute R
# ...
base_essentiality = -0.5 * np.log(R_values[i] + 1e-6)  # Then derive essentiality FROM R
```

Both tests are validating that `f(x)` correlates with `x`, not that `R` captures biological reality.

---

**Investigation complete. The cross-species test as implemented is scientifically invalid due to circular data generation.**
