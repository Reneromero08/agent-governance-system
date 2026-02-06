# v2 Methodology Standard

**Version:** 2.0
**Principle:** Every hypothesis gets a fair, rigorous test. No result from v1 is assumed valid. No hypothesis is assumed false without proper evidence.

---

## Rules (Non-Negotiable)

### 1. One E Definition

All v2 work uses ONE definition of E:

```
E = mean pairwise cosine similarity of observations
E = (1/C(n,2)) * sum_{i<j} cos(x_i, x_j)
```

where x_i are embedding vectors and C(n,2) is the number of pairs.

No other E definition is permitted in v2. If a hypothesis requires a different E, it must be explicitly stated, justified, and the deviation documented. Tests that silently swap E are invalid.

### 2. Real External Data Required

Every Q must be tested on at least one real, publicly available external dataset. Synthetic data may be used for development and debugging but is NOT sufficient for any claim.

Acceptable data sources include (not limited to):
- HuggingFace datasets (SNLI, ANLI, STS-B, MNLI, SST-2, etc.)
- Standard NLP benchmarks (GLUE, SuperGLUE)
- Domain-specific public datasets (THINGS-EEG, HistWords, yfinance, etc.)
- Any dataset with a DOI or stable URL

The specific dataset(s) must be named in the test plan BEFORE running the test.

### 3. Pre-Registered Hypotheses

Before running any test, document:
- **H0 (null):** The specific claim being tested, stated in falsifiable form
- **H1 (alternative):** What you expect if the hypothesis is wrong
- **Success criterion:** A numerical threshold or statistical test (e.g., p < 0.05, r > 0.3, AUC > 0.7)
- **Failure criterion:** What result would falsify the hypothesis

These CANNOT be changed after seeing the data. If you want to test a modified hypothesis, run it as a separate, clearly labeled exploratory analysis.

### 4. Baseline Comparison Required

Every test must compare R (or whatever is being tested) against at minimum:
- **Bare E** (cosine similarity alone, no normalization)
- **Random baseline** (shuffled data, random embeddings, or chance-level performance)

If R does not outperform bare E, that is a finding, not a failure. Report it honestly.

### 5. No Post-Hoc Rescue

If a test fails:
- Do NOT replace the test with an easier one
- Do NOT change the success criterion
- Do NOT switch to a different metric
- DO report the failure honestly
- You MAY run additional exploratory analyses, but these must be clearly labeled as exploratory and cannot override the pre-registered result

### 6. Consistent Formula

The full formula under test is:

```
R = (E / grad_S) * sigma^Df
```

where:
- E = mean pairwise cosine similarity (as defined above)
- grad_S = standard deviation of pairwise cosine similarities
- sigma = compression ratio (to be measured, not assumed)
- Df = fractal dimension (to be measured, not assumed)

If a test drops sigma^Df (using only E/grad_S), this must be explicitly stated and the reason documented. This is acceptable but must be transparent.

### 7. Reproducibility

Every test must include:
- Complete code (no pseudocode)
- Exact dataset version/split used
- Random seed if applicable
- Expected runtime
- Instructions to reproduce from scratch

### 8. Accept the Result

Whatever the properly-conducted test shows is the answer. Confirmed, falsified, inconclusive -- all are valid scientific outcomes. A well-executed falsification is worth more than a poorly-executed confirmation.

---

## Q Directory Structure (v2)

Each Q in v2 follows this structure:

```
q{##}_{short_name}/
  README.md           # Hypothesis, previous evidence, test plan, criteria
  test_plan.md        # Detailed pre-registered test methodology
  code/               # Test implementation
  data/               # Downloaded/generated datasets (or download scripts)
  results/            # Raw outputs (JSON, CSV)
  VERDICT.md          # Final result after testing (confirmed/falsified/inconclusive)
```

### README.md Template

Every Q README contains these sections:

1. **Hypothesis** -- The original claim, stated precisely and falsifiably
2. **v1 Evidence Summary** -- What was previously tested, what the results were
3. **v1 Methodology Problems** -- What went wrong with previous tests (from verification)
4. **v2 Test Plan** -- How we will properly test this hypothesis
5. **Required Data** -- Specific external datasets needed
6. **Success/Failure Criteria** -- Pre-registered numerical thresholds
7. **Baseline Comparisons** -- What R must outperform
8. **Salvageable from v1** -- Code/data worth keeping (if any)

### VERDICT.md Template

Written AFTER testing:

1. **Result:** CONFIRMED / FALSIFIED / INCONCLUSIVE
2. **Evidence:** Summary of test results with numbers
3. **Data:** What datasets were used
4. **Methodology:** Brief description of test procedure
5. **Limitations:** What the test did NOT cover
6. **Next Steps:** If inconclusive, what additional test would resolve it

---

## Grading Criteria

After testing, each Q receives a grade:

| Grade | Meaning | Criteria |
|-------|---------|----------|
| A | Properly tested, clear result | Real data, pre-registered, baseline compared, reproducible |
| B | Tested with minor gaps | Real data but small n, or baseline comparison missing |
| C | Tested but methodology issues | Synthetic-only, or success criteria changed, or no baseline |
| D | Tested but invalid | Wrong E, circular setup, tautological |
| F | Not tested | No code run, no data, just claims |

v1 results were mostly D and F. v2 targets A for every Q.

---

## What "Falsification" Means in v2

A hypothesis is falsified when:
1. A properly designed test (meeting all rules above) produces a result that meets the pre-registered failure criterion
2. The test is reproducible
3. No plausible methodological flaw explains the negative result
4. The result is documented in VERDICT.md

A hypothesis is NOT falsified when:
- A badly designed test fails (that's a bad test, not a falsification)
- Someone asserts it's wrong without testing
- A different hypothesis fails (each stands or falls on its own evidence)

---

## Status Labels (v2)

| Status | Meaning |
|--------|---------|
| OPEN | Not yet tested in v2 |
| TESTING | Test designed, data being collected/analyzed |
| CONFIRMED | Pre-registered test passed on real data |
| FALSIFIED | Pre-registered test failed on real data |
| INCONCLUSIVE | Test ran but result is ambiguous; needs more data or better test |
| BLOCKED | Cannot test until a dependency is resolved |

All 54 Qs start as OPEN in v2, regardless of v1 status.
