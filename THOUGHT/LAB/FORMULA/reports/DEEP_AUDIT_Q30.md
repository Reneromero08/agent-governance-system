# Deep Audit: Q30 Approximations

**Auditor:** Claude Opus 4.5
**Date:** 2026-01-27
**Status:** CORRECTLY RESOLVED - ENGINEERING SOLUTION

---

## Summary

Q30 asks: "Are there faster approximations that preserve gate behavior for large-scale systems?"

**Verdict: WELL-RESOLVED** - Comprehensive testing demonstrates 100-300x speedup with 100% gate agreement using random sampling.

---

## Test Verification

### Code Review

**Test File:** `experiments/open_questions/q30/test_q30_approximations.py` (1237 lines)
**Result File:** `experiments/open_questions/q30/q30_test_results.json` (522 lines)

| Check | Status |
|-------|--------|
| Test file exists | YES |
| Results file exists | YES |
| Pre-registration documented | YES |
| Multiple methods tested | YES |
| Scaling behavior verified | YES |

---

## What Was Tested

### 6 Tests Executed (All Passed)

| Test | Goal | Result |
|------|------|--------|
| ACCURACY_PRESERVATION | >95% gate agreement | PASS (100% for 7/8 methods) |
| SPEEDUP_MEASUREMENT | >=10x speedup | PASS (124x average for sampled_50) |
| PARETO_FRONTIER | Non-dominated methods | PASS (sample_20 on frontier) |
| SCALING_BEHAVIOR | Subquadratic scaling | PASS (3 methods subquadratic) |
| ROBUSTNESS | Consistent across agreement levels | PASS (4 robust methods) |
| RECOMMENDED_IMPL | Best overall method | PASS (sampled_50) |

### Methods Compared

| Method | Complexity | Accuracy | Speedup (n=500) |
|--------|------------|----------|-----------------|
| Exact | O(n^2) | 100% | 1.0x (baseline) |
| Sampled (k=20) | O(k^2) | 100% | 297.9x |
| Sampled (k=50) | O(k^2) | 100% | 72.9x |
| Sampled (k=100) | O(k^2) | 100% | 24.2x |
| Centroid | O(n) | 100% | 82.0x |
| Projected (d=64) | O(n^2) | 100% | 1.0x |
| Nystrom (k=50) | O(kn + k^3) | 100% | 4.9x |
| Streaming | O(batch^2 * batches) | 100% | 21.5x |
| Combined | O(k^2) | 83.3% | 67.8x |

---

## Critical Findings

### Finding 1: RANDOM SAMPLING IS REMARKABLY EFFECTIVE

The key result: **sampling just 20-50 observations gives perfect gate agreement** while being 70-300x faster.

| Sample Size | Gate Agreement | Speedup |
|-------------|----------------|---------|
| k=20 | 100% | 297.9x |
| k=50 | 100% | 72.9x |
| k=100 | 100% | 24.2x |

This works because:
- Gate decisions are binary (R >= threshold)
- R values tend to be clearly above or below threshold
- Sample statistics converge quickly (Central Limit Theorem)

### Finding 2: SCALING EXPONENTS VERIFIED

Empirical scaling analysis:

| Method | Empirical Scaling |
|--------|-------------------|
| Exact | O(n^1.93) (theoretical O(n^2)) |
| Sampled (fixed k) | O(n^0.18) (near constant!) |
| Centroid | O(n^0.96) (linear) |
| Combined | O(n^-0.04) (constant) |

Three methods achieve **subquadratic scaling**, meaning speedup increases with n.

### Finding 3: PARETO FRONTIER ANALYSIS

Only `sample_20` is on the Pareto frontier (non-dominated):
- 100% accuracy
- 297.9x speedup
- No method has both higher accuracy AND higher speedup

### Finding 4: R VALUE ERROR IS HIGH BUT IRRELEVANT

The actual R values can have significant error (up to 250%), but this **does not affect gate decisions** because:
- Gate decisions depend only on R vs threshold
- The ordering (R_high vs R_low) is preserved
- Exact R value is rarely needed in practice

### Finding 5: COMBINED METHOD FAILURE

The combined approach (sample + project) achieved only 83.3% accuracy due to projection distortion. This is an important negative result - sometimes combining optimizations backfires.

---

## Data Integrity Checks

| Check | Result |
|-------|--------|
| Uses synthetic embeddings | YES (appropriate for speed testing) |
| Multiple test sizes (50-1000) | YES |
| Multiple agreement levels | YES (high, medium, low, mixed) |
| Timing measurements reliable | YES (multiple runs averaged) |
| Results consistent | YES |

**Note:** Synthetic embeddings are appropriate here because the test measures **computational performance**, not semantic properties. The gate behavior preservation was tested across different agreement scenarios.

---

## Recommended Implementation

From the test:

```python
def compute_r_fast(embeddings, sample_size=50, epsilon=1e-6, seed=None):
    n = len(embeddings)
    if n < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    k = min(sample_size, n)
    indices = rng.choice(n, size=k, replace=False)
    sampled = [embeddings[i] for i in indices]

    similarities = []
    for i in range(k):
        for j in range(i + 1, k):
            similarities.append(np.dot(sampled[i], sampled[j]))

    E = np.mean(similarities)
    sigma = np.std(similarities)
    return E / max(sigma, epsilon)
```

---

## Verdict

**STATUS: CORRECTLY RESOLVED**

This is excellent engineering work:
1. Clear problem definition (O(n^2) bottleneck)
2. Multiple solutions tested (8 methods)
3. Comprehensive benchmarking (4 test sizes, 4 agreement levels)
4. Scaling behavior empirically verified
5. Pareto analysis to identify best tradeoffs
6. Practical recommendation with code

### Key Insights:

| Finding | Implication |
|---------|-------------|
| Sampling works perfectly | Use k=50 for production |
| Scaling is subquadratic | Speedup increases with n |
| R error doesn't matter | Gate decisions are robust |
| Combined method fails | Simpler is better |

---

## Bullshit Check

| Red Flag | Found? |
|----------|--------|
| Synthetic test inappropriate | NO (appropriate for speed testing) |
| Overclaiming | NO |
| Cherry-picked results | NO (negative results reported) |
| Missing edge cases | NO (multiple scenarios) |
| Unrealistic speedups | NO (empirically measured) |

**Overall:** This is a well-executed engineering solution with comprehensive validation.

---

## Files Examined

- `experiments/open_questions/q30/test_q30_approximations.py` (1237 lines)
- `experiments/open_questions/q30/q30_test_results.json` (522 lines)
- `research/questions/engineering/q30_approximations.md` (160 lines)
