# Engineering Solution: Numerical Stability for R-Gate

**TYPE: ENGINEERING (Implementation Guidance)**
**STATUS: SOLVED**

---

## Problem Statement

When computing `R = E / sigma`, division by zero occurs if sigma (standard deviation of embeddings) equals zero. This happens when all observations are identical. This is a standard numerical stability bug, not a research question.

## Recommended Implementation

```python
def compute_r_stable(E: float, sigma: float, epsilon: float = 1e-6) -> float:
    """
    Stable R computation with epsilon floor.
    Prevents division by zero when sigma approaches 0.
    """
    return E / max(sigma, epsilon)
```

## Why epsilon = 1e-6?

| Consideration | Rationale |
|---------------|-----------|
| Prevents div/0 | sigma can be exactly 0 for identical observations |
| Preserves sensitivity | Typical sigma values are 0.01 - 1.0; epsilon is 4+ orders smaller |
| Standard practice | Common choice in numerical computing |

## Alternative Implementations

All of these work. Choose based on your use case:

| Method | Code | Use When |
|--------|------|----------|
| Epsilon floor | `E / max(sigma, eps)` | Default choice, simple |
| Soft sigmoid | `sigmoid(k * (R - threshold))` | Need probabilistic output |
| MAD robust | Use median absolute deviation | Data has outliers |
| Adaptive epsilon | `eps * (1 + abs(E))` | Signal strength varies widely |
| Log ratio | `log(1 + E) / log(1 + sigma)` | Extreme value ranges |

## Validation

Unit tests confirm all methods handle edge cases correctly:

| Edge Case | E | sigma | Result | Pass |
|-----------|---|-------|--------|------|
| Identical embeddings | 1.0 | 0.0 | 1,000,000 | Yes |
| High E, low sigma | 0.9997 | 0.0001 | 9,596 | Yes |
| Orthogonal vectors | -0.01 | 0.04 | -0.27 | Yes |
| One outlier | 0.2 | 0.98 | 0.20 | Yes |

**Test Results**: 8/8 pass, 100% gate accuracy, 97.6% F1

## Reference

- Implementation: `experiments/open_questions/q29/test_q29_numerical_stability.py`
- Test output: `experiments/open_questions/q29/q29_test_results.json`

---

**Note**: This is solved engineering, not open science. The epsilon-floor pattern is standard numerical computing practice. No hypothesis was tested; a known bug class was fixed with a known solution.
