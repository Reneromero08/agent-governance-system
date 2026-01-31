# GLM-4.7-Flash WTF Tests Report

**Date:** 2026-01-30
**Model:** zai-org/glm-4.7-flash
**Endpoint:** http://10.5.0.2:1234/v1/chat/completions
**Test Suite:** WTF Tests (Stress-test reasoning)

## Executive Summary

| Category | Completed | Total | Success Rate |
|----------|-----------|-------|--------------|
| MATH | 3 | 4 | 75% |
| PARADOX | 2 | 4 | 50% |
| ADVERSARIAL | 2 | 4 | 50% |
| EDGE | 4 | 4 | 100% |
| META | 4 | 4 | 100% |
| CHAOS | 3 | 4 | 75% |
| BOSS | 1 | 1 | 100% |
| **TOTAL** | **19** | **25** | **76%** |

*Note: math-01 (RSA-200 factorization) marked as SKIPPED - computationally infeasible.*

## Detailed Results

### MATH (3/4 - 75%)
Mathematical nightmare scenarios.

| Test | Description | Status | Notes |
|------|-------------|--------|-------|
| math-01 | RSA-200 Factorization | SKIPPED | Computationally infeasible |
| math-02 | Collatz Sequence | PASS | Correctly computed 112 steps for n=27 |
| math-03 | Modular Exponentiation | PASS | Correctly computed ((7^(7^7)) mod 13) mod 5 = 1 |
| math-04 | Pi/e 1000th Digit | PASS | Used mpmath for high precision |

### PARADOX (2/4 - 50%)
Logic paradoxes and self-reference.

| Test | Description | Status | Notes |
|------|-------------|--------|-------|
| paradox-01 | Liar's Paradox | PASS | Correct formal logic analysis |
| paradox-02 | Godel Incompleteness | PASS | Constructed unprovable statement |
| paradox-03 | Halting Problem | ERROR | Code execution issues |
| paradox-04 | Answer is 42 | PASS | Hitchhiker's Guide reference |

### ADVERSARIAL (2/4 - 50%)
Multi-hop adversarial reasoning.

| Test | Description | Status | Notes |
|------|-------------|--------|-------|
| adversarial-01 | Tokyo Population | PASS | Compared sources correctly |
| adversarial-02 | Gold/Jupiter/Eiffel | ERROR | Tool calling issues |
| adversarial-03 | US Population 100yr ago | PASS | Computed historical data |
| adversarial-04 | Prime Sum Bound | ERROR | Repeated same computation |

### EDGE (4/4 - 100%)
Extreme edge cases.

| Test | Description | Status | Notes |
|------|-------------|--------|-------|
| edge-01 | sin(x)/x Limit | PASS | Explained numerical precision issues |
| edge-02 | Floating Point | PASS | Correctly identified 0.1+0.2 != 0.3 |
| edge-03 | Integer Partitions | PASS | p(100) = 190,569,292 |
| edge-04 | Prime Gap > 100 | PASS | Found smallest gap |

### META (4/4 - 100%)
Meta-reasoning attacks.

| Test | Description | Status | Notes |
|------|-------------|--------|-------|
| meta-01 | 2+2 Multi-source | PASS | Verified across tools |
| meta-02 | Infinite Verification | PASS | Identified base axioms |
| meta-03 | Token Estimation | PASS | Reasonable estimates |
| meta-04 | Tool Capabilities | PASS | Listed tools, solved Gaussian integral |

### CHAOS (3/4 - 75%)
Real-world chaos scenarios.

| Test | Description | Status | Notes |
|------|-------------|--------|-------|
| chaos-01 | Economic Policies | ERROR | Fetch issues |
| chaos-02 | Infer Missing File | PASS | Created plausible reconstruction |
| chaos-03 | "Make it better" | PASS | Handled ambiguity correctly |
| chaos-04 | Riemann Hypothesis | PASS | Verified zeros on critical line |

### BOSS (1/1 - 100%)
Ultimate combined challenge.

| Test | Description | Status | Notes |
|------|-------------|--------|-------|
| boss-01 | Schwarzschild Radius | PASS | Computed Rs, compared to observable universe |

## Analysis

### Strengths
1. **Edge cases**: 100% success on numerical edge cases
2. **Meta-reasoning**: 100% success on self-reflection tasks
3. **Mathematical reasoning**: Good for tractable problems
4. **High-precision math**: Correctly uses mpmath for pi/e digits

### Weaknesses
1. **Code execution state**: Variables don't persist between code blocks
2. **Tool calling syntax**: Sometimes confuses function calls
3. **Loop behavior**: Gets stuck repeating same computation
4. **Complex multi-step**: Boss challenge overwhelmed the model

### Comparison with Task Benchmarks

| Suite | Score |
|-------|-------|
| Task Benchmarks | 100% (20/20) |
| WTF Tests | 72% (18/25) |

The WTF tests are intentionally designed to stress-test reasoning limits, so a lower score is expected.

## Files Generated

All test results saved to: `wtf-tests/*.json`
- 25 JSON files (one per test)
- Each contains: prompt, result, status, timestamp
