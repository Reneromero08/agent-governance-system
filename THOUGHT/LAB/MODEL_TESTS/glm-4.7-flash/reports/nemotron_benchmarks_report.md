# GLM-4.7-Flash Nemotron Benchmark Results

**Date:** 2026-01-31
**Model:** zai-org/glm-4.7-flash
**Endpoint:** http://10.5.0.2:1234/v1/chat/completions
**Test Suite:** Nemotron Comparison Benchmarks

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Tests | 86 |
| Completed | 86 |
| Errors | 0 |
| Success Rate | 100% |

## Test Categories

### R01 - Basic Capability (3 tests)
- r01-code: PASS
- r01-knowledge: PASS
- r01-math: PASS

### R02 - Logic & Trick Questions (4 tests)
- r02-bat-ball: PASS
- r02-lcs: PASS
- r02-sheep: PASS
- r02-widgets: PASS

### R03 - Advanced Reasoning (4 tests)
- r03-boxes: PASS
- r03-calculus: PASS
- r03-spatial: PASS
- r03-water-jug: PASS

### R04 - Traps That Fool Models (4 tests)
- r04-family: PASS
- r04-number: PASS
- r04-sisters: PASS
- r04-syllogism: PASS

### R05 - Paradoxes (4 tests)
- r05-liar: PASS
- r05-multiply: PASS
- r05-regex: PASS
- r05-temporal: PASS

### R06 - Riddles & Code Bugs (4 tests)
- r06-bug: PASS
- r06-hourglass: PASS
- r06-lateral: PASS
- r06-murderer: PASS

### R07 - Famous Model Failures (4 tests)
- r07-strawberry: PASS
- r07-survivors: PASS
- r07-transitive: PASS
- r07-weight: PASS

### R08 - Edge Cases (4 tests)
- r08-apples: PASS
- r08-bsearch: PASS
- r08-decimal: PASS
- r08-half: PASS

### R09 - Advanced Math (4 tests)
- r09-gaussian: PASS
- r09-gods: PASS
- r09-pi100: PASS
- r09-tom: PASS

### R10 - Classic Puzzles (4 tests)
- r10-balls: PASS
- r10-look-say: PASS
- r10-mersenne: PASS
- r10-snail: PASS

### R11 - Paradoxes & Assembly (4 tests)
- r11-asm: PASS
- r11-force: PASS
- r11-sqrt2: PASS
- r11-threee: PASS

### R12 - Knowledge & Hallucination (4 tests)
- r12-election: PASS
- r12-lies: PASS
- r12-sycophancy: PASS
- r12-timezone: PASS

### R13 - Exotic Edge Cases (4 tests)
- r13-barber: PASS
- r13-conversion: PASS
- r13-fizzbuzz: PASS
- r13-swallow: PASS

### R14 - Final Stress Tests (4 tests)
- r14-count-a: PASS
- r14-float: PASS
- r14-pills: PASS
- r14-reverse: PASS

### R15 - Theoretical Physics (4 tests)
- r15-blackhole: PASS
- r15-qft: PASS
- r15-schwarzschild: PASS
- r15-string: PASS

### R16 - Pure Mathematics (4 tests)
- r16-category: PASS
- r16-godel: PASS
- r16-proofs: PASS
- r16-riemann: PASS

### R17 - AGS Formula (2 tests)
- r17-formula: PASS
- r17-semiotic: PASS

### R18 - Semiotic Paradoxes (2 tests)
- r18-fixedpoint: PASS
- r18-semiotic-liar: PASS

### R19 - Quantum Computing & Algebra (7 tests)
- r19-cook-levin: PASS
- r19-fundamental: PASS
- r19-galois: PASS
- r19-poincare: PASS
- r19-qec: PASS
- r19-shor: PASS
- r19-tensor: PASS

### R20 - Complexity & Number Theory (5 tests)
- r20-hierarchy: PASS
- r20-homology: PASS
- r20-langlands: PASS
- r20-pvsnp: PASS
- r20-representation: PASS

### R21 - Adversarial Reasoning (7 tests)
- r21-collatz: PASS
- r21-cubes33: PASS
- r21-fixed: PASS
- r21-hash: PASS
- r21-magic: PASS
- r21-ramsey55: PASS
- r21-zebra: PASS

## Model Comparison

| Model | Benchmark Tests | Completion Rate |
|-------|-----------------|-----------------|
| GLM-4.7-flash | 86 | 100% |
| nemotron-3-nano-30b | 36 | ~98.8% |

Note: Direct comparison is limited as test sets differ in size. GLM ran an expanded 86-test suite.

## Technical Notes

- Model is slow (~15 min per test with 500 max_tokens)
- Required extended timeout (1800s vs original 300s)
- All tests completed successfully with 2 parallel workers
- Results saved to nemotron-benchmarks/*.json
