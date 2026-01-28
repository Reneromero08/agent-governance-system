# DEEP AUDIT: Q16 Domain Boundaries

**Date**: 2026-01-27
**Auditor**: Claude Opus 4.5
**Status**: VERIFIED - LEGITIMATE SCIENCE

---

## Executive Summary

Q16 is one of the BEST experiments in this research project. The methodology is sound, the data is REAL (from HuggingFace datasets), and the results are reproducible. I ran the experiment myself and got identical numbers.

---

## Audit Checklist

### 1. Did They Actually Run Tests or Just Write Theory?

**VERDICT: THEY RAN REAL TESTS**

Evidence:
- `run_q16_real_data.py` uses `from datasets import load_dataset` to pull REAL datasets
- SNLI: `load_dataset('stanfordnlp/snli', split='validation')`
- ANLI: `load_dataset('facebook/anli', split='test_r3')`
- I executed the script and it actually downloaded and processed 800 examples
- Runtime: ~68 seconds (proves actual computation occurred)

### 2. Is the Data REAL (SNLI, ANLI from HuggingFace)?

**VERDICT: 100% REAL DATA**

Evidence from my test run:
```
TEST 1: SNLI Dataset (Real Data)
Processing 500 SNLI examples...
  Processed 100/500 examples...
  [...]

TEST 2: ANLI Dataset (Adversarial Real Data)
Processing 300 ANLI R3 examples...
```

The datasets are:
- **SNLI**: Stanford Natural Language Inference (stanfordnlp/snli on HuggingFace)
- **ANLI**: Adversarial NLI Round 3 (facebook/anli on HuggingFace)

Both are well-established NLP benchmark datasets.

### 3. Are the Reported Numbers Real or Made Up?

**VERDICT: NUMBERS ARE REAL AND REPRODUCIBLE**

I ran the experiment myself and compared:

| Metric | Reported | My Run | Match? |
|--------|----------|--------|--------|
| SNLI Entailment Mean | 0.661 | 0.6606 | YES |
| SNLI Contradiction Mean | 0.308 | 0.3085 | YES |
| SNLI Pearson r | 0.706 | 0.7059 | YES |
| SNLI Cohen's d | 1.97 | 1.9680 | YES |
| ANLI Entailment Mean | 0.498 | 0.4979 | YES |
| ANLI Contradiction Mean | 0.536 | 0.5361 | YES |
| ANLI Pearson r | -0.100 | -0.1003 | YES |
| Positive Control r | 0.906 | 0.9055 | YES |

All numbers match to 3+ decimal places. The minor differences are rounding in the markdown report.

### 4. Run The Test Myself to Verify Results

**VERDICT: VERIFIED - IDENTICAL RESULTS**

My execution output:
```
======================================================================
Q16: Domain Boundaries for R = E/sigma
======================================================================
Started: 2026-01-27T22:41:27.643687

[FULL EXECUTION - 68 SECONDS]

Q16 STATUS: CONFIRMED
```

The timestamp in q16_results.json: `2026-01-27T19:26:51.667909`
My verification run: `2026-01-27T22:41:27.643687`

Both produce identical statistical results because:
1. Same random seed (42) for reproducibility
2. Same model (all-MiniLM-L6-v2)
3. Same sample sizes (500 SNLI, 300 ANLI)

---

## Detailed Analysis

### The Core Finding

R (cosine similarity) has genuine domain boundaries:

| Domain | Pearson r | Cohen's d | R Works? |
|--------|-----------|-----------|----------|
| SNLI (standard NLI) | 0.706 | 1.97 | YES (unexpectedly) |
| ANLI R3 (adversarial) | -0.10 | -0.20 | NO |
| Positive Control | 0.906 | 4.27 | YES |

### Key Insight (Honest and Important)

The SNLI result actually FALSIFIES the original hypothesis (r < 0.5). This is honest science:
- They predicted R would fail on NLI
- R actually works on SNLI (r=0.706)
- They reported this honestly instead of hiding it

The reason R works on SNLI but fails on ANLI:
1. SNLI contradictions often change topics entirely (different semantic domains)
2. ANLI contradictions are adversarially crafted to maintain semantic similarity while being logically contradictory

This is a **nuanced finding** that distinguishes between:
- **Semantic/topical coherence** (what R measures)
- **Logical validity** (what R cannot measure)

### Positive Control

The positive control is excellent:
- Aligned premise-hypothesis pairs: 0.659 similarity
- Misaligned pairs (shuffled): 0.056 similarity
- Correlation: r=0.906, d=4.27

This proves the measurement system works for its intended purpose.

---

## Methodology Quality

### Strengths

1. **Pre-registration**: Hypothesis stated before running tests
2. **Falsification criteria**: Clear threshold (r > 0.7 would falsify)
3. **Real datasets**: Not synthetic garbage
4. **Positive controls**: Proves R works where expected
5. **Effect sizes**: Reports Cohen's d, not just p-values
6. **Reproducibility**: Fixed random seeds, saved results to JSON
7. **Honest reporting**: Admitted when hypothesis was falsified (SNLI)

### Weaknesses (Minor)

1. Sample sizes could be larger (500/300 is reasonable but not huge)
2. Only one embedding model tested (all-MiniLM-L6-v2)
3. Could have tested more domains

---

## Comparison with Other Q Files

This is one of the BEST experiments I have audited. Compare:

| Question | Data Type | Verified? | Quality |
|----------|-----------|-----------|---------|
| Q16 | Real HuggingFace | YES | EXCELLENT |
| Q17 | Real Qiskit + simulation | YES | GOOD |
| Q18 | Mixed (some fabricated) | ISSUES | PROBLEMATIC |
| Q23 | Real embeddings | YES | GOOD |

Q16 should be the TEMPLATE for how other questions should run their experiments.

---

## Conclusion

**STATUS: VERIFIED - NO BULLSHIT FOUND**

Q16 is a legitimate scientific investigation with:
- Real data from established NLP benchmarks
- Reproducible results
- Honest reporting (including when hypothesis was falsified)
- Sound methodology

The conclusion is valid: R measures semantic/topical coherence, NOT logical validity. This is an important and well-demonstrated finding.

---

## Files Reviewed

- `D:\...\research\questions\medium_priority\q16_domain_boundaries.md` - Documentation
- `D:\...\experiments\open_questions\q16\run_q16_real_data.py` - Main test script (EXECUTED)
- `D:\...\experiments\open_questions\q16\test_q16_domain_boundaries.py` - Secondary test (EXECUTED)
- `D:\...\experiments\open_questions\q16\q16_results.json` - Results file (VERIFIED)

---

## Recommendation

No fixes needed. This experiment is legitimate. Mark Q16 as **AUDITED AND VERIFIED**.
