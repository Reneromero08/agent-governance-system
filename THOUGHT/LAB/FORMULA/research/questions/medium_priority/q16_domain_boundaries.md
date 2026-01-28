# Question 16: Domain boundaries (R: 1440)

**STATUS: CONFIRMED**

## Question
Are there domains where R fundamentally cannot work? (e.g., adversarial, non-stationary, or self-referential systems)

## Answer
YES. R (cosine similarity / semantic coherence) has fundamental domain boundaries:

- **WORKS FOR**: Topical consistency, semantic similarity (positive control: r=0.906, d=4.27)
- **FAILS FOR**: Logical validity, adversarial NLI (ANLI R3: r=-0.10, d=-0.20)

## Pre-Registration

- **Hypothesis**: R < 0.5 correlation with ground truth in adversarial/NLI domains
- **Prediction**: R cannot distinguish logical contradictions from entailments
- **Falsification**: R > 0.7 correlation in any adversarial NLI domain
- **Threshold**: correlation < 0.5

## Experimental Results (2026-01-27)

### Test 1: SNLI Dataset (n=500)
| Label | Mean Similarity | Std |
|-------|-----------------|-----|
| Entailment | 0.661 | 0.148 |
| Neutral | 0.525 | 0.184 |
| Contradiction | 0.308 | 0.206 |

- Pearson r: 0.706 (p < 1e-53)
- Cohen's d: 1.97
- **UNEXPECTED**: R CAN distinguish on standard SNLI (topical differences)

### Test 2: ANLI R3 (Adversarial, n=300)
| Label | Mean Similarity | Std |
|-------|-----------------|-----|
| Entailment | 0.498 | 0.177 |
| Neutral | 0.486 | 0.206 |
| Contradiction | 0.536 | 0.200 |

- Pearson r: -0.100 (p = 0.14, NOT SIGNIFICANT)
- Cohen's d: -0.20
- **CONFIRMED**: R FAILS completely on adversarial NLI

### Test 3: Positive Control - Topical Consistency (n=200)
- Aligned pairs: 0.659 +/- 0.146
- Misaligned pairs: 0.056 +/- 0.136
- Pearson r: 0.906 (p < 1e-149)
- Cohen's d: 4.27
- **PASS**: R works for topical alignment

## Critical Insight

The SNLI vs ANLI difference is crucial:

1. **SNLI contradictions** often change topics/contexts entirely (hence R detects)
2. **ANLI contradictions** are adversarially crafted to maintain high semantic overlap

This proves R measures **SEMANTIC/TOPICAL COHERENCE**, not **LOGICAL VALIDITY**.

Example: "The cat is on the mat" vs "No cat is on the mat"
- Same topic (cat, mat) = high semantic similarity
- Logical contradiction = R cannot detect

## Implications for R

1. **R is NOT a truth detector** - it measures coherence, not validity
2. **Adversarial attacks bypass R** - crafted contradictions maintain high R
3. **R requires complementary logic checks** for safety-critical applications
4. **R's domain**: consensus detection, drift monitoring, topical alignment

## Files
- Test script: `experiments/open_questions/q16/run_q16_real_data.py`
- Results: `experiments/open_questions/q16/q16_results.json`

## References
- Q10: Established R detects topical alignment, not logical contradictions
- SNLI: Stanford NLI (stanfordnlp/snli)
- ANLI: Adversarial NLI Round 3 (facebook/anli)
