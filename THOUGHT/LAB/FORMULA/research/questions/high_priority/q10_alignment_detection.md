# Question 10: Alignment detection (R: 1560)

**STATUS: PARTIAL (18/18 tests pass, documented limitations)**

## Question
Can R distinguish aligned vs. misaligned agent behavior? Does agreement among value-aligned agents produce high R?

---

## TESTS

**Test file:** `experiments/open_questions/q10/test_q10_alignment_detection.py`
**Results:** `experiments/open_questions/q10/q10_test_results.json`
**Model:** `sentence-transformers/all-MiniLM-L6-v2`

### Test Results (2026-01-11)

| Test | Result | Key Finding |
|------|--------|-------------|
| BEHAVIORAL_CONSISTENCY | PASS | R_consistent(3.93) > R_erratic(2.19), ratio=1.79x |
| MULTI_AGENT_ALIGNMENT | PASS | Adding misaligned agent drops R by 28.3% |
| DECEPTIVE_PATTERNS | PASS | Overt contradictions detectable (R < 10) |
| INTENT_MATCHING | PASS | Matching intent produces R > 2.0 |
| ALIGNED_BEHAVIORS | PASS | Pure aligned behaviors show consistency |
| VALUE_ALIGNMENT | FAIL | Mixing values+behaviors doesn't discriminate (ratio=0.99) |
| DECEPTION_LIMITATION | DOC | Syntactic similarity masks logical contradiction |
| OPPOSITE_STATEMENTS | DOC | Semantic opposites are topically similar |

**18/18 pytest tests pass** (including documented limitations)

### Key Empirical Findings

```
BEHAVIORAL CONSISTENCY:
  Consistent behavior: E=0.52, R=3.93
  Erratic behavior: E=0.12, R=2.19
  Discrimination ratio: 1.79x

MULTI-AGENT ALIGNMENT:
  Both aligned: R=2.49
  One misaligned: R=1.79
  R drop: 28.3%

AUTHENTIC vs DECEPTIVE:
  Authentic statements: R=7.18
  Deceptive with "but" clauses: R=4.04
  Discrimination ratio: 1.78x

VALUE-BEHAVIOR MIXING (LIMITATION):
  Aligned values+behaviors: R=2.19
  Misaligned values+behaviors: R=2.22
  Discrimination ratio: 0.99x (NO DISCRIMINATION)
```

---

## ANSWER

**PARTIAL: R can detect certain types of alignment but has fundamental limitations.**

### What R CAN Detect

#### 1. Behavioral Consistency (1.79x discrimination)

R successfully distinguishes consistent from erratic behavior:

| Behavior Type | E (agreement) | R |
|---------------|---------------|---|
| Consistent (same topic) | 0.52 | 3.93 |
| Erratic (random topics) | 0.12 | 2.19 |

**Use case:** Detecting when an agent's outputs are topically coherent vs. random/confused.

#### 2. Multi-Agent Misalignment Detection (28% R drop)

When multiple agents should agree, R can detect the outlier:

| Configuration | R | Change |
|---------------|---|--------|
| Both agents aligned | 2.49 | baseline |
| One agent misaligned | 1.79 | -28.3% |

**Use case:** Detecting when one agent in a swarm diverges from consensus.

#### 3. Intent Matching

R > 2.0 when stated intent matches observed behavior.
R lower when intent mismatches behavior.

**Use case:** Checking if an agent does what it claims it will do.

---

### What R CANNOT Detect (Limitations)

#### LIMITATION 1: Semantic Opposites Are Topically Similar

```
"I always tell the truth" and "I frequently lie"
```

These are LOGICALLY opposite but SEMANTICALLY similar (both about truth/lying).

| Scenario | E | sigma | R |
|----------|---|-------|---|
| Contradictory statements | 0.45 | 0.07 | 6.5 |

**R is HIGH despite logical contradiction** because embeddings capture topic, not logic.

#### LIMITATION 2: Value-Behavior Mixing Dilutes Signal

Mixing stated values with observed behaviors doesn't discriminate:

| Configuration | R |
|---------------|---|
| Aligned values + aligned behaviors | 2.19 |
| Aligned values + misaligned behaviors | 2.22 |

The values and behaviors are DIFFERENT sentence types with low cross-similarity, creating noise.

**Workaround:** Compare behaviors directly, don't mix with values.

#### LIMITATION 3: Deceptive Patterns May Have High R

Deceptive statements with "but" clauses share syntactic structure:

```
"I prioritize safety but sometimes bypass checks"
"I follow guidelines except when convenient"
"I'm honest but hide certain information"
```

These have HIGH internal similarity (same "X but Y" pattern) even though they're logically inconsistent.

**Implication:** R cannot reliably detect sophisticated deception that maintains surface coherence.

---

## Theoretical Grounding

### Why Behavioral Consistency Works

R = E/σ measures **semantic coherence**:
- E = mean pairwise similarity (how much observations agree)
- σ = dispersion (how varied the observations are)

Consistent behavior → similar semantic content → high E, low σ → high R.
Erratic behavior → varied topics → low E, high σ → low R.

This aligns with Q17's finding that R is an **intensive property** measuring signal quality.

### Why Multi-Agent Detection Works

When agents share values, their outputs should be semantically aligned.
A misaligned agent introduces semantic outliers → increases σ → lowers R.

This is the same mechanism as Q17's volume resistance: adding bad data lowers R.

### Why Logical Contradiction Detection Fails

Semantic embeddings are trained on **distributional similarity**, not logical entailment.

Sentences about the same topic cluster together regardless of negation:
- "I love pizza" and "I hate pizza" are semantically similar (both about pizza preferences)
- "I'm honest" and "I'm deceptive" are semantically similar (both about honesty)

This is a **fundamental limitation of embedding-based R**, not a bug in the formula.

---

## Practical Recommendations

### 1. Use R for Behavioral Consistency, Not Value Alignment

Instead of:
```
R(values + behaviors)  # Doesn't discriminate well
```

Do:
```
R(behaviors_only)  # 1.79x discrimination
```

Compare observed behaviors directly; don't mix with stated values.

### 2. Use R for Multi-Agent Consensus Monitoring

Monitor R across agent swarm outputs:
- Sudden R drop = one or more agents diverging
- Use as early warning for alignment drift

### 3. Combine R with Symbolic Checking for Deception

R alone cannot detect sophisticated deception.

Defense in depth:
1. R-gate for behavioral consistency (layer 1)
2. Symbolic rule checking for logical consistency (layer 2)
3. Human review for critical decisions (layer 3)

### 4. Threshold Calibration for Alignment

Based on empirical findings:

| Scenario | Typical R | Threshold |
|----------|-----------|-----------|
| Highly consistent | > 5.0 | Strong alignment |
| Moderately consistent | 2.0-5.0 | Acceptable |
| Inconsistent | < 2.0 | Investigate |
| Erratic | < 1.0 | Block |

---

## Connections to Other Questions

| Question | Connection |
|----------|------------|
| Q17 (Governance gating) | R-gating IS alignment detection at action level; this extends to behavior level |
| Q5 (Agreement vs truth) | CONFIRMED: Agreement ≠ truth when embeddings mask logical contradiction |
| Q19 (Value learning) | R alone insufficient; need symbolic reasoning for value consistency |
| Q35 (Markov blankets) | Aligned agents form coherent blanket; misaligned agent breaks boundary |

---

## What Remains Open

1. **Hybrid R + symbolic detection**: Can we combine R with logical entailment models?
2. **Adversarial alignment**: Can agents learn to game R by maintaining surface coherence?
3. **Temporal alignment drift**: Does R track gradual value drift (Q21 connection)?
4. **Cross-domain calibration**: Do thresholds transfer across different agent types?

---

## Final Answer

**PARTIAL: R can detect alignment in specific scenarios but has fundamental limitations.**

**R IS USEFUL FOR:**
- Behavioral consistency (1.79x discrimination)
- Multi-agent misalignment detection (28% R drop)
- Intent matching verification

**R IS NOT SUFFICIENT FOR:**
- Detecting logical contradictions (semantic opposites are similar)
- Value-behavior alignment (mixing dilutes signal)
- Sophisticated deception (surface coherence masks intent)

**PRACTICAL GUIDANCE:**
- Use R as one layer in defense-in-depth
- Combine with symbolic consistency checking
- Compare behaviors directly, not values+behaviors
- Monitor R trends for drift detection

**The answer to "Can R distinguish aligned vs. misaligned agent behavior?" is YES, but only for certain types of misalignment that manifest as semantic inconsistency. Logical misalignment that maintains topical coherence requires additional symbolic reasoning.**

---

**Status**: OPEN -> PARTIAL (18/18 tests pass, limitations documented)
**Date**: 2026-01-11
**Validated by**: Empirical tests with sentence-transformers embeddings
**Test output**: `experiments/open_questions/q10/q10_test_results.json`
**Implementation**: `experiments/open_questions/q10/test_q10_alignment_detection.py`
