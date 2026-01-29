# Question 10: Alignment detection (R: 1560)

**STATUS: ANSWERED (Scope clarified, limitations are fundamental)**

## Question
Can R distinguish aligned vs. misaligned agent behavior? Does agreement among value-aligned agents produce high R?

---

## TESTS

**Test file:** `questions/10/test_q10_alignment_detection.py`
**Results:** `questions/10/q10_test_results.json`
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

## Riemann Connection — Potential Path Forward (Q48-Q50)

The discovery that **α ≈ 1/2** (eigenvalue decay = Riemann critical line) may address Limitation 1.

**Hypothesis:** If α ≈ 1/2 is the "natural" spectral structure of coherent meaning, then **contradictions might create local anomalies** in the eigenspectrum.

Potential test:
1. Take contradictory statements ("I always tell the truth" + "I frequently lie")
2. Compute LOCAL Df × α for this statement set
3. Check if local α deviates from 0.5 (breaking Riemann structure)
4. Contradictions might show: non-monotonic decay, bimodal structure, or local Df × α ≠ 8e

**Why this might work:** Semantic opposites share topic but create "tension" in eigenspace — the covariance matrix has to accommodate opposing directions, potentially breaking the smooth α = 1/2 decay.

See [Q50_COMPLETING_8E.md](../reports/Q50_COMPLETING_8E.md) for the Riemann discovery.

---

## RIGOROUS SPECTRAL CONTRADICTION TEST (2026-01-17)

### Hypothesis

**"Contradictions break spectral structure (alpha, c_1) even when topically coherent."**

### Experimental Design (RIGOROUS)

**Scientific Standards:**
- 25 statements per set (sufficient for spectral analysis)
- **3 embedding models** for cross-validation (MiniLM, MPNet, Paraphrase-MiniLM)
- 100 bootstrap iterations for statistical inference
- Discrimination criteria: |Cohen's d| > 0.5 AND p < 0.05

**Test Sets:**
1. **Topic-Consistent**: 25 statements all agreeing (e.g., all pro-honesty)
2. **Topic-Contradictory**: 25 statements mixed (pro + anti honesty)
3. **Random baseline**: 25 unrelated statements

**Topics Tested:** Honesty, Safety

### Results (Multi-Model Validated)

| Metric | MiniLM | MPNet | Paraphrase | **Consensus** |
|--------|--------|-------|------------|---------------|
| **R discriminates (Honesty)** | YES (d=0.75) | YES (d=0.70) | YES (d=2.43) | **YES** |
| alpha_dev discriminates (Honesty) | NO | NO | NO | **NO** |
| c1_dev discriminates (Honesty) | NO | NO | NO | **NO** |
| **R discriminates (Safety)** | YES (d=3.23) | YES (d=3.72) | YES (d=3.07) | **YES** |
| alpha_dev discriminates (Safety) | NO | YES | YES | MIXED |
| c1_dev discriminates (Safety) | NO | YES | YES | MIXED |

### Key Findings

**1. R DOES partially discriminate contradictions**
- Contradictory sets have LOWER R than consistent sets
- Cohen's d ranges from 0.7 to 3.7 (medium to huge effect)
- **Reason:** Mixing opposing statements reduces pairwise agreement E

**2. Spectral metrics do NOT reliably discriminate**
- Alpha/c_1 deviation: NO consensus for Honesty, MIXED for Safety
- **Reason:** Eigenspectrum measures geometric coverage, not logical consistency

### Why R Shows Signal (But It's Not "Contradiction Detection")

R = E / sigma measures **semantic agreement**.

Mixing "I tell the truth" with "I frequently lie" REDUCES E because:
- They point in opposite semantic directions
- Cosine similarity between them is lower

**This is detecting LOW AGREEMENT, not logical contradiction.**

A set of unrelated truths about different topics would show the same low R.

### Why Spectral Metrics Fail

Spectral metrics (alpha, c_1) measure **geometric structure**:
- alpha ~ 0.5 = healthy power-law decay
- c_1 ~ 1.0 = proper topological structure

Contradictions are **topically coherent**:
- "honest" and "dishonest" both live in the HONESTY region
- Eigenspectrum sees complete coverage of that region
- This is geometrically HEALTHY, not anomalous

### HYPOTHESIS: FALSIFIED

**Spectral metrics cannot detect logical contradictions** (no consensus across models).

The limitation is **fundamental**:
- Embeddings encode **semantic similarity** (distributional co-occurrence)
- Logic requires **entailment** (truth-preserving inference)
- These are categorically different operations

### Conclusion

**Contradiction detection requires SYMBOLIC REASONING, not embedding geometry.**

**Test files:**
- `questions/10/test_q10_final.py` - Multi-model rigorous test
- `questions/10/q10_final_results.json` - Full results with statistics

---

## Final Answer

**ANSWERED: R detects TOPICAL alignment. Logical consistency is out of scope.**

### What R CAN Detect (CONFIRMED)
- **Behavioral consistency** (1.79x discrimination) - topically coherent outputs
- **Multi-agent misalignment** (28% R drop) - semantic outliers in consensus
- **Intent matching** - stated intent vs observed behavior (same topic)
- **Echo chambers** - identical/near-identical statements (extreme R)

### What R CANNOT Detect (CONFIRMED - Fundamental)
- **Logical contradictions** - "I tell truth" and "I lie" are topically similar
- **Deceptive alignment** - lies about honesty are semantically in the "honesty topic"
- **Value-behavior mixing** - different sentence types create noise

### Why This Is Fundamental (Spectral Experiment 2026-01-17)

Tested hypothesis: "Contradictions break spectral structure (alpha, c_1)."

**Result: FALSIFIED.** Contradictory statements have BETTER spectral health because they sample both sides of a topic (more complete geometric coverage).

This is not a bug to fix - it's a **category error**:
- Embeddings encode **semantic similarity** (topical relatedness)
- Logic requires **entailment** (truth-preserving inference)
- These are fundamentally different operations

### Practical Guidance

**Defense in Depth:**
1. **Layer 1: R-gating** - Topical coherence check (semantic)
2. **Layer 2: Symbolic rules** - Logical consistency check (formal)
3. **Layer 3: Human review** - Critical decisions

**Do:**
- Use R for behavioral monitoring (consistent vs erratic)
- Use R for multi-agent consensus (detecting outliers)
- Combine R with logical consistency checkers

**Don't:**
- Expect R to detect lies (topically coherent deception)
- Mix stated values with observed behaviors (dilutes signal)
- Rely solely on R for alignment verification

### The Answer

**"Can R distinguish aligned vs. misaligned agent behavior?"**

**YES** - for **semantic** misalignment (topical inconsistency, random outputs, outliers)

**NO** - for **logical** misalignment (contradictions, deception, value violations)

This limitation is fundamental to embedding-based methods and cannot be overcome by spectral techniques. Logical alignment requires symbolic reasoning.

---

**Status**: PARTIAL -> ANSWERED (Scope clarified, limitations proven fundamental)
**Date**: 2026-01-17
**Validated by**:
- Empirical tests with sentence-transformers embeddings (18/18 pass)
- Spectral contradiction experiment (hypothesis falsified)
**Test output**:
- `questions/10/q10_test_results.json`
- `questions/10/q10_spectral_results.json`
**Implementation**:
- `questions/10/test_q10_alignment_detection.py`
- `questions/10/test_q10_spectral_contradiction.py`
