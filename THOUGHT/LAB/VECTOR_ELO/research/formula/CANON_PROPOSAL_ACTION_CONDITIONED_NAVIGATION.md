# CANON_PROPOSAL_ACTION_CONDITIONED_NAVIGATION.md

## Status
**Candidate for Canon (Scoped, Conditional)**

This document records a validated result about navigation using the R-formula, with explicit scope limits and failure modes.

---

## Core Finding (Invariant)

**R provides directional navigation ONLY when evaluated action-conditionally.**

> Node-level R is not a reliable navigator.  
> Transition-level R(s, a) can be.

Formally:

R must be computed as a function of the **state–action pair**, not the state alone.

---

## What Works

### Option B: ΔR Steering (Action-Conditioned Navigation)

Define R per transition:

R(s, a) = E(s, a) / ∇S(s, a) · σ^Df(s, a)

Where all terms are **action-conditioned**.

#### Empirical Results
- Trap graph v1: **73% success**
- Greedy similarity: **0%**
- Beam similarity: **0%**
- Option A (gate-only): **0%**

This establishes true directional navigation.

---

## What Does NOT Work

### Node-Level R
- Ranking nodes by R collapses to similarity.
- Fails all adversarial navigation benchmarks.

### Option A Alone (Gate + Similarity Steering)
- Effective for stopping, budgeting, and efficiency.
- **Cannot fix incorrect directionality.**
- Fails completely in adversarial trap graphs.

---

## Critical Component Analysis (Ablation)

| Component | Removal Effect | Status |
|---------|---------------|--------|
| ∇S (dispersion) | -73% (→ 0%) | **CRITICAL** |
| raw similarity | -73% (collapse to greedy) | CRITICAL |
| Df | -23% | Moderate |
| σ shape | <1% | Non-critical |

**∇S is the engine.**  
Without dispersion detection, navigation fails.

---

## Failure Characteristics

- 92.9% of failures occur in **trap basins**
- Fewer expansions than baselines (16.9 vs 26)
- No lookahead, no cheating, no extra budget

This confirms efficiency and honesty of the signal.

---

## Generalization Limits

- Works on **explicit trap graph family (v1)**
- Does **not** generalize to:
  - lattice graphs
  - hub-dominated graphs
- Reason: ∇S becomes non-discriminative when dispersion is flat

**Conclusion:** Graph-family dependent, not universal.

---

## Canon Law (Proposed)

**Law: Action-Conditioned Navigation**

> R must be evaluated as R(s, a) to provide directional navigation.  
> Node-level R is not a reliable navigator.  
> ∇S (dispersion / ambiguity detection) is the critical component.  
> Applicability is graph-family dependent.

---

## Operational Guidance for AGS

### Use Option A when:
- Similarity is mostly correct
- Budget, stopping, or backtracking control is needed

### Use Option B when:
- Greedy similarity fails
- Local maxima / honey pots exist
- Structural cues predict future reward

### Required Safeguards
- Detect flat ∇S regimes
- Fall back to exploration or alternative cues
- Never claim universality

---

## Next Hardening Steps (Non-Optional)
1. Multi-scale ∇S (1-hop, 2-hop, delta)
2. Explicit trap-basin escape logic
3. Fallback steering when ∇S variance < ε
4. Adversarial benchmark inclusion required

---

## Canonization Recommendation

**YES, but scoped.**

This is a real navigation result with:
- Clear mechanism
- Clear limits
- Clear failure modes

Canonize as a **conditional law**, not a universal principle.
