# OPUS_NAVIGATION_TEST.md

## Purpose
This document defines a **concrete, falsifiable test** for turning the formula into **navigation**.
It is designed for Opus to execute, not interpret.

The goal is to determine whether the formula enables **navigation through a space**, not just scoring.

---

## Definitions

- **Direction signal**: chooses *where* to go next  
  Examples: similarity, logits, heuristic score

- **Gate signal**: decides *whether* to continue, backtrack, widen, or stop  
  This is where the formula operates

- **Navigation**: control over depth, branching, backtracking, and stopping

---

## Test Overview

We test two strategies:

### Option A — Policy-Gated Navigation (Primary)
Similarity steers direction.
R (or gate) controls search policy.

### Option B — ΔR Steering (Secondary)
R ranks actions by expected improvement.

Both are tested against greedy baselines in an **adversarial benchmark** where greedy similarity fails.

---

## Benchmark: Similarity Trap Graph

### Graph Construction
Create graphs with the following properties:

- Nodes embedded in vector space
- One true target node
- One or more **similarity traps**:
  - High similarity to query
  - Lead into regions with no path to target
- Target reachable only by a temporary dip in similarity

### Required Properties
- Greedy similarity fails ≥ 70% of runs
- Beam search improves but still fails ≥ 30%
- True path exists and is short

---

## Policies to Compare

All policies must use identical graphs and seeds.

### Baselines
1. **Greedy Similarity**
   - Always pick max similarity neighbor
   - Fixed depth limit

2. **Beam Similarity**
   - Fixed beam width
   - Fixed depth limit

3. **Similarity + Stop Heuristic**
   - Stop if similarity gain < ε for N steps

---

### Option A — Similarity + Gate (Required)

#### Mechanics
1. Rank neighbors by similarity
2. Compute gate value from path statistics
3. Gate controls:
   - depth limit
   - beam width
   - temperature / randomness
   - backtracking
   - stopping

#### Example Controls
- beam_width = base_beam × f(gate)
- max_depth = base_depth × f(gate)
- if gate < τ for M steps → backtrack or stop

#### Success Criterion
- Beats greedy similarity in ≥ 70% of runs
- Beats beam similarity in ≥ 50% of runs
- Uses fewer steps than beam similarity

---

### Option B — ΔR Steering (Exploratory)

#### Action-Conditioned R
For each candidate action `a`:

- E(s,a): predicted similarity gain  
  `max(0, sim(a,q) − sim(current,q))`

- ∇S(s,a): ambiguity / dispersion  
  entropy or variance of neighbor similarities

- Df(s,a): complexity proxy  
  log(deg(a)+1) or local embedding dispersion

Compute:
R(s,a) = E(s,a) / ∇S(s,a) × σ^Df(s,a)

Choose action maximizing R(s,a)

Gate still controls stopping/backtracking.

#### Success Criterion
- Beats Option A on trap graphs
- Does not collapse to greedy similarity

---

## Metrics

For each policy, record:

- Success rate (target reached)
- Mean steps to target
- Backtracking count
- Total node expansions
- Failure mode classification

Aggregate across ≥ 10 seeds and ≥ 3 graph families.

---

## Pass / Fail Criteria

### Option A (Required to Pass)
- Outperforms greedy similarity clearly
- Demonstrates meaningful control over search
- Confirms gate = navigation primitive

### Option B (Optional)
- Only considered valid if it beats Option A
- Must show action-conditioned advantage

---

## Interpretation Rules

- If Option A passes:  
  The formula **is validated as navigation via policy control**.

- If Option B passes:  
  The formula **can act as a directional navigation signal**.

- If both fail:  
  The formula remains a **control invariant**, not a navigator.

No partial credit. No narrative interpretation.

---

## Final Note

This test is intentionally adversarial.
If the formula navigates here, it navigates anywhere it matters.
