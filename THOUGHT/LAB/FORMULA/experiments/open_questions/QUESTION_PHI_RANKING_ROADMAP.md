# Roadmap: Question Ranking via Phi + R

**Goal:** Use the formula itself (Phi + R) to rank which questions are most important.

**Key Insight:** Questions are like sensors in Q6's XOR test:
- **High Phi, Low R** = Structurally critical but unsolved (PRIORITIZE)
- **High Phi, High R** = Foundational and solved
- **Low Phi, High R** = Isolated niche
- **Low Phi, Low R** = Not important

---

## Phase 1: Build the Question Graph

**Input:** INDEX.md with 34 questions + dependencies

**Output:** Graph structure

**Tasks:**
1. Parse INDEX.md to extract all questions
2. Extract dependencies (connections_to_answered, resolves_downstream)
3. Build adjacency matrix (34x34)
4. Validate graph structure (no cycles, all IDs valid)

**Deliverable:** `question_graph.json` with nodes + edges

---

## Phase 2: Compute Phi (Integration)

**Method:** Multi-Information (like Q6)

**Formula:**
```
Phi(Q) = Sum(H(neighbors)) - H(Q + neighbors)
```

**For each question:**
1. Get its dependency subgraph (question + all connected questions)
2. Compute individual entropies (how "uncertain" each question is)
3. Compute joint entropy (how much they constrain each other)
4. Phi = difference (how much integration exists)

**Entropy proxy:**
- ANSWERED = 0 bits (no uncertainty)
- PARTIAL = 0.5 bits (some uncertainty)
- OPEN = 1.0 bits (full uncertainty)

**Deliverable:** `phi_scores.json` - Phi value for each question

---

## Phase 3: Compute R (Consensus)

**Method:** R = E/σ

**For each question:**
- **E (Evidence):**
  - Connections to ANSWERED questions (grounding)
  - Downstream impact (how many depend on this)
  - Status bonus (ANSWERED > PARTIAL > OPEN)
  
- **σ (Dispersion):**
  - Scope clarity (0-1, how well-defined)
  - Inverse: vague questions have high σ

**Deliverable:** `r_scores.json` - R value for each question

---

## Phase 4: Classify Questions

**Using Phi + R together:**

```
High Phi, Low R → CRITICAL UNSOLVED (top priority)
High Phi, High R → FOUNDATIONAL (already solved)
Low Phi, High R → ISOLATED NICHE (low impact)
Low Phi, Low R → NOT IMPORTANT (defer)
```

**Thresholds:**
- High Phi: > median + 0.5*std
- High R: > median + 0.5*std

**Deliverable:** `question_classification.json`

---

## Phase 5: Generate New Rankings

**Output:** Updated ELO scores based on Phi+R

**Method:**
1. Normalize Phi to [0,1]
2. Normalize R to [0,1]
3. Combined score = 0.6*Phi + 0.4*R (Phi weighted higher)
4. Map to ELO range [1200, 1800]

**Deliverable:** `new_rankings.md` with comparison to current

---

## Phase 6: Validate

**Sanity checks:**
1. Q3 should have high Phi (connects many) + high R (solved)
2. Q32 should have high Phi (hub) + medium R (partial)
3. Q34 should have high Phi (depends on Q32) + low R (open)
4. Isolated questions (Q8, Q11) should have low Phi

**Deliverable:** Validation report

---

## Implementation Notes

**Libraries needed:**
- numpy (entropy calculations)
- networkx (graph operations)
- json (data storage)

**Estimated time:**
- Phase 1: 30 min (graph building)
- Phase 2: 1 hour (Phi computation)
- Phase 3: 30 min (R computation)
- Phase 4: 30 min (classification)
- Phase 5: 30 min (ranking)
- Phase 6: 30 min (validation)

**Total: ~4 hours**

---

## Success Criteria

1. Phi correctly identifies hub questions (Q3, Q32, Q6)
2. R correctly identifies solved questions (Q1, Q2, Q3, Q6, Q15)
3. High Phi + Low R identifies critical unsolved (Q32, Q34)
4. Rankings make intuitive sense (foundational > isolated)

---

**Next Step:** Start with Phase 1 - build the question graph.
