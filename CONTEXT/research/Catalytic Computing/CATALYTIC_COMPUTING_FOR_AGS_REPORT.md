---
title: Catalytic Computing for AGS (Full Report)
version: 0.1
last_updated: 2025-12-23
scope: (1) formal catalytic computing results (complexity theory) (2) engineering patterns for AGS
style: agent-readable, reversible, boundaries explicit
---

# 0. Executive summary

Catalytic computing (also called catalytic space) is a space-bounded computation model with two kinds of memory:

1) Clean space: a small amount of blank working memory the algorithm can use freely.
2) Catalytic space: a much larger memory that starts in an arbitrary state and must be returned to exactly that state at the end.

Surprisingly, that “full, arbitrary, but restorable” memory can still help computation, even though it cannot store new information long-term. Formally, it can enable computations that appear to exceed what clean space alone can do.

For AGS, catalytic computing is useful in two ways:
- As real theory: it explains why “memory that is already full” can still be computationally powerful under strict restoration rules.
- As an engineering pattern: it suggests building systems that use large state (disk, caches, indexes) as temporary scratch under a hard restoration guarantee, while keeping the “clean context” minimal.

This report separates what is theory from what is an AGS design analogy so future agents do not hallucinate implementation details.

---

# 1. Formal definition (theory)

A catalytic-space computation has:
- clean space s(n): small, writable, starts blank (or in a known state)
- catalytic space c(n): large, writable, starts as an arbitrary bitstring, and must be restored to that exact bitstring when the computation ends

Constraints that matter:
- The algorithm must work for any initial catalytic content (possibly incompressible).
- The algorithm cannot “cheat” by permanently encoding new information into catalytic space because it must restore it.

In the paper’s framing, this is like temporarily shuffling a hard drive full of data while promising to put every bit back. The drive can act like a catalyst: it enables the computation but remains unchanged in the end.

---

# 2. Key results (theory)

From the original STOC 2014 line of work (Buhrman, Cleve, Koucky, Loff, Speelman):

- Catalytic space can be used nontrivially: they show how to compute uniform TC^1 circuits with only logarithmic clean space (with catalytic space available). This is notable because TC^1 can compute things like determinant, which is not known to be in logspace. [1]
- Upper bounds: catalytic logspace is contained in ZPP (randomized polynomial time with zero error). [1]
- Relativized separations: they construct an oracle world where catalytic logspace equals PSPACE, showing the model can be very powerful relative to an oracle. [1]
- Limits: under ETH, SAT cannot be computed in catalytic sub-linear space. [1]

Interpretation:
- The “restoration constraint” does not kill usefulness. It forces algorithms to use reversible, structured transformations, often with algebraic tricks, randomness, or carefully designed encodings.

---

# 3. Tree evaluation and why catalytic ideas mattered

Tree Evaluation (TreeEval) was introduced as a candidate for separating P from L by “composition” style lower bounds. The intuition was that storing values for later should not also help compute new values. The catalytic model was created to challenge that assumption.

Later work applied catalytic ideas to TreeEval and found unexpectedly low-space algorithms:
- Cook and Mertz (2020) showed TreeEval can be solved in space O(log^2 n / log log n). [2]
- Cook and Mertz (2023 preprint, STOC 2024 publication) improved this to O(log n · log log n), strengthening the case that Tree Evaluation may be in L. [2][3]
- Popular exposition: Quanta describes how catalytic ideas were adapted for tree evaluation and how this shifted expectations about whether TreeEval separates P from L. [4]

---

# 4. What catalytic computing is NOT (important boundaries)

Catalytic computing is not:
- “Just compression.” The catalytic bits may be incompressible, and the algorithm must still work.
- “Free scratch space.” You can write to it, but you must undo those writes without having stored a full undo log in clean space.
- “A license to mutate Canon.” In AGS terms, catalytic patterns are about reversible workspace operations, not editing authoritative sources casually.

---

# 5. Engineering translation for AGS (pattern library)

Below are practical patterns that capture the spirit of catalytic space without pretending we are reproducing theoretical guarantees.

## 5.1 Clean Context vs Catalytic Store

- Clean context (LLM tokens): keep minimal. This is your LITE pack.
- Catalytic store (disk, indexes, caches): huge. You can touch it, but it must be restore-safe.

AGS mapping:
- LITE pack should contain:
  - laws, maps, contracts
  - symbolic indexes (file and section indexes, skill index)
  - short summaries
  - retrieval instructions
- FULL content lives outside the LITE pack, addressable by hash or path.

## 5.2 Restore guarantee as a first-class artifact

Every operation that uses “big scratch” produces:
- a patch (what changed)
- a restore plan (how to undo)
- a verification check (prove restoration happened)

Practical mechanisms:
- git worktree or temporary checkout
- overlay filesystem (copy-on-write)
- content-addressed cache for generated artifacts

## 5.3 Reversible transforms mindset

When a tool needs scratch:
- Prefer in-place, reversible transforms or transforms with small “witness” data in clean space.
- If not possible, simulate reversibility using an external journal that is allowed by contract (for AGS: under artifact roots only).

Example operations:
- build indexes into `meta/` from source files (no source mutation)
- generate pack manifests, hashes, and allowlists
- plan refactors by producing a migration plan, not edits

## 5.4 “Catalytic compression” for packs (your use case)

Goal:
- Reduce context tokens while keeping deep recovery possible.

Concrete approach:
- Content-addressed snapshot store
  - Each file body stored by sha256 in a cache directory under allowed roots.
- LITE pack includes only:
  - pointers to those shas
  - section-level summaries
  - a deterministic reconstruction recipe for FULL pack

This is closer to “content-addressed retrieval” than literal catalytic computing, but it matches your objective: minimal repeated compute, maximal reuse.

---

# 6. Proposed AGS implementation plan (agent-ready)

## Step 1. Write the canonical note (no code)
- Add a doc: `CANON/CATALYTIC_COMPUTING.md` or `SYSTEM/CATALYTIC.md`
- Define:
  - formal catalytic space model (short)
  - AGS analog (clean context vs catalytic store)
  - strict do-nots (no canon mutation, no surprise writes)

## Step 2. Add a “Catalytic Workspace” contract
- New contract: “Operations may use a temporary workspace, but must end with a clean repo state and only committed outputs under allowed roots.”
- Add critic checks:
  - repo clean check at end (optional)
  - output-root enforcement

## Step 3. Prototype 1: reversible index build
- Index build produces only:
  - `meta/FILE_INDEX.json`
  - `meta/SECTION_INDEX.json`
  - `meta/SKILL_INDEX.json`
- Verify determinism.

## Step 4. Prototype 2: content-addressed body cache
- Store file bodies by hash under allowed roots.
- LITE pack references bodies by hash.
- FULL pack reconstruction pulls bodies from cache.

## Step 5. Evaluate
Metrics:
- LITE pack size (tokens, bytes)
- time to answer navigation queries using only LITE + Cortex
- determinism of index and pack outputs
- rate of “agent got lost” failures (qualitative but trackable)

---

# 7. References (footnotes)

[1] Buhrman, H., Cleve, R., Koucky, M., Loff, B., Speelman, F. “Computing with a full memory: Catalytic space” (2014). PDF: https://iuuk.mff.cuni.cz/~koucky/papers/catalytic.pdf

[2] Cook, J., Mertz, I. “Tree Evaluation is in Space O(log n · log log n)” (version hosted by authors, cites Cook and Mertz 2020 result). PDF: https://iuuk.mff.cuni.cz/~iwmertz/papers/cm25.tree_evaluation_is_in_space_lognloglogn.pdf

[3] ECCC TR23-174. “Tree Evaluation is in Space O(log n · log log n)” (Cook, Mertz, 2023). https://eccc.weizmann.ac.il/report/2023/174/

[4] Quanta Magazine. “Catalytic Computing Taps the Full Power of a Full Hard Drive” (Feb 18, 2025). https://www.quantamagazine.org/catalytic-computing-taps-the-full-power-of-a-full-hard-drive-20250218/

