---
title: AGS Roadmap (Master)
version: 3.1
last_updated: 2025-12-28
scope: Agent Governance System (repo + packer + cortex + CI)
style: agent-readable, task-oriented, minimal ambiguity
driver: @F0 (The Living Formula)
source_docs:
  - CONTEXT/archive/planning/AGS_3.0_COMPLETED.md
---

# Purpose

Maximize Resonance ($R$) by aligning Essence ($E$) (Human Intent) with Execution ($f$) through low-Entropy ($\nabla S$) governance.

**Note:** Fully completed lanes (A, B, E) have been archived to `CONTEXT/archive/planning/AGS_3.0_COMPLETED.md` to reduce entropy and focus attention on active work.

# Active Lanes

---

# Lane C: Index database (Cortex) (P0 to P1)

## C2. Build the indexer (P0)

Problem:
- Section-level indexing and stable anchors are not guaranteed.

Tasks:
- [ ] [P0] Implement indexer logic to parse markdown headings and stable anchors.
- [ ] [P0] Build `meta/FILE_INDEX.json` and `meta/SECTION_INDEX.json` from the indexer output.
- [ ] [P1] Ensure index build is deterministic and does not modify authored files unless explicitly enabled.

## C3. Summarization layer (P1)

Problem:
- Summaries are needed for navigation but not defined or stored.

Tasks:
- [ ] [P1] Add a summarizer that writes summaries into the DB (not into source files).
- [ ] [P1] Define max summary length and "summary freshness" policy.
- [ ] [P1] Extend `CORTEX/query.py` with section-level queries (`find`, `get`, `neighbors`).

---

# Lane D: Skills library expansion (P1)

## D2. Add the "dev browser" skill (P3)

Goal:
- A local web UI that browses Cortex indexes and supports deterministic planning.

Tasks:
- [ ] [P3] Build a read-only UI for files, sections, summaries, and graph relations.
- [ ] [P3] Export plans as a migration plan file (no direct edits).
- [ ] [P3] Optional: apply plans via a separate deterministic, fixture-tested migration skill.

---

# Lane F: Catalytic computing (Research -> Integration) (P1 to P3)

## F1. Canonical note: "Catalytic Computing for AGS" (P1)

Deliverable:
- A dedicated document defining catalytic computing (formal) and the AGS analog (engineering), plus strict boundaries (metaphor vs implementation).

Tasks:
- [/] [P1] Write the canonical catalytic computing note with explicit boundaries and non-goals.

## F2. Prototype: "Catalytic Scratch Layer" (P3)

Goal:
- Enable high-impact operations using large disk state as scratch while guaranteeing restoration.

Tasks:
- [ ] [P3] Define the scratch workflow (worktree/overlay + patch set + restore plan).
- [ ] [P3] Prototype the scratch layer with deterministic outputs under allowed roots.

## F3. Prototype: "Catalytic Context Compression" (P3)

Goal:
- Keep LITE packs minimal while enabling deep recovery when needed.

Tasks:
- [ ] [P3] Define content-addressed cache strategy and retrieval instructions.
- [ ] [P3] Prove FULL pack reconstruction from hashes and pointers.

---

# Lane G: The Living Formula Integration (P1)

Goal: Deeply measure and optimize Resonance ($R$).

## G1. Metric Definition (P1)
- [ ] Define precise metrics for Essence, Entropy, and Fractal Dimension within the codebase.

## G2. Feedback Loops (P2)
- [ ] Implement feedback mechanisms where agents report "Resonance" of their tasks.

# Definition of Done (Global)

- [ ] [P0] `python TOOLS/critic.py` passes
- [ ] [P0] `python CONTRACTS/runner.py` passes
- [ ] [P0] CI workflows pass on PR and push (as applicable)
