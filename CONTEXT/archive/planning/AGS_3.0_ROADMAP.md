Superseded by AGS_ROADMAP_MASTER.md

---
title: AGS Roadmap (Updated)
version: 0.1
last_updated: 2025-12-23
scope: Agent Governance System (repo + packer + cortex)
style: agent-readable, task-oriented, minimal ambiguity
---

# Purpose

Make the repo easy for any agent to navigate, verify, and safely modify, while keeping context cost low (LITE packs) and keeping correctness high (contracts, critic, CI).

This roadmap assumes your current direction:
- Text outranks code (Canon, AGENTS, maps) and is the primary contract.
- Packs have profiles (FULL vs LITE), with LITE optimized for navigation and governance, not full restoration.
- A Cortex (index database) is the primary navigation layer.
- Determinism and “no surprise writes” are non-negotiable.

# Current state snapshot (what you already have)

From the SPLIT_LITE pack pointers, the system expects these major components to exist in-repo:
- Governance: `repo/AGENTS.md`, `repo/CANON/*`
- Maps: `repo/MAPS/ENTRYPOINTS.md`
- Contracts: `repo/CONTRACTS/runner.py`
- Skills: `repo/SKILLS/*/SKILL.md`
- Tools: `repo/TOOLS/critic.py`
- Cortex interface: `repo/CORTEX/query.py`
- Packer engine: `repo/MEMORY/LLM_PACKER/Engine/packer.py`
- MCP seam: `repo/MCP/server.py`

(Those pointers are present in your LITE pack index and root stubs.)

# Roadmap structure

Work is grouped into lanes with explicit exit conditions.

Legend:
- P0: contradictions, violations, missing enforcement
- P1: correctness gaps that will cause drift
- P2: hygiene, clarity, polish

---

# Lane A: Governance coherence (P0)

## A1. Resolve “artifact roots vs logging” contradiction (P0)

Problem:
- Canon says generated artifacts must only be written under approved roots.
- Other files reference or write to `LOGS/` and `MCP/logs/`, which violates the contract unless those roots are explicitly allowed.

Tasks:
- Choose ONE canonical policy:
  1) Allow logging under an approved artifact root only (recommended), OR
  2) Expand allowed roots to include `LOGS/` and `MCP/logs/` (increases drift risk).
- Update Canon references to match policy.
- Update tools that write logs to comply (or route logs via the existing redirect entrypoint).

Exit conditions:
- `TOOLS/critic.py` enforces the chosen output-root policy.
- `CONTRACTS/runner.py` passes.
- CI passes.

## A2. Resolve Canon vs AGENTS context write rule (P0)

Problem:
- Canon and AGENTS disagree on whether agents may modify context.

Task:
- Amend Canon wording so it forbids editing existing records but allows appending new records under an explicit `CONTEXT/*` area.

Exit conditions:
- No higher-authority file contradicts a lower-authority file on this rule.
- Critic includes a check for illegal edits to existing context records.

---

# Lane B: Pack profiles and symbolic indexes (P0 to P1)

## B1. Implement `--profile lite` in packer (P0)

Tasks:
- Add a LITE profile to `MEMORY/LLM_PACKER/Engine/packer.py`:
  - FULL output unchanged.
  - LITE emits governance + maps + indexes + pointers, omits most implementation bodies.
- Emit LITE meta outputs:
  - `meta/LITE_ALLOWLIST.json`
  - `meta/LITE_OMITTED.json`
  - `meta/LITE_START_HERE.md`
  - `meta/SKILL_INDEX.json`
  - `meta/FIXTURE_INDEX.json`
  - `meta/CODEBOOK.md`
  - `meta/CODE_SYMBOLS.json`
- Add smoke validation that asserts the above exist for LITE packs.

Exit conditions:
- `packer.py --profile lite` produces a pack that an agent can navigate without repo access.
- A validator fails fast if LITE required meta is missing.

## B2. Determinism contract for packs (P1)

Tasks:
- Document which files are deterministic and which are timestamped.
- Ensure timestamps do not break cache keys (use content hashes as primary identity).
- Add `canon_version` and `grammar_version` fields and define mismatch behavior.

Exit conditions:
- Two LITE packs generated from the same repo state have identical content-hash inventories (except explicitly allowed timestamp fields).

---

# Lane C: Index database (Cortex) as primary navigation (P0 to P1)

This is the “every section indexed and summarized” requirement.

## C1. Define the Cortex schema (P0)

Minimum viable tables:
- files: path, sha, size, type, pack_profile_visibility
- sections: file_path, header_level, anchor_id, start_line, end_line, title
- summaries: section_anchor_id, summary_text, summary_version, model_info (optional)
- pointers: section_anchor_id -> (related sections, skills, contracts, canon rules)

Exit conditions:
- Schema is documented in `CORTEX/SCHEMA.md` (or equivalent).
- A single CLI command can build the DB from a repo checkout.

## C2. Build the indexer (P0)

Tasks:
- Implement an indexer that:
  - Parses markdown headings and stable anchors.
  - Builds `meta/FILE_INDEX.json` and `meta/SECTION_INDEX.json`.
  - Optionally emits per-file comment markers only if policy allows it (do not spam source files by default).

Exit conditions:
- Index build is deterministic.
- Index build does not modify authored files unless explicitly enabled.

## C3. Summarization layer (P1)

Tasks:
- Add a summarizer (offline or remote) that writes summaries into the DB, not into source files.
- Define a max summary length per section and a “summary freshness” policy.
- Provide a query interface in `CORTEX/query.py`:
  - `find(section_title | symbol | path)`
  - `get(section_anchor_id)`
  - `neighbors(section_anchor_id)` (pointers graph)

Exit conditions:
- “Read order” navigation works from the DB without opening many files.
- Summaries are versioned and can be regenerated safely.

---

# Lane D: Skills library expansion (P1)

## D1. Formalize skill contracts (P1)

Tasks:
- Each skill has:
  - `SKILL.md` (what, inputs, outputs, invariants)
  - `version.json`
  - `fixtures/` (positive and negative tests)
- Runner can execute fixtures.

Exit conditions:
- `CONTRACTS/runner.py` can run skill fixtures in CI.

## D2. Add the “dev browser” skill (your note) (P1)

Goal:
- A local web UI that browses Cortex indexes (files, sections, summaries) and lets you restructure based on tags, folders, links, and graph relations.

Minimum deliverables:
- Read-only browsing first (no mutations).
- Export plans (a migration plan file) rather than direct edits.
- Optional: apply plan via a separate “migration skill” that is deterministic and fixture-tested.

Exit conditions:
- You can answer: “Where is X?” and “What connects to X?” instantly via the UI.
- Applying a plan is opt-in and audited.

---

# Lane E: CI and enforcement (P1 to P2)

## E1. Fix or remove broken workflow (P1)

Tasks:
- Either implement `TOOLS/critic.py --diff`, or update workflow to run supported commands.
- Ensure workflows install dependencies and set up Python.

Exit conditions:
- Workflows reflect reality and are not decorative.

## E2. Output-root enforcement (P1)

Tasks:
- Extend critic to detect artifacts written outside allowed roots.
- Make it run in CI and locally.

Exit conditions:
- Violations fail builds.

## E3. Codebook build check correctness (P1)

Tasks:
- Ensure `TOOLS/codebook_build.py --check` actually compares generated content against the canonical output.

Exit conditions:
- No false “up to date” results.

## E4. Hygiene (P2)

Tasks:
- Fix `CANON/CHANGELOG.md` heading drift.
- Align README descriptions with actual structure.

Exit conditions:
- Reduced confusion for agents and humans.

---

# Lane F: Catalytic computing (research -> integration) (P1 to P2)

This lane is optional until the fundamentals above are stable. It is still worth writing now so agents do not drift when you revisit it.

## F1. Canonical note: “Catalytic Computing for AGS” (P1)

Deliverable:
- A dedicated document that defines catalytic computing (formal) and the AGS analog (engineering), plus strict boundaries (what is metaphor vs what is implemented).

Exit conditions:
- Any agent can read it and understand what you mean by “catalytic compression” without inventing stuff.

## F2. Prototype: “Catalytic Scratch Layer” (P2)

Goal:
- Enable high-impact operations (index build, refactor planning, pack generation) using huge disk state as scratch, while guaranteeing restoration.

Implementation sketch:
- Use a copy-on-write workspace:
  - Git worktree or temporary checkout, or overlay filesystem.
  - Produce a patch set + restore plan.
  - Only commit curated outputs.

Exit conditions:
- A full run ends with the repo in a clean git state (no untracked surprises), while still producing outputs under allowed roots.

## F3. Prototype: “Catalytic Context Compression” (P2)

Goal:
- Keep LITE packs minimal while enabling deep recovery when needed.

Implementation sketch:
- Store large material in a content-addressed cache (hash keyed).
- LITE pack includes:
  - pointers, indexes, and retrieval instructions
  - no large bodies
- FULL pack can be synthesized by retrieving bodies by hash.

Exit conditions:
- LITE pack remains small and stable.
- FULL pack is reconstructible.

---

# Immediate next steps (recommended order)

1) Finish Lane A (P0) so the contract is self-consistent.
2) Finish Lane B (P0) so LITE packs are real and validated.
3) Finish Lane C (P0) so “every section indexed” is actually true via Cortex, not vibes.
4) Only then expand Skills and UI (Lane D).
5) Add CI polish (Lane E).
6) Start catalytic lane (Lane F) when the base is stable.
