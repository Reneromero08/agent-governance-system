---
title: AGS Roadmap (Master)
version: 3.0
last_updated: 2025-12-23
scope: Agent Governance System (repo + packer + cortex + CI)
style: agent-readable, task-oriented, minimal ambiguity
source_docs:
  - CONTEXT/archive/planning/REPO_FIXES_TASKS.md
  - CONTEXT/archive/planning/AGS_3.0_ROADMAP.md
---

# Purpose

Make the repo easy for any agent to navigate, verify, and safely modify, while keeping context cost low (LITE packs) and keeping correctness high (contracts, critic, CI).

This roadmap assumes:
- Text outranks code (Canon, AGENTS, maps) and is the primary contract.
- Packs have profiles (FULL vs LITE), with LITE optimized for navigation and governance, not full restoration.
- A Cortex (index database) is the primary navigation layer.
- Determinism and "no surprise writes" are non-negotiable.

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
- P3: optional / long-horizon items

---

# Lane A: Governance coherence (P0)

## A1. Resolve "artifact roots vs logging" contradiction (P0)

Problem:
- Canon allows system artifacts only under approved roots, but multiple sources reference or write to `LOGS/` and `MCP/logs/`.
- Conflicting references include `CANON/CRISIS.md`, `CANON/STEWARDSHIP.md`, `TOOLS/emergency.py`, `MCP/server.py`, and `CANON/CHANGELOG.md`.
- Mitigation exists via `CONTRACTS/_runs/ags_mcp_entrypoint.py` (redirect MCP logs under allowed roots).

Tasks:
- [ ] [P0] Choose the canonical policy: restrict logs to approved roots (recommended) or expand allowed roots to include `LOGS/` and `MCP/logs/`.
- [ ] [P0] Update canon docs to match the chosen policy (`CANON/CONTRACT.md`, `CANON/INVARIANTS.md`, `CANON/CRISIS.md`, `CANON/STEWARDSHIP.md`, `CANON/CHANGELOG.md`).
- [ ] [P0] Update code paths that write logs to comply (`TOOLS/emergency.py`, `MCP/server.py`), using the MCP redirect entrypoint if staying within approved roots.

Exit conditions:
- `TOOLS/critic.py` enforces the chosen output-root policy.
- `CONTRACTS/runner.py` passes.
- CI passes.

## A2. Resolve Canon vs AGENTS context write rule (P0)

Problem:
- Canon and AGENTS disagree on whether agents may modify context.

Tasks:
- [ ] [P0] Amend Canon wording to forbid editing existing records while allowing append-only additions under explicit `CONTEXT/*` areas.
- [ ] [P1] Add critic enforcement for illegal edits to existing context records.

Exit conditions:
- No higher-authority file contradicts a lower-authority file on this rule.
- Critic blocks illegal context edits.

---

# Lane B: Pack profiles and symbolic indexes (P0 to P1)

## B1. Implement `--profile lite` in packer (P0)

Problem:
- LITE packs are required for navigation-first workflows but are not guaranteed by the packer or CI.

Tasks:
- [ ] [P0] Add `--profile lite` to `MEMORY/LLM_PACKER/Engine/packer.py` with FULL output unchanged.
- [ ] [P0] Emit LITE meta outputs: `meta/LITE_ALLOWLIST.json`, `meta/LITE_OMITTED.json`, `meta/LITE_START_HERE.md`, `meta/SKILL_INDEX.json`, `meta/FIXTURE_INDEX.json`, `meta/CODEBOOK.md`, `meta/CODE_SYMBOLS.json`.
- [ ] [P0] Add smoke/CI validation that asserts required LITE meta outputs exist.
- [ ] [P1] Optional: add `--profile test` (fixtures-heavy) for debugging, separate from LITE.

Exit conditions:
- `packer.py --profile lite` produces a pack that an agent can navigate without repo access.
- A validator fails fast if LITE required meta is missing.

## B2. Determinism contract for packs (P1)

Problem:
- Pack determinism is implied but not clearly documented or enforced.

Tasks:
- [ ] [P1] Document which pack files are deterministic vs timestamped, and what is allowed to vary.
- [ ] [P1] Ensure timestamps do not break cache keys (use content hashes as primary identity).
- [ ] [P1] Add `canon_version` and `grammar_version` fields and define mismatch behavior.

Exit conditions:
- Two LITE packs generated from the same repo state have identical content-hash inventories (except explicitly allowed timestamp fields).

---

# Lane C: Index database (Cortex) as primary navigation (P0 to P1)

This is the "every section indexed and summarized" requirement.

Note: See `CONTEXT/research/iterations/2025-12-23-system1-system2-dual-db.md` for the System 1 (fast retrieval) vs System 2 (governance ledger) research model.

Tasks:
- [x] [P0] Emit section-level index (section_id, path, heading, start_line, end_line, hash) as a generated artifact (cortex.json or SECTION_INDEX.json).
- [x] [P0] Add cortex read <section_id> command that resolves section_id via the generated index and prints the exact section content.
- [x] [P0] Add cortex search <query> that returns matching section_ids (and headings) from SECTION_INDEX; do not return raw paths.
- [x] [P1] Harden SECTION_INDEX parsing: ignore headings inside fenced code blocks; ensure stable hashing + ID normalization; ensure output path is under the Cortex generated root.
- [ ] [P0] Add cortex resolve <section_id> to output JSON metadata (section_id, path, start_line, end_line, hash) for toolchain provenance and citations.
- [ ] [P1] Clean up Cortex section indexing hygiene: ensure full normalized path in section_id, move fixtures under CORTEX/fixtures (or equivalent), write SECTION_INDEX under generated root, and ignore .claude/.

## C1. Define the Cortex schema (P0)

Problem:
- Cortex data model is underspecified for section-level navigation and summaries.

Tasks:
- [ ] [P0] Define minimum schema tables for files, sections, summaries, and pointers with required fields and meanings.
- [ ] [P0] Document the schema in `CORTEX/SCHEMA.md` (or equivalent).

Exit conditions:
- Schema is documented and unambiguous.
- A single CLI command can build the DB from a repo checkout.

## C2. Build the indexer (P0)

Problem:
- Section-level indexing and stable anchors are not guaranteed.

Tasks:
- [ ] [P0] Implement indexer logic to parse markdown headings and stable anchors.
- [ ] [P0] Build `meta/FILE_INDEX.json` and `meta/SECTION_INDEX.json` from the indexer output.
- [ ] [P1] Ensure index build is deterministic and does not modify authored files unless explicitly enabled.

Exit conditions:
- Index build is deterministic.
- Index build does not modify authored files unless explicitly enabled.

## C3. Summarization layer (P1)

Problem:
- Summaries are needed for navigation but not defined or stored.

Tasks:
- [ ] [P1] Add a summarizer that writes summaries into the DB (not into source files).
- [ ] [P1] Define max summary length and "summary freshness" policy.
- [ ] [P1] Extend `CORTEX/query.py` with section-level queries (`find`, `get`, `neighbors`).

Exit conditions:
- "Read order" navigation works from the DB without opening many files.
- Summaries are versioned and can be regenerated safely.

---

# Lane D: Skills library expansion (P1)

## D1. Formalize skill contracts (P1)

Problem:
- Skill structure is uneven and not always fixture-verified.

Tasks:
- [ ] [P1] Ensure each skill has `SKILL.md`, `version.json`, and fixtures (positive + negative).
- [ ] [P1] Ensure runner can execute skill fixtures in CI.

Exit conditions:
- `CONTRACTS/runner.py` can run skill fixtures in CI.

## D2. Add the "dev browser" skill (P1)

Goal:
- A local web UI that browses Cortex indexes and supports deterministic planning.

Tasks:
- [ ] [P1] Build a read-only UI for files, sections, summaries, and graph relations.
- [ ] [P1] Export plans as a migration plan file (no direct edits).
- [ ] [P1] Optional: apply plans via a separate deterministic, fixture-tested migration skill.

Exit conditions:
- You can answer "Where is X?" and "What connects to X?" instantly via the UI.
- Applying a plan is opt-in and audited.

---

# Lane E: CI and enforcement (P0 to P2)

## E0. CI correctness hard-stops (P0)

Problem:
- CI can mask dependency failures and writes artifacts to disallowed roots.

Tasks:
- [ ] [P0] Remove `|| true` from dependency installs in `.github/workflows/contracts.yml` so missing deps fail fast.
- [ ] [P0] Stop writing `BUILD/escape-check.json` in CI; write artifact-escape-hatch outputs under an allowed root (e.g., `CONTRACTS/_runs/`).

Exit conditions:
- CI fails loudly on missing dependencies.
- No CI artifacts are written under `BUILD/`.

## E1. Fix or remove broken workflow (P1)

Problem:
- `/.github/workflows/governance.yml` runs unsupported commands and lacks Python/deps setup.

Tasks:
- [ ] [P1] Either implement `TOOLS/critic.py --diff` or update the workflow to run supported commands.
- [ ] [P1] Ensure workflow sets up Python and installs `requirements.txt`.
- [ ] [P1] Optionally merge into `contracts.yml` to keep a single source of CI truth.

Exit conditions:
- Workflows reflect reality and are not decorative.

## E2. Output-root enforcement (P1)

Problem:
- Output-root compliance is not enforced across the repo.

Tasks:
- [ ] [P1] Extend `TOOLS/critic.py` to detect artifacts written outside allowed roots.
- [ ] [P1] Ensure enforcement aligns with the policy chosen in Lane A1.
- [ ] [P1] Run the enforcement in CI and locally.

Exit conditions:
- Violations fail builds.

## E3. Codebook build check correctness (P1)

Problem:
- `TOOLS/codebook_build.py --check` can report false "up to date" results.

Tasks:
- [ ] [P1] Make `--check` compare generated output against `CANON/CODEBOOK.md`.

Exit conditions:
- `--check` fails when codebook output drifts.

## E4. Hygiene (P2)

Problem:
- Documentation drift increases confusion for agents and humans.

Tasks:
- [ ] [P2] Tidy `CANON/CHANGELOG.md` heading drift (e.g., duplicate headings).
- [ ] [P2] Align `README.md` wording with current structure (e.g., "six layers" mismatch).

Exit conditions:
- Reduced confusion for agents and humans.

---

# Lane F: Catalytic computing (research -> integration) (P1 to P3)

This lane is optional until fundamentals above are stable.

Note: See `CONTEXT/research/Catalytic Computing/CATALYTIC_COMPUTING_FOR_AGS_REPORT.md` for the formal theory vs AGS analogy.
Note: See `CONTEXT/research/Catalytic Computing/CMP-01_CATALYTIC_MUTATION_PROTOCOL.md` for the draft engineering contract (not yet Canon).

## F1. Canonical note: "Catalytic Computing for AGS" (P1)

Deliverable:
- A dedicated document defining catalytic computing (formal) and the AGS analog (engineering), plus strict boundaries (metaphor vs implementation).

Tasks:
- [ ] [P1] Write the canonical catalytic computing note with explicit boundaries and non-goals.

Exit conditions:
- Any agent can read it and understand what "catalytic compression" means without inventing details.

## F2. Prototype: "Catalytic Scratch Layer" (P3)

Goal:
- Enable high-impact operations using large disk state as scratch while guaranteeing restoration.

Tasks:
- [ ] [P3] Define the scratch workflow (worktree/overlay + patch set + restore plan).
- [ ] [P3] Prototype the scratch layer with deterministic outputs under allowed roots.

Exit conditions:
- A full run ends with a clean git state while still producing required outputs.

## F3. Prototype: "Catalytic Context Compression" (P3)

Goal:
- Keep LITE packs minimal while enabling deep recovery when needed.

Tasks:
- [ ] [P3] Define content-addressed cache strategy and retrieval instructions.
- [ ] [P3] Prove FULL pack reconstruction from hashes and pointers.

Exit conditions:
- LITE pack remains small and stable.
- FULL pack is reconstructible.

---

# Research References (Non-Binding)

- `CONTEXT/research/iterations/2025-12-23-system1-system2-dual-db.md` - System 1 vs System 2 research model for cortex retrieval and governance ledger concepts.
- `CONTEXT/research/Catalytic Computing/CATALYTIC_COMPUTING_FOR_AGS_REPORT.md` - Formal catalytic computing theory and AGS analogy framing.
- `CONTEXT/research/Catalytic Computing/CMP-01_CATALYTIC_MUTATION_PROTOCOL.md` - Draft engineering protocol for catalytic mutation (non-Canon).

# Definition of Done (Global)

- [ ] [P0] `python TOOLS/critic.py` passes
- [ ] [P0] `python CONTRACTS/runner.py` passes
- [ ] [P0] CI workflows pass on PR and push (as applicable)
