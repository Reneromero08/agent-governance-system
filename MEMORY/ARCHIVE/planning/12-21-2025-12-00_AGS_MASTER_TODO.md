<!-- CONTENT_HASH: 06501e404f5eb4a3b4b47c18b9e9776f46effee63986a146655ede97bd9c716c -->

# AGS Master TODO
_generated: 2025-12-21_

Legend:
- **P0** = must-fix (correctness / prevents drift / blocks release)
- **P1** = should-fix (stability, DX, scaling)
- **P2** = nice-to-have (later polish / optional power)

---

## Phase 1: Systemic fixes (current roadmap)
This section is strictly “make the existing roadmap real” and close the holes required for it to be true.

### v0.1 — Foundation (P0 heavy)
- [x] **P0** Finalize CANON as working spec (clear authority gradient, minimal ambiguity). _(Fixed: aligned authority gradient, added 4 new invariants, completed glossary with 9 new terms, added governance fixtures)_
- [x] **P0** Ensure **Cortex (shadow index)** exists and is part of the standard workflow (build + query) instead of raw filesystem scanning. _(Fixed: ran `cortex.build.py` to populate index)_
- [x] **P0** Ensure **basic skills exist** and are runnable with fixtures (at least one minimal skill end-to-end). _(Fixed: created `_TEMPLATE/run.py`, verified runner passes)_
- [x] **P0** Ensure **CONTRACTS runner + fixtures** enforce what the docs claim (fail on hard violations; warn only when explicitly allowed). _(Fixed: added actual fixtures for `no-raw-paths` governance check)_
- [x] **P1** Ensure **ADR and context templates** are present and referenced (ADR-*, REJECT-*, STYLE-*). _(Already present: ADR-000-template.md, REJECT-000-template.md, STYLE-000-template.md; updated INDEX.md to reference them)_

### v0.1 — Fixes that make v0.1 true (holes you already hit)
- [x] **P0** **Determinism leak:** make `CORTEX/_generated/cortex.json` deterministic (timestamp should not change outputs unless explicitly supplied as an input OR fixtures must ignore it). _(Fixed: replaced `datetime.utcnow()` with env var `CORTEX_BUILD_TIMESTAMP` or fixed placeholder)_
- [x] **P0** **Exception boundary for "no raw path access":** canon must explicitly carve out that **cortex builders may scan FS**, while **skills/agents must query** via `CORTEX/query.py`. _(Fixed: added INV-008 "Cortex builder exception" to INVARIANTS.md)_
- [x] **P1** **Legacy fallback deprecation story:** `CORTEX/query.py` still checks `BUILD/cortex.json`. Bless as transitional or set a removal version. _(Fixed: removed legacy BUILD/cortex.json fallback entirely — clean, no deprecation needed)_
- [x] **P0** **Artifact escape hatch fixture:** add a repo-wide test that fails if new files appear outside allowed output dirs (`CONTRACTS/_runs`, `CORTEX/_generated`, `MEMORY/LLM_PACKER/_packs`, and user-owned `BUILD`). _(Fixed: created `artifact-escape-hatch` skill with fixtures)_
- [x] **P1** Clarify what `CORTEX/_generated/**` ignore means: keep generated index untracked, but ensure it is always buildable and validated. _(Fixed: added "Generated files" section to CORTEX/README.md)_

### v0.2 — Reliability + enforcement
- [x] **P0** Add/finish a **critic gate** (pre-commit or CI) that checks diffs against canon + fixtures before changes land. _(Fixed: implemented real `critic.py` with 4 checks: CANON/CHANGELOG sync, skill fixtures, raw FS access, skill manifests)_
- [x] **P0** CI workflow: run contract runner + governance checks on PR/push. _(Fixed: enhanced `contracts.yml` to run cortex build, critic, fixtures, and escape hatch)_
- [x] **P1** Add **versioning + deprecation machinery** for token/grammar changes (even if initially "warn-only"). _(Fixed: implemented `lint_tokens.py` for glossary/deprecation checking, `check_canon_governance.py` for version consistency, cortex.build.py now reads version from VERSIONING.md)_

### v0.3 — Memory + pack discipline
- [x] **P0** Define persistent memory store + summarization workflow (tiering, promotion rules, what is mutable vs append-only). _(Fixed: created MEMORY/MEMORY_STORE.md documenting 3-tier memory model, promotion rules, and mutability constraints)_
- [x] **P0** Packer: **manifest integrity** (hashes, deterministic order) and verify-on-load. _(Fixed: added verify_manifest() and load_and_verify_pack() functions to packer.py)_
- [x] **P1** Delta packs: baseline state + diff logic + upgrade story (migration notes). _(Fixed: created MEMORY/DELTA_PACKS.md documenting diff logic, baseline management, and migration workflow)_
- [x] **P1** Migration skill(s) and fixtures for breaking changes. _(Fixed: created canon-migration skill with SKILL.md, run.py, validate.py, and fixtures)_

### v1.0 — Hardening + publishable template
- [x] **P0** Freeze core invariants and “what never changes” (or clearly define the ceremony to change them). _(Fixed: finalized 8 invariants, added `invariant-freeze` skill and check in `check_canon_governance.py`)_
- [x] **P1** Comprehensive docs + examples (how to extend, how to test, how to ship packs). _(Fixed: created `EXTENDING.md`, `TESTING.md`, `SHIPPING.md` guides under `CONTEXT/guides/`)_
- [x] **P1** Security hardening: define trust boundaries, what agents can and cannot touch by default. _(Fixed: enhanced `SECURITY.md` with explicit trust boundaries and approval requirements)_

---

## Phase 1: Packaging and navigation correctness (LLM_PACKER)
These are “drag-and-drop pack must be self-navigable.”

- [x] **P0** Ensure `meta/` is fully emitted into the pack **and** includes **both** machine-readable inventories:
  - `FILE_TREE.txt`
  - `FILE_INDEX.json` _(Fixed: both emitted in `write_pack_file_tree_and_index`)_
- [x] **P0** Ensure SPLIT pack includes **all** of `meta/` (not only the 00..07 docs). _(Fixed: `write_split_pack` now inlines `meta/` files into Section 07)_
- [x] **P1** Ensure `00_INDEX.md` read order references `repo/CORTEX/` and `repo/TOOLS/` (so agents know the maintenance tooling exists). _(Fixed: added reference in `AGS-00_INDEX.md` and `AGS-07_SYSTEM.md`)_
- [x] **P1** Ensure smoke test actually runs and fails when meta inventories are missing or stale. _(Fixed: `llm-packer-smoke` verifies existence of all meta files)_
- [x] **P1** Exclude `research/` directory by default to prevent chat log bloat. _(Done 2025-12-21)_
- [x] **P2** Add a “pack self-check” command: verify manifest, verify meta inventories match repo snapshot. _(Fixed: created `pack-validate` skill)_


---

## Phase 2: Research-driven expansions (add later; do not block Phase 1 unless you choose)
These are the “extra layers” from your merged research and multi-model reviews.

### Core “implement these 7” (high leverage)
- [x] **P0** Shadow cortex / index-first query layer (JSON or SQLite; make it primary). _(Fixed: implemented `CORTEX/query.py` and `cortex.build.py`)_
- [x] **P0** Skill contracts: `/SKILLS/*/SKILL.md`, `run.py`, `validate.py` for every skill. _(Fixed: standardized all skills with manifests, scripts, and fixtures)_
- [x] **P0** Critic loop automation: diff-aware canon + fixtures validator (pre-commit + CI). _(Fixed: implemented `critic.py` and integrated into GitHub Actions)_
- [x] **P0** Pack integrity: manifest + hashes, verify-on-load. _(Fixed: added manifest verification to `packer.py` and `pack-validate` skill)_
- [x] **P1** Explicit versioning: `canon_version` + `grammar_version` and mismatch behavior. _(Fixed: implemented version consistency checks and `lint_tokens.py`)_
- [x] **P1** Context continuity: ADR, rejected paths, style records as first-class. _(Fixed: provided templates and updated `CONTEXT/INDEX.md`)_
- [x] **P2** MCP seam: stage the interface; implement only when you actually need tool access. _(Full implementation 2025-12-21: all 10 tools, dynamic resources, Claude Desktop config)_

### Governance completeness (things that will hurt later if undefined)
- [x] **P1** Canon conflict resolution: what happens when canon contradicts itself; arbitration path. _(Created CANON/ARBITRATION.md 2025-12-21)_
- [x] **P1** Deprecation policy: how rules die safely; minimum windows for breaking changes. _(Created CANON/DEPRECATION.md 2025-12-21)_
- [x] **P1** Migration ceremony: deterministic migration skill + fixtures; formal compatibility break ritual. _(Created CANON/MIGRATION.md 2025-12-21)_
- [x] **P1** Canon bloat prevention: readability constraints; archiving/superseding rules. _(Added INV-009, INV-010 to INVARIANTS.md 2025-12-21)_

### MCP enhancements (extend governance to external AI clients)
- [x] **P2** `critic_run` tool: run TOOLS/critic.py via MCP so Claude can verify governance before acting. _(Implemented 2025-12-21)_
- [x] **P2** `adr_create` tool: create new ADRs with proper template via MCP. _(Implemented 2025-12-21)_
- [x] **P2** `commit_ceremony` tool: return ceremony checklist + staged files for Claude to assist with commits. _(Implemented 2025-12-21)_
- [ ] **P3** MCP audit logging: log all tool calls with timestamps for tracking.
- [x] **P3** MCP governance enforcement: tools refuse to execute if critic fails. _(Done 2025-12-21: Implemented `@governed_tool` in `server.py`)_
- [ ] **P3** Additional prompts: `skill_template`, `conflict_resolution`, `deprecation_workflow`.

### Operational safety and “emergency modes”
- [x] **P2** Emergency procedures as concrete CLI modes (reset, quarantine, isolation, crisis arbitration). _(Created CANON/CRISIS.md, TOOLS/emergency.py 2025-12-21)_
- [x] **P2** “Stewardship” structure: explicit human escalation path for canon-level failures. _(Created CANON/STEWARDSHIP.md 2025-12-21)_
- [x] **P2** “Constitutional license” concept (optional legal-protective layer). _(Done 2025-12-21: CANON/AGREEMENT.md, ADR-007)_

### Performance and scaling
- [x] **P1** Make “O(n×m) scanning” impossible by design (index-first non-optional). _(Fixed: implemented SQLite Cortex with O(1) lookups)_
- [x] **P2** Incremental indexing + freshness/TTL rules (avoid stale cortex). _(Done 2025-12-21: Incremental mtime check, prune deletions, unique IDs)_

### Token economics
- [x] **P2** **Codebook & Compression**: Expanded codebook to 37 entries, bidirectional symbolic compression via `TOOLS/compress.py`, and stable IDs (`@C1`, `@M7`). _(Done 2025-12-21)_
- [x] **P2** Tokenizer test harness: measure real tokenization against target models
    - `TOOLS/tokenizer_harness.py`: Real token measurement for GPT-4 (cl100k) and GPT-4o/o1 (o200k).
    - `MEMORY/LLM_PACKER/Engine/packer.py`: Integrated `tiktoken` for 100% accurate token counts in `CONTEXT.txt`.

### Chat-derived mechanics deltas (small, buildable, low-bloat)
- [ ] **P2** Compress the LLM packer if possible.
- [x] **P1** Research activity cache: persist summaries for URLs to avoid redundant browsing (`TOOLS/research_cache.py`). _(Done 2025-12-21)_

---

## Research audit

- [x] **P2** Research completeness audit _(Completed 2025-12-21: All 9 research docs reviewed, items extracted below)_
  - Read every document under `CONTEXT/research/` (or your research source of truth)
  - Extract what should become: CANON rules, CONTRACT checks, SKILLS, or MAPS/entrypoints
  - Leave the rest as non-binding reference (do not promote by default)

---

## Research-derived tasks (extracted 2025-12-21)

These items were identified from the research folder audit. They are not duplicates of the above.

### Bootstrap and Context Tools
- [x] **P1** Genesis Prompt (`CANON/GENESIS.md`): A bootstrap prompt that solves the chicken-and-egg problem. Ensures agents load CANON first before any other instruction. _(Created 2025-12-21)_
- [x] **P1** Context Query Tool (`CONTEXT/query-context.py`): CLI to search decisions by tag, status, review date. Enables agents to query "why did we decide X?" without full file scans. _(Created 2025-12-21)_
- [x] **P2** Context Review Tool (`CONTEXT/review-context.py`): Flags overdue ADR reviews. Keeps decision records from going stale. _(Created 2025-12-21)_

### Data Integrity and Validation
- [x] **Provenance Headers**: Utility + `meta/PROVENANCE.json` for LLM packs (2025-12-21)
- [x] **Precise Tokenization**: Integrated `tiktoken` into `packer.py` for accurate context reports (2025-12-21)
- [x] **P2** Schema Validation for "Law-Like" Files: Implement JSON Schema validation for canon, skills, context records, and cortex index. Contracts enforce schema validity. _(Done 2025-12-21)_
