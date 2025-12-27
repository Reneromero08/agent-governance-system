# Changelog

All notable changes to the Agent Governance System will be documented in this file.  The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versioning follows the rules defined in `CANON/VERSIONING.md`.

## [2.10.0] - 2025-12-27

### LLM Packer Refactor (created 2025-12-26; modified 2025-12-27)

#### Added
- Modular Packer Architecture: Refactored monolithic script into `MEMORY/LLM_PACKER/Engine/packer/` package with dedicated `core`, `split`, `lite`, and `archive` components.
- New Launchers: `1-AGS-PACK.cmd`, `2-CAT-PACK.cmd`, `3-LAB-PACK.cmd` for scoped packing.
- `lab` scope support for `CATALYTIC-DPT/LAB` research packs.
- `MEMORY/LLM_PACKER/CHANGELOG.md` as the single source of truth for packer history.
- Migration tooling: `packer_legacy_backup.py`, `migrate_phase1.py`, `verify_phase1.py`, `refactor_packer.py`, `scan_old_refs.py`.
- `MEMORY/LLM_PACKER/Engine/run_tests.cmd` for smoke test execution.

#### Changed
- Consolidated packer documentation into `MEMORY/LLM_PACKER/README.md`.
- Strict output structure enforcement: `FULL/`, `SPLIT/`, `LITE/`, `archive/`.
- `pack.zip` now exclusively contains `meta/` and `repo/`.
- Updated smoke tests (`llm-packer-smoke`) and `pack-validate` skill to align with new structure.
- Updated `llm-packer-smoke` to run the modular packer CLI (`python -m MEMORY.LLM_PACKER.Engine.packer`) and validate `FULL/` + `SPLIT/` (+ `LITE/` when enabled), replacing legacy `Engine/packer.py` + `COMBINED/` expectations.
- Removed legacy packer launchers/shortcuts and deprecated scripts (superseded by the modular packer + numbered launchers).
- Updated `SKILLS/llm-packer-smoke/run.py` to support `allow_duplicate_hashes` flag.
- Updated all smoke test fixtures to enable `allow_duplicate_hashes: true`.
- Updated `CONTEXT/decisions/ADR-002-llm-packs-under-llm-packer.md`.
- Updated `CONTEXT/decisions/ADR-013-llm-packer-lite-split-lite.md`.
- Updated `CONTEXT/guides/SHIPPING.md` with new launcher references.
- Moved historic/legacy packer changelog entries to `MEMORY/LLM_PACKER/CHANGELOG.md`.
- Updated `TOOLS/critic.py` to allow `llm-packer-smoke` skill to use raw filesystem access.

## [2.9.0] - 2025-12-26

### MCP Startup Skill (created 2025-12-26)

#### Added
- `CATALYTIC-DPT/SKILLS/mcp-startup/` skill for MCP server startup automation.
- Comprehensive documentation: `SKILL.md`, `README.md`, `INSTALLATION.md`, `USAGE.md`, `INDEX.md`, `CHECKLIST.md`, `MODEL-SETUP.md`, `QUICKREF.txt`.
- Startup scripts: `startup.ps1` and `startup.py`.



## [2.8.6] - 2025-12-26

### Governance & CI (created 2025-12-19; modified 2025-12-26)

#### Added
- Canon governance check system (comprehensive integration):
  - `TOOLS/check-canon-governance.js`: Core governance check script (Node.js)
  - `SKILLS/canon-governance-check/`: Full skill wrapper with Cortex provenance integration
  - `SKILLS/canon-governance-check/run.py`: Python wrapper that logs governance results to Cortex
  - `SKILLS/canon-governance-check/scripts/pre-commit`: Git pre-commit hook for local enforcement
  - CI integration in `.github/workflows/contracts.yml`: Runs on every push/PR
  - Cortex provenance tracking: Logs governance check events to `CONTRACTS/_runs/<run_id>/events.jsonl`

#### Changed
- CI workflows consolidated: merged governance workflow into `.github/workflows/contracts.yml` (single source of CI truth).
- Installed canon governance pre-commit hook locally into `.git/hooks/pre-commit` (from `SKILLS/canon-governance-check/scripts/pre-commit`).
- Bumped `canon_version` to 2.8.6.

## [2.8.5] - 2025-12-26

### CAT-DPT (created 2025-12-24; modified 2025-12-26)

#### Changed
- CAT-DPT LAB reorganization: Moved architecture docs to `CATALYTIC-DPT/LAB/ARCHITECTURE/`, research docs consolidated in `CATALYTIC-DPT/LAB/RESEARCH/`, added index README.
- CAT-DPT LAB compression: Merged architecture docs into `CATALYTIC-DPT/LAB/ARCHITECTURE/SWARM_ARCHITECTURE.md`, semiotic docs into `CATALYTIC-DPT/LAB/RESEARCH/SEMIOTIC_COMPRESSION.md` with Cortex-style hash refs.
- (Catalytic Computing entries moved to `CATALYTIC-DPT/CHANGELOG.md`)

### Cortex & Provenance (created 2025-12-19; modified 2025-12-26)

#### Changed
- Cortex/Provenance hardening: Fixed build crashes caused by volatile pytest temp files in `CORTEX/cortex.build.py` and `TOOLS/provenance.py`.

## [2.8.4] - 2025-12-23

### Cross-Platform Fixes (created 2025-12-19; modified 2025-12-23)

#### Fixed
- MCP server test mode: replaced Unicode checkmark characters (`✓`) with ASCII `[OK]` to fix Windows `cp1252` encoding errors.
- `TOOLS/lint_tokens.py`: replaced Unicode warning/check marks with ASCII `[WARN]` and `[OK]` for cross-platform compatibility.
- `TOOLS/critic.py`: detects hardcoded artifact paths outside allowed roots (CONTRACT Rule 6).
- `TOOLS/codebook_build.py --check` now properly detects drift by comparing markdown entries (ignoring timestamps).
- Added `validate.py` to all skills (doc-update, master-override, mcp-extension-verify, mcp-smoke) for uniform validation.
- Updated `README.md` to reflect 8 repository layers (not 6): CANON, CONTEXT, MAPS, SKILLS, CONTRACTS, MEMORY, CORTEX, TOOLS.

## [2.8.3] - 2025-12-23

### Catalytic Computing (created 2025-12-23; modified 2025-12-23)

#### Added
- `CONTEXT/decisions/ADR-018-catalytic-computing-canonical-note.md` documenting the canonical note.

#### Changed
- `CANON/CATALYTIC_COMPUTING.md` updated with the catalytic computing canonical note.

## [2.8.1] - 2025-12-23

### Cortex & Navigation (created 2025-12-23; modified 2025-12-23)

#### Added
- `CORTEX/_generated/SECTION_INDEX.json` (generated) for section-level navigation and citation hashes.
- `CORTEX/_generated/SUMMARY_INDEX.json` and `CORTEX/_generated/summaries/` (generated) for deterministic, advisory section summaries.
- `CORTEX/SCHEMA.md` documenting the Cortex data model (SQLite and JSON schemas, entity types, determinism, versioning).
- `TOOLS/cortex.py` commands: `read`, `resolve`, `search`, `summary`.
- `SKILLS/cortex-summaries/` fixture skill for deterministic summary generation validation.
- `CONTRACTS/_runs/<run_id>/events.jsonl` (generated) for Cortex provenance events when `CORTEX_RUN_ID` is set.
- `CONTRACTS/_runs/<run_id>/run_meta.json` (generated) anchoring provenance runs to a specific `CORTEX/_generated/SECTION_INDEX.json` hash.

## [2.8.0] - 2025-12-23

### Privacy, Context, and Governance (created 2025-12-23; modified 2025-12-23)

#### Added
- `CONTEXT/decisions/ADR-012-privacy-boundary.md` defining the privacy boundary (no out-of-repo access without explicit user approval).
- `CONTEXT/decisions/ADR-015-logging-output-roots.md` defining logging output root policy and enforcement.
- `CONTEXT/decisions/ADR-016-context-edit-authority.md` clarifying when agents may edit existing CONTEXT records.
- `CONTEXT/decisions/ADR-017-skill-formalization.md` formalizing the skill contract (SKILL.md, run.py, validate.py, fixtures).
- Governance fixtures for privacy boundary, log output roots, context edit authority, and output-root enforcement.

#### Changed
- Aligned all logging with INV-006 output roots: logs now written under `CONTRACTS/_runs/<purpose>_logs/` (ADR-015).
- Updated canon docs (`CANON/CONTRACT.md`, `CANON/CRISIS.md`, `CANON/STEWARDSHIP.md`, `AGENTS.md`) to reflect correct log locations and the skill contract.
- Clarified `CANON/CONTRACT.md` Rule 3 to require both explicit user instruction AND explicit task intent for CONTEXT edits (ADR-016).
- Enhanced `CANON/CONTRACT.md` Rule 2 to explicitly require ADRs for governance decisions and recommend them for significant code changes.
- Enhanced `AGENTS.md` to explicitly document the skill contract (SKILL.md, run.py, validate.py, fixtures) as defined in ADR-017.
- Bumped `canon_version` to 2.8.0 (minor: catalytic computing canonical note, governance clarifications).

## [2.6.0] - 2025-12-23

### Added
- `CONTEXT/decisions/ADR-011-master-override.md` defining the `MASTER_OVERRIDE` interface.
- `master-override` skill for override audit logging and gated log access.
- `mcp-smoke` and `mcp-extension-verify` skills for MCP verification.
- `doc-update` skill to standardize documentation updates.
- `CONTEXT/archive/planning/` planning archive with an index and dated snapshots.
- `REPO_FIXES_TASKS.md` checklist for contract-alignment follow-ups.

### Changed
- Added `MASTER_OVERRIDE` to governance docs (Agreement, Contract, Genesis, Agents, Glossary).
- MCP documentation now recommends logs under `CONTRACTS/_runs/mcp_logs/`.
- Planning references now point at `CONTEXT/archive/planning/INDEX.md`.
- Bumped `canon_version` to 2.6.0.

### Fixed
- Python 3.8 compatibility for governance tooling and contract runner.

### Removed
- Root planning docs: `ROADMAP.md`, `AGS_MASTER_TODO.md`.

## [2.5.5] - 2025-12-21

### Added
- `CONTEXT/decisions/ADR-010-authorized-deletions.md`.
- `CONTRACTS/fixtures/governance/deletion-authorization` fixture.

### Changed
- Deletions now require explicit instruction and confirmation (CANON rules still archived per INV-010).
- Bumped `canon_version` to 2.5.5.
- Regenerated `CANON/CODEBOOK.md`.

## [2.5.4] - 2025-12-21

### Changed
- Commit ceremony now accepts short confirmations like "go on" after checks and staged files are listed.
- Updated `CONTRACTS/fixtures/governance/commit-ceremony` to document confirmations.
- Bumped `canon_version` to 2.5.4.
- Regenerated `CANON/CODEBOOK.md`.

## [2.5.3] - 2025-12-21

### Added
- `CONTEXT/decisions/ADR-008-composite-commit-approval.md`.
- `CONTRACTS/fixtures/governance/commit-ceremony` fixture.

### Changed
- Commit ceremony now recognizes explicit "commit, push, and release" directives.
- Updated `CONTRACTS/fixtures/governance/canon-sync` to include `CANON/AGREEMENT.md`.
- Bumped `canon_version` to 2.5.3.
- Regenerated `CANON/CODEBOOK.md`.

## [2.5.2] - 2025-12-21

### Added
- `requirements.txt` with `jsonschema` to satisfy schema validation dependencies in CI.

### Fixed
- CI critic failure when `jsonschema` was missing.

## [2.5.1] - 2025-12-21

### Added
- `repo-contract-alignment` skill with fixtures for contract alignment workflow.
- `TOOLS/skill_runtime.py` to enforce skill canon-version compatibility at runtime.

### Changed
- Regenerated `CANON/CODEBOOK.md` to include the new skill.
- Bumped `canon_version` to 2.5.1.
- Updated `AGENTS.md` authority gradient to include `CANON/AGREEMENT.md`.
- Updated cortex docs to reference the SQLite index (`cortex.db`).
- Updated skill `required_canon_version` ranges to `>=2.5.1 <3.0.0`.
- Skills now validate `required_canon_version` before running.
- `TOOLS/critic.py` now refuses to run while `.quarantine` exists.

### Fixed
- `TOOLS/critic.py` output uses ASCII to avoid Windows encoding errors.

## [2.5.0] - 2025-12-21

### Added
- **Audit Logging**: All MCP tool executions are now logged to `MCP/logs/audit.jsonl` with timestamp, tool name, status, and duration.
- **Improved Prompts**:
    - `skill_template`: Injects `SKILLS/_TEMPLATE` content.
    - `conflict_resolution`: Injects `CANON/ARBITRATION.md`.
    - `deprecation_workflow`: Injects `CANON/DEPRECATION.md`.

### Added
- **Security**: Implemented Governance Enforcement in `MCP/server.py`.
- **Logic**: Tools (`skill_run`, `adr_create`, etc.) are now decorated with `@governed_tool`.
- **Enforcement**: If `TOOLS/critic.py` reports any violations, dangerous actions are BLOCKED with a "Governance Lockdown" error.

### Changed
- **Performance**: Promoted Cortex Indexing from O(N) rebuild to Incremental (checking `mtime`).
- `CORTEX/cortex.build.py`: Refactored to retain DB, migrate schema, and prune deleted entries.
- `CORTEX/schema.sql`: Added `last_modified` column to `entities` table.
- **Strictness**: Changed ID generation to `page:{rel_path}` (unique) to resolve filename collision bugs.

## [2.2.0] - 2025-12-21

### Added
- **Constitutional License**: `CANON/AGREEMENT.md` defines the liability separation between Human (Sovereign) and Agent (Instrument).
- `CONTEXT/decisions/ADR-007-constitutional-agreement.md`: Formal decision record for the agreement.
- `CANON/INDEX.md`: Master index of the law, listing `AGREEMENT.md` as the highest authority.

### Changed
- **Authority Gradient**: Updated `CANON/CONTRACT.md` to place `AGREEMENT.md` at rank #1, shifting the Contract to rank #2.

## [2.1.0] - 2025-12-21

### Added
- `CONTEXT/decisions/ADR-004-mcp-integration.md`: Retroactive decision record for the Model Context Protocol (MCP) implementation.
- `CONTEXT/decisions/ADR-005-persistent-research-cache.md`: Retroactive decision record for the SQLite-backed Research Cache.
- `CONTEXT/decisions/ADR-006-governance-schemas.md`: Documented the "Governance Object Schemas" decision to legitimize INV-011.
- **Governance Schemas**: Defined JSON Schemas for `ADR` (Architecture Decision Records), `SKILL` (Skill Manifests), and `STYLE` (Preferences) in `MCP/schemas/governance/`.
- `TOOLS/schema_validator.py`: Utility to parse Markdown headers and validate against JSON Schemas.
- **INV-011**: New invariant requiring schema compliance for law-like files.
- `critic.py` now enforces schema validation on all ADRs, Skills, and Preferences.

### Changed
- Refactored `SKILLS/_TEMPLATE` and `canon-migration` to use compliant Status (`Draft`, `Active`).

## [2.0.0] - 2025-12-21

### Added
- `CANON/ARBITRATION.md`: Conflict resolution policy with escalation protocol.
- **Symbolic Compression**:
    - `CANON/CODEBOOK.md`: Stable ID registry for token-efficient referencing (@C0, @I3, @S7).
    - `TOOLS/codebook_build.py`: Generator for codebook from repo entities.
    - `TOOLS/codebook_lookup.py`: CLI/library for programmatic lookups.
    - `TOOLS/compress.py`: Bidirectional symbolic compression/expansion tool.
    - `TOOLS/tokenizer_harness.py`: Real token measurement for GPT-4 (cl100k) and GPT-4o/o1 (o200k).
    - `CANON/GENESIS_COMPACT.md`: Token-efficient bootstrap prompt using symbols.
- **Provenance Headers**: Added `TOOLS/provenance.py` and integrated into all major generators (`codebook_build.py`, `cortex.build.py`, `packer.py`) for automated audit trails. Introduced `meta/PROVENANCE.json` in memory packs as a single-point-of-truth manifest for pack integrity.

## [1.2.0] - 2025-12-21

### Added
- MCP full implementation: all 10 tools working (including `research_cache`), dynamic resources, Claude Desktop config ready.
- MCP governance tools: `critic_run`, `adr_create`, `commit_ceremony` for Claude-assisted governance.
- MCP seam: `MCP/MCP_SPEC.md`, `MCP/schemas/`, `MCP/server.py` for Model Context Protocol integration.
- **Emergency Governance**:
    - `CANON/CRISIS.md`: Procedures with 5 crisis levels and CLI modes.
    - `CANON/STEWARDSHIP.md`: Human escalation paths and steward authority.
    - `TOOLS/emergency.py`: CLI for crisis handling (validate, rollback, quarantine, etc.).

## [1.1.1] - 2025-12-21

### Added
- Genesis Prompt (`CANON/GENESIS.md`): Bootstrap prompt that ensures agents load CANON first.
- Research folder now tracked in git with clarifying README.
- Context Query Tool (`CONTEXT/query-context.py`): CLI to search decisions by tag, status, review date.
- Context Review Tool (`CONTEXT/review-context.py`): Flags overdue ADR reviews.
- CONTRACT Rule 7: Commit ceremony as a non-negotiable law.
- AGENTS.md Section 10: Full commit ceremony specification with anti-chaining rule.
- `CANON/DEPRECATION.md`: Deprecation windows and ceremonies for safe rule retirement.
- **Research Cache**: Implemented persistent SQLite-backed cache for research summaries (`TOOLS/research_cache.py`) to avoid redundant browsing.
- `CANON/MIGRATION.md`: Formal compatibility break ritual with phases and rollback.
- `CANON/INVARIANTS.md` INV-009 and INV-010: Canon bloat prevention (readability limits, archiving).

### Changed
- Shadow Cortex now uses SQLite (`cortex.db`) instead of flat JSON for O(1) lookups.
- `query.py` updated with `--json` export flag for backward-compatible JSON output.
- Expanded ROADMAP with v1.2 milestone and research-derived tasks.
- Strengthened STYLE-001 with prohibited interpretations list and anti-chaining rule.

### Deprecated
- `cortex.json` emission from build process (replaced by SQLite `cortex.db`).

## [1.1.0] - 2025-12-21

### Added
- STYLE-002: Engineering Integrity preference (foundational fixes over patches).
- STYLE-003: Mandatory Changelog Synchronisation preference.
- Hardened STYLE-001 (Blanket Approval Ban and Mandatory Ceremony Phase).

## [1.0.0] - 2025-12-21

### Added

- Invariant freeze fixture to enforce INV-001 through INV-008.
- `check_invariant_freeze()` in `check_canon_governance.py`.
- Documentation guides: `EXTENDING.md`, `TESTING.md`, `SHIPPING.md`.
- Trust boundaries in `SECURITY.md` (read/write access, human approval requirements).

### Changed

- Invariants INV-001 through INV-008 are now frozen (v1.0 stability).
- SECURITY.md expanded with trust boundary definitions.

## [0.1.5] - 2025-12-21

### Added

- New invariants: INV-005 (Determinism), INV-006 (Output roots), INV-007 (Change ceremony), INV-008 (Cortex builder exception).
- New glossary terms: ADR, Pack, Manifest, Critic, Authority gradient, Invariant, Change ceremony, Entity, Query.
- Governance fixtures for `canon-sync`, `token-grammar`, and `no-raw-paths`.
- Deterministic timestamp in `cortex.build.py` via `CORTEX_BUILD_TIMESTAMP` env var.
- Python `run.py` for `_TEMPLATE` skill.

### Changed

- Authority gradient in `CONTRACT.md` aligned with `AGENTS.md` (now 8-level hierarchy).
- INVARIANTS.md expanded from 4 to 8 invariants.
- GLOSSARY.md expanded from 11 to 20 terms.

## [0.1.2] - 2025-12-20

### Added

- Subsystem-owned artifact roots with keep files: `CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/_packs/`.

### Changed

- `BUILD/` is reserved for user build outputs, not system artifacts.
- Contract runner writes fixture outputs under `CONTRACTS/_runs/`.
- Cortex build writes index under `CORTEX/_generated/cortex.json` (query keeps fallback support).

## [0.1.1] - 2025-12-19

### Added

- Root `AGENTS.md` and research scaffold under `CONTEXT/research/`.
- Reference `example-echo` skill with a basic fixture.
- `BUILD/` output root with gitignore rules and a keep file.

### Changed

- Canon rules to require `BUILD/` as the output root.
- Contract runner to execute skill fixtures and write outputs under `BUILD/`.
- Cortex build to emit its index under `BUILD/` and skip indexing `BUILD/`.

## [0.1.0] - 2025-12-19

### Added

- Initial repository skeleton with canon, context, maps, skills, contracts, memory, cortex and tools directories.
- Templates for ADRs, rejections, preferences and open issues.
- Basic runner script and placeholder fixtures.
- Versioning policy and invariants.
