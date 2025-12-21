# Changelog

All notable changes to the Agent Governance System will be documented in this file.  The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versioning follows the rules defined in `CANON/VERSIONING.md`.

## [Unreleased]

### Added
- None.

### Changed
- None.

### Fixed
- None.

### Removed
- None.

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

### Fixed
- None.

### Removed
- None.

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

### Fixed
- Nothing.

### Removed
- Nothing.

## [1.2.0] - 2025-12-21

### Added
- MCP full implementation: all 10 tools working (including `research_cache`), dynamic resources, Claude Desktop config ready.
- MCP governance tools: `critic_run`, `adr_create`, `commit_ceremony` for Claude-assisted governance.
- MCP seam: `MCP/MCP_SPEC.md`, `MCP/schemas/`, `MCP/server.py` for Model Context Protocol integration.
- **Emergency Governance**:
    - `CANON/CRISIS.md`: Procedures with 5 crisis levels and CLI modes.
    - `CANON/STEWARDSHIP.md`: Human escalation paths and steward authority.
    - `TOOLS/emergency.py`: CLI for crisis handling (validate, rollback, quarantine, etc.).

### Fixed
- Nothing.

### Removed
- Nothing.

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

### Fixed
- Nothing.

### Removed
- Nothing.

## [1.1.0] - 2025-12-21

### Added
- STYLE-002: Engineering Integrity preference (foundational fixes over patches).
- STYLE-003: Mandatory Changelog Synchronisation preference.
- Official Blue Launcher with icon for LLM_PACKER.

### Changed
- LLM_PACKER refactoring: Moved core logic to `Engine/` subfolder.
- Renamed `LLM-PACKER` to `LLM_PACKER` to resolve Python import limitations.
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
- Memory packer produces LLM packs under `MEMORY/_packs/` and records a `BUILD/` file tree inventory.
- LLM packer tooling relocated under `MEMORY/LLM-PACKER/` (PowerShell wrapper retained).

### Removed

- Nothing.

## [0.1.3] - 2025-12-20

### Changed

- LLM pack output root moved to `MEMORY/LLM-PACKER/_packs/` (from `MEMORY/_packs/`).

## [0.1.4] - 2025-12-20

### Removed

- Legacy `MEMORY/_packs/` directory (no longer used).

## [0.1.1] - 2025-12-19

### Added

- Root `AGENTS.md` and research scaffold under `CONTEXT/research/`.
- Reference `example-echo` skill with a basic fixture.
- `BUILD/` output root with gitignore rules and a keep file.

### Changed

- Canon rules to require `BUILD/` as the output root.
- Contract runner to execute skill fixtures and write outputs under `BUILD/`.
- Cortex build to emit its index under `BUILD/` and skip indexing `BUILD/`.

### Removed

- Nothing.

## [0.1.0] - 2025-12-19

### Added

- Initial repository skeleton with canon, context, maps, skills, contracts, memory, cortex and tools directories.
- Templates for ADRs, rejections, preferences and open issues.
- Basic runner script and placeholder fixtures.
- Versioning policy and invariants.

### Changed

- Nothing. This is the first release.

### Removed

- Nothing.
