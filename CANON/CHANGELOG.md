# Changelog

All notable changes to the Agent Governance System will be documented in this file.  The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the versioning follows the rules defined in `CANON/VERSIONING.md`.

## [0.1.2] - 2025-12-20

### Added

- Subsystem-owned artifact roots with keep files: `CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/_packs/`.

### Changed

- `BUILD/` is reserved for user build outputs, not system artifacts.
- Contract runner writes fixture outputs under `CONTRACTS/_runs/`.
- Cortex build writes index under `CORTEX/_generated/cortex.json` (query keeps fallback support).
- Memory packer produces LLM packs under `MEMORY/_packs/` and records a `BUILD/` file tree inventory.
- LLM packer tooling relocated under `MEMORY/LLM-PACKER-1.0/` (PowerShell wrapper retained).

### Removed

- Nothing.

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
