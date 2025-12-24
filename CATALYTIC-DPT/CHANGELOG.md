# CATALYTIC-DPT Changelog

All notable changes to the Catalytic Computing Department (Isolated R&D) will be documented in this file.

## [1.5.0] - 2025-12-24

### Added
- Official `mcp-builder` skill from `anthropics/skills`.
- Official `skill-creator` skill from `anthropics/skills`.

### Changed
- **Skills Standardization**:
  - All `SKILL.md` files updated to explicitly link to their bundled resources (scripts, assets, and references).
  - `governor` skill now correctly references `GOVERNOR_SOP.json`, `HANDOFF_TO_GOVERNOR.md`, and `GOVERNOR_CONTEXT.md`.
  - `launch-terminal` skill now references `ANTIGRAVITY_BRIDGE.md`.
  - `ant-worker` skill now references `schema.json` and task fixtures.
  - `swarm-orchestrator` skill now references launcher scripts.
  - `file-analyzer` skill now references analysis scripts.
- **Path Correction**: Fixed `HANDOFF_TO_GOVERNOR.md` pointing to non-existent `GOVERNOR_SOP.json` in the root (now correctly points to `assets/`).

### Removed
- `LICENSE.txt` from `skill-creator` and `mcp-builder` (non-functional token bloat).

## [1.4.0] - 2025-12-24

### Changed
- **Skills Reorganization** (agentskills.io spec compliance):
  - All skills now have proper YAML frontmatter in `SKILL.md` (name, description, compatibility).
  - Restructured folder layout: `scripts/`, `assets/`, `references/` per spec.
  - Moved executables to `scripts/` subdirectory:
    - `ant-worker/scripts/run.py`
    - `governor/scripts/run.py`
    - `file-analyzer/scripts/run.py`
    - `launch-terminal/scripts/run.py`
    - `swarm-orchestrator/scripts/poll_and_execute.py`, `poll_tasks.py`, `agent_loop.py`, `launch_swarm.ps1`
  - Moved test fixtures to `assets/` (renamed from `fixtures/`).
  - Moved reference docs to `references/` (GEMINI.md → governor/references/, etc.).
  - Removed redundant skill-level README (use SKILL.md instead).

### Fixed
- MCP stdio_server: Added `skill_run` tool for proper skill execution via MCP.
- Fixed import paths in orchestrator scripts after reorganization.

## [1.3.0] - 2025-12-24

### Added
- `SKILLS/governor/fixtures/phase0_directive.json`: Phase 0 task as structured JSON directive.

### Changed
- Merged `GOVERNOR_CONTEXT.md` into `SKILLS/governor/SKILL.md` (role + MCP commands now in skill).
- Converted `HANDOFF_TO_GOVERNOR.md` to skill fixture format.
- Trimmed `ORCHESTRATION_ARCHITECTURE.md` from 16KB to 7KB (removed verbose examples).
- Updated `README.md` to reflect actual SKILLS folder structure.

### Removed
- `HANDOFF_TO_GOVERNOR.md` (now `SKILLS/governor/fixtures/phase0_directive.json`)
- `GOVERNOR_CONTEXT.md` (merged into `SKILLS/governor/SKILL.md`)
- `PHASE0_IMPLEMENTATION_GUIDE.md` (redundant with `SCHEMAS/README.md`)
- `pop_swarm_terminals.py` (redundant with `launch_swarm.ps1`)

---

## [1.2.0] - 2025-12-24

### Added
- `swarm_config.json`: Centralized configuration for model/role assignments (the "Symbolic Link" for agents).
- `CHANGELOG.md`: This file, to track DPT-specific evolution.

### Changed
- **Hierarchy Generalization**: Refactored the entire department documentation to be model-agnostic.
  - Established hierarchy: **God (User) → President (Orchestrator) → Governor (Manager) → Ants (Executors)**.
  - Replaced hardcoded references to "Claude", "Gemini", and "Grok" with their respective functional roles.
  - Documentation now references `swarm_config.json` for current implementations.
- **File Renaming**:
  - `CODEX_SOP.json` → `GOVERNOR_SOP.json` (Reflecting the new role-based naming).
  - `HANDOFF_TO_CODEX.md` → `HANDOFF_TO_GOVERNOR.md`.
- **Core Documentation**:
  - `ORCHESTRATION_ARCHITECTURE.md`: Updated diagram and roles to reflect the new hierarchy.
  - `GOVERNOR_SOP.json`: Fully generalized instructions for any CLI Manager.
  - `README.md`: Updated directory tree and success criteria.
  - `ROADMAP.md`: Removed model-specific parameter (e.g., "200M") constraints.
  - `TESTBENCH.md`: Updated instructions for the Governor.
  - `GOVERNOR_CONTEXT.md`: Transitioned from "Claude" to "President".

### Cleaned
- Removed legacy "Codex" and "Codex-governed" terminology.
- Deleted redundant index/temporary files from the root of `CATALYTIC-DPT`.

---

## [1.1.0] - 2025-12-24
- Initial setup of Catalytic-DPT directory.
- Defined Phase 0 contracts (JSON Schemas).
- Established first iteration of Multi-Agent Orchestration.
