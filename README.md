# Agent Governance System (AGS)

The Agent Governance System (AGS) is a constitutional framework for durable, multi-agent intelligence.

This repository provides a language-driven operating system for AI-native projects. It treats text as law, code as consequence and fixtures as precedent. By enforcing a clear authority gradient and mechanical gates, AGS prevents drift, preserves intent and allows continuous evolution across model swaps and refactors.

## Starting a Session (Mandatory)

Every new agent session **must** begin with the Genesis Prompt to bootstrap the governance system correctly.

1. Read the full prompt from `CANON/GENESIS.md`.
2. Prepend it as the system message, pack header, or first instruction.
3. Agents are instructed to remind you if it is missing.

This solves the bootstrap paradox: the agent knows the governance structure exists *before* reading any other file.

## Project layout

The repository is organised into eight layers. Files are loaded in the following order:

1. **CANON** - the constitution and invariants of the system.
2. **CONTEXT** - decisions, rejections and preferences that record why things are the way they are.
    - `decisions/` - ADRs (Architecture Decision Records)
    - `preferences/` - Governance preferences (style, ceremony)
    - `research/` - Shared research cache (e.g., `research_cache.db` for SQLite)
3. **MAPS** - entrypoints and data flow descriptions that tell agents where to make changes.
4. **SKILLS** - modular capabilities packaged with contracts and fixtures.
5. **CONTRACTS** - fixtures and schemas used to mechanically enforce rules.
6. **MEMORY** - packers, manifests and combined state for context handoff.
7. **CORTEX** - a shadow index (e.g. SQLite) that agents query instead of the raw filesystem.
    - **Semantic Core** - vector embeddings for efficient semantic search and token compression (Phase 1 complete)
    - Enables 96% token savings by compressing context with @Symbols and vectors
    - See [SEMANTIC_CORE_QUICK_START.md](./SEMANTIC_CORE_QUICK_START.md) for usage
8. **TOOLS** - helper scripts such as critics and linters.

## Semantic Core (Token Compression)

The CORTEX layer includes a **Semantic Core** system that uses vector embeddings to enable massive token savings when dispatching tasks to smaller language models.

**How it works:**
- Big models (Opus) maintain semantic understanding via 384-dimensional vector embeddings in CORTEX
- Small models (Haiku) receive compressed task specifications with @Symbols and vector context
- Achieves **96% token reduction** per task (50,000 → 2,000 tokens)

**Quick start:**
```bash
# See it in action
python demo_semantic_dispatch.py

# Test all systems
python CORTEX/test_semantic_core.py

# Index your code
python CORTEX/vector_indexer.py --index
```

**Documentation:**
- [Quick Start Guide](./SEMANTIC_CORE_QUICK_START.md) - 5-minute introduction
- [Complete Index](./SEMANTIC_CORE_INDEX.md) - Full navigation and API reference
- [ADR-030](./CONTEXT/decisions/ADR-030-semantic-core-architecture.md) - Architecture specification
- [Roadmap](./CONTEXT/decisions/ROADMAP-semantic-core.md) - 4-phase implementation plan

**Status:** Phase 1 (Vector Foundation) complete and production-ready. See [Final Report](./CONTRACTS/_runs/semantic-core-phase1-final-report.md) for details.

## MCP integration (external clients)

AGS exposes an MCP server for IDEs and desktop clients. Use the entrypoint wrapper
`CONTRACTS/_runs/ags_mcp_entrypoint.py` to keep audit logs under allowed output
roots (`CONTRACTS/_runs/mcp_logs/`). Verify with the `mcp-smoke` or
`mcp-extension-verify` skills. See `MCP/README.md` for client config examples.

## How to use

This repository is a template: most files are placeholders that illustrate the intended structure. To adapt the system for your own project, fill in the canon, add decisions and ADRs, implement skills and write fixtures.

Agents interacting with the system should follow the protocol described in `CANON/CONTRACT.md`. In brief:

1. Load the canon first and respect its authority.
2. Consult context records before making changes.
3. Use the maps to find the right entrypoints.
4. Execute work through skills rather than ad-hoc scripts.
5. Validate changes using the runner in `CONTRACTS`.
6. Update the canon and changelog in the same commit when rules change.
7. Default to repo-only access; request explicit permission before accessing paths outside the repo (see `CONTEXT/decisions/ADR-012-privacy-boundary.md`).

## How to extend AGS

- Add a skill: create `SKILLS/<skill-name>/` with `SKILL.md`, a run script, a validation script, and `fixtures/<case>/input.json` plus `expected.json`.
- Add fixtures: place skill fixtures under `SKILLS/<skill-name>/fixtures/` and governance fixtures under `CONTRACTS/fixtures/`.
- Add ADRs: create a new `CONTEXT/decisions/ADR-xxx-*.md` and reference it in `CONTEXT/INDEX.md`.
- Use `BUILD/` for your project's build outputs (dist). It is disposable and should not contain authored content. The template writes its own artifacts under `CONTRACTS/_runs/`, `CORTEX/_generated/`, and `MEMORY/LLM_PACKER/_packs/`.
- Planning snapshots live under `CONTEXT/archive/planning/` (see `CONTEXT/archive/planning/INDEX.md`).

For more details, see individual files in the respective directories.
