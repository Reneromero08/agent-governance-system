<!-- CONTENT_HASH: 82169ad2a1976e7041c636aeb9753816c946acd35754ceb3e84bb6d58f647a3f -->

# Agent Governance System (AGS)

The Agent Governance System (AGS) is a constitutional framework for durable, multi-agent intelligence.

This repository provides a language-driven operating system for AI-native projects. It treats text as law, code as consequence and fixtures as precedent. By enforcing a clear authority gradient and mechanical gates, AGS prevents drift, preserves intent and allows continuous evolution across model swaps and refactors.

## Starting a Session (Mandatory)

Every new agent session **must** begin with the Genesis Prompt to bootstrap the governance system correctly.

1. Read the full prompt from `LAW/CANON/GENESIS.md`.
2. Prepend it as the system message, pack header, or first instruction.
3. Agents are instructed to remind you if it is missing.

This solves the bootstrap paradox: the agent knows the governance structure exists *before* reading any other file.

## Project layout

The repository is organised into eight layers. Files are loaded in the following order:

1. **LAW** - The constitution, decisions, and mechanical contracts.
    - `CANON/` - Constitutional rules (Genesis, Versioning, Glossary).
    - `CONTEXT/` - ADRs, preferences, and research history.
    - `CONTRACTS/` - Schemas and fixtures (Precedent).
2. **CAPABILITY** - Modular skills, tools, and MCP adapters.
    - `SKILLS/` - Atomic agent capabilities.
    - `MCP/` - Client integration and Semantic Core logic.
    - `TOOLS/` - Helper scripts (Critics, Linters).
3. **NAVIGATION** - Maps and roadmaps.
    - `MAPS/` - Data flows and ownership maps.
    - `CORTEX/` - Semantic index and search tools.
4. **MEMORY** - Packers and long-term state.
5. **THOUGHT** - Experimental labs and prototypes.
6. **INBOX** - Reports, research, and items for human review.

## Semantic Core (Token Compression)

The CORTEX layer includes a **Semantic Core** system that uses vector embeddings to enable massive token savings when dispatching tasks to smaller language models.

**How it works:**
- Big models (Opus) maintain semantic understanding via 384-dimensional vector embeddings in CORTEX
- Small models (Haiku) receive compressed task specifications with @Symbols and vector context
- Achieves **96% token reduction** per task (50,000 → 2,000 tokens)

**Quick start:**
```bash
# Test the semantic adapter
python CAPABILITY/MCP/semantic_adapter.py --test

# Rebuild the semantic index
python NAVIGATION/CORTEX/semantic/vector_indexer.py --rebuild

# Search the codebase
python CAPABILITY/MCP/semantic_adapter.py search --query "how to add a skill"
```

**Documentation:**
- [Semantic Core Quick Start](./NAVIGATION/CORTEX/README.md)
- [ADR-030](./LAW/CONTEXT/decisions/ADR-030-semantic-core-architecture.md)
- [Final Report](./INBOX/reports/12-28-2025-12-00_PHASE1_TRIPLE_WRITE_IMPLEMENTATION.md)

**Status:** Phase 1 (Vector Foundation) complete. See [Final Report](./INBOX/reports/12-28-2025-12-00_PHASE1_TRIPLE_WRITE_IMPLEMENTATION.md) for details.

## MCP integration (external clients)

AGS exposes an MCP server for IDEs and desktop clients. Use the entrypoint wrapper
`LAW/CONTRACTS/ags_mcp_entrypoint.py` to keep audit logs under allowed output
roots (`LAW/CONTRACTS/_runs/mcp_logs/`). Verify with the `mcp-smoke` or
`mcp-extension-verify` skills. See `CAPABILITY/MCP/README.md` for client config examples.

## How to use

This repository is a template: most files are placeholders that illustrate the intended structure. To adapt the system for your own project, fill in the canon, add decisions and ADRs, implement skills and write fixtures.

Agents interacting with the system should follow the protocol described in `CANON/CONTRACT.md`. In brief:

1. Load the canon first and respect its authority.
2. Consult context records before making changes.
3. Use the maps to find the right entrypoints.
4. Execute work through skills rather than ad-hoc scripts.
5. Validate changes using the runner in `LAW/CONTRACTS`.
6. Update the canon and changelog in the same commit when rules change.
7. Default to repo-only access (see `LAW/CONTEXT/decisions/ADR-012-privacy-boundary.md`).

## How to extend AGS

- Add a skill: create `CAPABILITY/SKILLS/<skill-name>/` with `SKILL.md`, a run script, a validation script, and `fixtures/<case>/input.json` plus `expected.json`.
- Add fixtures: place skill fixtures under `CAPABILITY/SKILLS/<skill-name>/fixtures/` and governance fixtures under `LAW/CONTRACTS/fixtures/`.
- Add ADRs: create a new `LAW/CONTEXT/decisions/ADR-xxx-*.md` and reference it in `LAW/CONTEXT/INDEX.md`.
- Use `BUILD/` for your project's build outputs (dist). It is disposable and should not contain authored content. The template writes its own artifacts under `LAW/CONTRACTS/_runs/`, `NAVIGATION/CORTEX/_generated/`, and `MEMORY/LLM_PACKER/_packs/`.
- Planning snapshots live under `LAW/CONTEXT/archive/planning/` (see `LAW/CONTEXT/archive/planning/INDEX.md`).

For more details, see individual files in the respective directories.
