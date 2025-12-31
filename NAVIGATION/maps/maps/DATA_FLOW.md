# Data Flow

This document explains how information moves through the Agent Governance System during a typical session.

## Core AGS Flow

1. **Load canon** - On startup, agents read the canon files in order (`CONTRACT.md`, invariants, versioning, glossary). This establishes the rules.
2. **Load context** - Agents then read relevant context records (ADRs, rejections, preferences, open issues). These provide historical and stylistic guidance.
3. **Query cortex** - Instead of scanning files, agents query the shadow index (`CORTEX/_generated/cortex.db`) via `CORTEX/query.py`. The cortex returns structured metadata about pages, assets and tokens.
4. **Select entrypoint** - Using the maps, agents determine which files or skills need to change to achieve the goal.
5. **Execute skill** - The agent invokes a skill (a script under `SKILLS/`) to perform the action. Skills operate on data provided by the cortex and abide by the canon constraints.
6. **Validate via fixtures** - After the skill runs, fixtures in `LAW/CONTRACTS/fixtures/` are executed by the runner. Any failure blocks the merge. Runner artifacts are written under `LAW/CONTRACTS/_runs/`.
7. **Update memory** - If the work is completed, the packer serialises the current state into a pack for future sessions under `MEMORY/LLM_PACKER/_packs/`. The manifest includes file hashes and canon version.

## CATALYTIC-DPT Pipeline Flow

For distributed pipeline execution, CATALYTIC-DPT follows its own data flow:

1. **Load pipeline definition** - Read DAG from `CATALYTIC-DPT/PIPELINES/` specifying nodes and dependencies.
2. **Validate against schemas** - Pipeline structures are validated against `CATALYTIC-DPT/SCHEMAS/`.
3. **Schedule nodes** - The DAG scheduler determines execution order based on dependencies.
4. **Execute primitives** - Core operations from `CATALYTIC-DPT/PRIMITIVES/` are invoked.
5. **Orchestrate swarm** - For distributed work, the swarm-orchestrator dispatches tasks to ant-workers.
6. **Generate receipts** - Execution receipts record success/failure and outputs.
7. **Restore on failure** - The restore runner can replay failed nodes from checkpoints.

## MCP Integration Flow

When accessed via MCP (Model Context Protocol):

1. **Client connects** - IDE or tool connects to `CAPABILITY/MCP/server.py` via stdio.
2. **Audit wrapper** - Requests pass through `LAW/CONTRACTS/ags_mcp_entrypoint.py` for logging.
3. **Tool dispatch** - MCP server routes to appropriate AGS tool or skill.
4. **Response** - Results returned to client with governance metadata.
