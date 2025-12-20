# Data Flow

This document explains how information moves through the Agent Governance System during a typical session.

1. **Load canon** - On startup, agents read the canon files in order (`CONTRACT.md`, invariants, versioning, glossary). This establishes the rules.
2. **Load context** - Agents then read relevant context records (ADRs, rejections, preferences, open issues). These provide historical and stylistic guidance.
3. **Query cortex** - Instead of scanning files, agents query the shadow index (`CORTEX/_generated/cortex.json`) via `CORTEX/query.py`. The cortex returns structured metadata about pages, assets and tokens.
4. **Select entrypoint** - Using the maps, agents determine which files or skills need to change to achieve the goal.
5. **Execute skill** - The agent invokes a skill (a script under `SKILLS/`) to perform the action. Skills operate on data provided by the cortex and abide by the canon constraints.
6. **Validate via fixtures** - After the skill runs, fixtures in `CONTRACTS/fixtures/` are executed by the runner. Any failure blocks the merge. Runner artifacts are written under `CONTRACTS/_runs/`.
7. **Update memory** - If the work is completed, the packer serialises the current state into a pack for future sessions under `MEMORY/LLM-PACKER-1.0/_packs/`. The manifest includes file hashes and canon version.
