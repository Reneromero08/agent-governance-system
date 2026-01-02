<!-- CONTENT_HASH: 0dd3a0c3fde497ea0ca5c1fd7b9b0d5f466626b13d40b14bb6e7793c8d36dd72 -->

# Entrypoints

This document lists the primary entrypoints where agents and humans are expected to make changes.

## 1. LAW (Rules)

- `LAW/CANON/CONTRACT.md`: Modify to change non-negotiable rules. Requires full ceremony.
- `LAW/CANON/CHANGELOG.md`: Log system behavior changes.
- `LAW/CONTEXT/decisions/`: Architecture Decision Records (ADR).
- `LAW/SCHEMAS/`: JSON validation schemas.
- `LAW/CONTRACTS/fixtures/`: Governance tests and privacy boundaries.

## 2. CAPABILITY (Tools)

- `CAPABILITY/SKILLS/`: Application code and agent toolkits.
- `CAPABILITY/TOOLS/`: Critics, linters, and cleaners.
- `CAPABILITY/MCP/`: Hardware interface server.
- `CAPABILITY/PIPELINES/`: DAG definitions for distributed execution.
- `CAPABILITY/PRIMITIVES/`: Core execution logic (nodes, runners).
- `CAPABILITY/TESTBENCH/`: Validation suite.

## 3. NAVIGATION (Maps)

- `NAVIGATION/CORTEX/`: Semantic Index configuration.
- `NAVIGATION/ROADMAPS/`: Strategy documents and planning.
- `NAVIGATION/MAPS/`: Architecture maps.

## 4. THOUGHT (Experiments)

- `THOUGHT/LAB/`: Volatile experimental features.
- `INBOX/research/`: Non-binding analysis.

## 5. MEMORY (Archives)

- `INBOX/reports/`: Signed implementation reports.
- `MEMORY/LLM_PACKER/`: Pack definitions.

---

## CATALYTIC-DPT Components

The Distributed Pipeline Toolkit components are now integrated into the buckets:
- **Skills**: `CAPABILITY/SKILLS/`
- **Pipelines**: `CAPABILITY/PIPELINES/`
- **Primitives**: `CAPABILITY/PRIMITIVES/`
- **Schemas**: `LAW/SCHEMAS/`
- **Lab**: `THOUGHT/LAB/`
