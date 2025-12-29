# System Map

This map describes the overall architecture of the Agent Governance System. It provides a high-level view of the interactions between layers.

```
┌──────────────┐
│   CANON      │ ← highest authority
└───────┬──────┘
        │
┌───────▼──────┐
│   CONTEXT    │ - why and when
└───────┬──────┘
        │
┌───────▼──────┐
│    MAPS      │ - where
└───────┬──────┘
        │
┌───────▼──────┐
│    SKILLS    │ - how
└───────┬──────┘
        │
┌───────▼──────┐
│  CONTRACTS   │ - enforce
└───────┬──────┘
        │
┌───────▼──────┐
│   MEMORY     │ - persist and pack
└──────────────┘
```

## Supporting Components

```
┌──────────────┐
│   CORTEX     │ - indexing and querying across all layers
└──────────────┘

┌──────────────┐
│    TOOLS     │ - critics, linters, validators, runtime utilities
└──────────────┘

┌──────────────┐
│     MCP      │ - Model Context Protocol server for IDE integration
└──────────────┘
```

## Subsystem: CATALYTIC-DPT

CATALYTIC-DPT is a self-contained distributed pipeline toolkit with its own governance structure:

```
CATALYTIC-DPT/
├── AGENTS.md        - agent definitions for pipeline execution
├── SKILLS/          - pipeline-specific skills (swarm-orchestrator, ant-worker, etc.)
├── PIPELINES/       - DAG definitions and pipeline configurations
├── PRIMITIVES/      - core building blocks (nodes, executors)
├── CONTRACTS/       - pipeline-specific fixtures and validation
├── SCHEMAS/         - JSON schemas for pipeline structures
├── SPECTRUM/        - capability spectrum definitions
├── LAB/             - experimental features and research
├── TESTBENCH/       - testing infrastructure
├── FIXTURES/        - test fixtures for pipeline validation
└── CONTEXT/demos/           - demonstration pipelines
```

## Other Directories

| Directory | Purpose |
|-----------|---------|
| `CONTEXT/demos/` | Project-level demonstration scripts |
| `.github/` | GitHub workflows and CI configuration |

Each arrow in the main hierarchy represents a dependency. For example, skills depend on maps to know where to operate, and contracts enforce the correct behaviour of skills.
