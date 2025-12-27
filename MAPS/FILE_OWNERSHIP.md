# File Ownership

To minimise conflicts and maintain clarity, this document assigns responsibility for different parts of the repository. Ownership means that changes to those files should be reviewed by the designated team or individual.

## Core AGS

| Path | Owner | Notes |
|---|---|---|
| `CANON/*` | Core governance maintainers | Changes require full ceremony and consensus |
| `CONTEXT/decisions/*` | Architecture reviewers | Review ADRs for correctness and rationale |
| `CONTEXT/preferences/*` | Style guides team | Ensure preferences reflect team consensus |
| `CONTEXT/rejected/*` | Architecture reviewers | Document rejected proposals |
| `MAPS/*` | Project leads | Update maps when adding new capabilities |
| `SKILLS/*` | Skill authors | Each skill directory should have a clearly identified owner |
| `CONTRACTS/*` | QA and policy maintainers | Fixtures and schemas enforce behaviour |
| `MEMORY/*` | Persistence maintainers | Manage the packer and manifest formats |
| `MEMORY/LLM_PACKER/Engine/*` | Automation leads | Core packer logic and launcher assets |
| `CORTEX/*` | Index team | Maintain the schema and build scripts for the shadow index |
| `TOOLS/*` | Tooling team | Implement critics, linters and migration scripts |
| `MCP/*` | Integration team | MCP server and protocol implementation |

## CATALYTIC-DPT Subsystem

| Path | Owner | Notes |
|---|---|---|
| `CATALYTIC-DPT/AGENTS.md` | Pipeline architects | Agent definitions for distributed execution |
| `CATALYTIC-DPT/SKILLS/*` | Pipeline skill authors | Swarm orchestrator, ant-worker, etc. |
| `CATALYTIC-DPT/PIPELINES/*` | Pipeline designers | DAG definitions and configurations |
| `CATALYTIC-DPT/PRIMITIVES/*` | Core pipeline team | Fundamental building blocks |
| `CATALYTIC-DPT/CONTRACTS/*` | Pipeline QA | Pipeline-specific fixtures |
| `CATALYTIC-DPT/SCHEMAS/*` | Schema maintainers | JSON schemas for pipeline structures |
| `CATALYTIC-DPT/SPECTRUM/*` | Capability team | Spectrum definitions |
| `CATALYTIC-DPT/LAB/*` | Research leads | Experimental features |
| `CATALYTIC-DPT/TESTBENCH/*` | Pipeline QA | Testing infrastructure |

## Root-Level Files

| Path | Owner | Notes |
|---|---|---|
| `AGENTS.md` | Project leads | Top-level agent definitions |
| `AGS_ROADMAP_MASTER.md` | Project leads | Master roadmap and planning |
| `README.md` | Documentation team | Project overview |
| `demos/*` | Demo maintainers | Demonstration scripts |
| `.github/*` | DevOps team | CI/CD workflows |

Ownership can be shared or transferred, but must always be clearly documented.
