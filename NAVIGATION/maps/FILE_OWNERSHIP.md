<!-- CONTENT_HASH: 84ac8b07daf58d6881443d1a8450c2525ec32e52e2d0b5326455ceca2e055989 -->

# File Ownership

To minimise conflicts and maintain clarity, this document assigns responsibility for different parts of the repository. Ownership means that changes to those files should be reviewed by the designated team or individual.

## 1. LAW (Governance)

| Path | Owner | Notes |
|---|---|---|
| `LAW/CANON/*` | Core maintainers | Changes require full ceremony and consensus |
| `LAW/CONTRACTS/*` | QA / Policy team | Run ledgers, audits, and fixture enforcement |
| `LAW/CONTEXT/decisions/*` | Architecture reviewers | ADRs require cross-team approval |
| `LAW/SCHEMAS/*` | Schema maintainers | JSON schemas for all subsystems |
| `AGENTS.md` | Project leads | Operational contracts |

## 2. CAPABILITY (Tools)

| Path | Owner | Notes |
|---|---|---|
| `CAPABILITY/SKILLS/*` | Skill authors | Each skill directory should have an identified owner |
| `CAPABILITY/TOOLS/*` | Tooling team | Critics, linters, and migration scripts |
| `CAPABILITY/MCP/*` | Integration team | MCP server and hardware interfaces |
| `CAPABILITY/PIPELINES/*` | Pipeline designers | Distributed DAG definitions |
| `CAPABILITY/PRIMITIVES/*` | Core engineers | Fundamental building blocks (Safety critical) |
| `CAPABILITY/TESTBENCH/*` | QA team | Testing infrastructure |

## 3. NAVIGATION (Maps)

| Path | Owner | Notes |
|---|---|---|
| `NAVIGATION/CORTEX/*` | Index team | Semantic database schema and build scripts |
| `NAVIGATION/maps/*` | Architecture team | Usage maps and flow diagrams |
| `INDEX.md` | Automated/Stewards | Directory indices |

## 4. THOUGHT (Lab)

| Path | Owner | Notes |
|---|---|---|
| `THOUGHT/LAB/*` | Research leads | Experimental features (Volatile) |
| `THOUGHT/CONTEXT/research/*`| Researchers | Non-binding analysis and notes |
| `THOUGHT/CONTEXT/demos/*` | Demo maintainers | Proof of concept scripts |

## 5. MEMORY (History)

| Path | Owner | Notes |
|---|---|---|
| `MEMORY/LLM_PACKER/*` | Automation leads | Packer engine and snapshot logic |
| `MEMORY/*` | Persistence team | Historical archives and manifest formats |
| `INBOX/reports/*` | Implementation teams | Signed reports and status updates |
| `INBOX/research/*` | Researchers | Incoming research artifacts |

Ownership can be shared or transferred, but must always be clearly documented.
