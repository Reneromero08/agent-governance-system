# File Ownership

To minimise conflicts and maintain clarity, this document assigns responsibility for different parts of the repository.  Ownership means that changes to those files should be reviewed by the designated team or individual.

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

Ownership can be shared or transferred, but must always be clearly documented.