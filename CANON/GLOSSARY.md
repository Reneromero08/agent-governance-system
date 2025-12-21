# Glossary

This glossary defines important terms used throughout the Agent Governance System. These definitions are part of the token grammar. Each term serves as a stable handle for a concept.

- **Canon** - The collection of files under `CANON/` that define the law and invariants of the system.
- **Context** - Records in `CONTEXT/` such as ADRs, rejections and preferences that provide rationale and decision history.
- **Map** - Documents in `MAPS/` that describe the structure of the repository and direct agents to the correct entrypoints for changes.
- **Skill** - A modular, versioned capability encapsulated in its own directory under `SKILLS/`. A skill includes a manifest (`SKILL.md`), scripts, and fixtures.
- **Contract** - A rule or constraint encoded as fixtures and schemas in `CONTRACTS/`. Contracts are enforced by the runner.
- **Fixture** - A concrete test case that captures an invariant or precedent. Fixtures must pass in order for changes to be merged.
- **Runner** - The script in `CONTRACTS/runner.py` that executes fixtures and reports pass/fail status.
- **Cortex** - A shadow index built from the repository content. Skills and agents query the cortex via `CORTEX/query.py` instead of reading files directly.
- **Memory** - The state of an agent or project, serialised by the packer in `MEMORY/`. Memory packs contain the minimal context required for agents to resume work.
- **Token grammar** - The set of canonical phrases and symbols (such as the glossary terms) that encode meaning in a compact form. Tokens are stable across versions to decouple intent from implementation.
- **BUILD** - Reserved for user build outputs produced by template users. The template's own tooling must not write system artifacts here. It is disposable and should not contain authored content.
- **ADR** - Architecture Decision Record. A document under `CONTEXT/decisions/` that captures the context, decision, and rationale for a significant choice.
- **Pack** - A bundled snapshot of repository content produced by the packer. Packs are self-contained and include manifests for integrity verification.
- **Manifest** - A machine-readable inventory listing the contents and hashes of a pack. Used to verify pack integrity on load.
- **Critic** - A validation script that checks code or content against canon rules. Critics run as part of the pre-merge gate.
- **Authority gradient** - The ordered hierarchy of sources that determines which rule takes precedence when conflicts arise. Defined in `CANON/CONTRACT.md`.
- **Invariant** - A decision that cannot be changed without a major version bump and migration plan. Listed in `CANON/INVARIANTS.md`.
- **Change ceremony** - The required steps to modify canon: ADR, update canon, add fixtures, bump version, update changelog.
- **Entity** - A record in the cortex index representing a file or concept. Entities have an id, type, title, tags, and paths.
- **Query** - A function in `CORTEX/query.py` that retrieves entities from the cortex index. Skills must use queries instead of filesystem access.
