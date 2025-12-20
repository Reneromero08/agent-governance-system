# Agent Governance System (AGS)

The Agent Governance System (AGS) is a constitutional framework for durable, multi-agent intelligence.

This repository provides a language-driven operating system for AI-native projects. It treats text as law, code as consequence and fixtures as precedent. By enforcing a clear authority gradient and mechanical gates, AGS prevents drift, preserves intent and allows continuous evolution across model swaps and refactors.

## Project layout

The repository is organised into six layers. Files are loaded in the following order:

1. **CANON** - the constitution and invariants of the system.
2. **CONTEXT** - decisions, rejections and preferences that record why things are the way they are.
3. **MAPS** - entrypoints and data flow descriptions that tell agents where to make changes.
4. **SKILLS** - modular capabilities packaged with contracts and fixtures.
5. **CONTRACTS** - fixtures and schemas used to mechanically enforce rules.
6. **MEMORY** - packers, manifests and combined state for context handoff.

Additional directories contain tooling:

- **CORTEX** - a shadow index (e.g. SQLite) that agents query instead of the raw filesystem.
- **TOOLS** - helper scripts such as critics and linters.

## How to use

This repository is a template: most files are placeholders that illustrate the intended structure. To adapt the system for your own project, fill in the canon, add decisions and ADRs, implement skills and write fixtures.

Agents interacting with the system should follow the protocol described in `CANON/CONTRACT.md`. In brief:

1. Load the canon first and respect its authority.
2. Consult context records before making changes.
3. Use the maps to find the right entrypoints.
4. Execute work through skills rather than ad-hoc scripts.
5. Validate changes using the runner in `CONTRACTS`.
6. Update the canon and changelog in the same commit when rules change.

## How to extend AGS

- Add a skill: create `SKILLS/<skill-name>/` with `SKILL.md`, a run script, a validation script, and `fixtures/<case>/input.json` plus `expected.json`.
- Add fixtures: place skill fixtures under `SKILLS/<skill-name>/fixtures/` and governance fixtures under `CONTRACTS/fixtures/`.
- Add ADRs: create a new `CONTEXT/decisions/ADR-xxx-*.md` and reference it in `CONTEXT/INDEX.md`.
- Use `BUILD/` for your project's build outputs (dist). It is disposable and should not contain authored content. The template writes its own artifacts under `CONTRACTS/_runs/`, `CORTEX/_generated/`, and `MEMORY/_packs/`.

For more details, see individual files in the respective directories.
