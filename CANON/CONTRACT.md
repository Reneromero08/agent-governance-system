# Canon Contract

This document defines the non-negotiable rules and the authority gradient for the Agent Governance System (AGS). It is the highest source of truth for all agents and humans interacting with this repository.

## Non-negotiable rules

1. **Text outranks code.**  The canon (this directory) defines the law; implementation must follow.
2. **No behavior change without ceremony.** Any change to the behavior of the system must:
   - add or update appropriate fixtures;
   - update the canon (if constraints change);
   - occur within the same merge request.
3. **No mutation of authored content.**  Agents may not modify files under `CANON`, `CONTEXT` or authored assets without explicit permission.
4. **Stable token grammar.**  Tokens used to reference entities and rules form a stable API.  Changes to tokens require a major version bump and deprecation cycle.
5. **Determinism.** Given the same inputs and canon, the system must produce the same outputs.
6. **Output roots.** System-generated artifacts must be written only to:
   - `CONTRACTS/_runs/`
   - `CORTEX/_generated/`
   - `MEMORY/LLM-PACKER-1.1/_packs/`

   `BUILD/` is reserved for user build outputs and must not be used for system artifacts.

## Authority gradient

When conflicts arise, the following order of precedence applies:

1. `CANON/CONTRACT.md`
2. Other files in `CANON/`
3. `ROADMAP.md`
4. Fixtures in `CONTRACTS/`
5. Implementation (code and skills)

## Change ceremony

To change the canon:

1. Draft an Architecture Decision Record (ADR) under `CONTEXT/decisions/` explaining the context, decision and rationale.
2. Update the relevant canon file(s) with the new rule or modification.
3. Add or update fixtures in `CONTRACTS/fixtures/` to enforce the new rule.
4. Increment the version in `CANON/VERSIONING.md` accordingly.
5. Submit a merge request. The critic and runner must pass before the change is accepted.
