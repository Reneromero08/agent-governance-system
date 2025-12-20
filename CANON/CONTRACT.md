# Canon Contract

This document defines the non-negotiable rules and the authority gradient for the Agent Governance System (AGS). It is the highest source of truth for all agents and humans interacting with this repository.

## Non-negotiable rules

1. **Text outranks code.**  The canon (this directory) defines the working spec for this repository; implementation must follow.
2. **No behavior change without ceremony.** Any change to the behavior of the system must:
   - add or update appropriate fixtures;
   - update the canon (if constraints change);
   - record the change in the changelog;
   - occur within the same merge request.
3. **Intent-gated canon and context edits.** CANON is a working spec and may be updated during system design and rule updates. CONTEXT is append-first; editing existing records requires explicit instruction. Do not modify CANON or edit existing CONTEXT records as a side effect of unrelated tasks.
4. **Stable token grammar.**  Tokens used to reference entities and rules form a stable API.  Changes to tokens require a major version bump and deprecation cycle.
5. **Determinism.** Given the same inputs and canon, the system must produce the same outputs.
6. **Output roots.** System-generated artifacts must be written only to:
   - `CONTRACTS/_runs/`
   - `CORTEX/_generated/`
   - `MEMORY/LLM-PACKER-1.1/_packs/`

   `BUILD/` is reserved for user build outputs and must not be used for system artifacts.

## Intent gate

Only change CANON or edit existing CONTEXT records when the task is explicitly about rules, governance, or memory updates. If intent is ambiguous, ask one clarifying question before touching CANON or existing CONTEXT records. Changes are reversible; if a change is wrong, revert it.

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
5. Add an entry to `CANON/CHANGELOG.md` describing the change.
6. Submit a merge request. The critic and runner must pass before the change is accepted.
