# Canon Contract

This document defines the non-negotiable rules and the authority gradient for the Agent Governance System (AGS). It is the highest source of truth for all agents and humans interacting with this repository.

## Non-negotiable rules

1. **Text outranks code.**  The canon (this directory) defines the working spec for this repository; implementation must follow.
2. **No behavior change without ceremony.** Any change to the behavior of the system must:
   - create an ADR (Architecture Decision Record) under `CONTEXT/decisions/` to document the decision, rationale, and consequences (required for governance decisions; recommended for significant code changes);
   - add or update appropriate fixtures;
   - update the canon (if constraints change);
   - update the affected module docs when operation changes (MAPS, SKILLS, MEMORY, CORTEX, TOOLS), not just AGENTS and CONTRACT;
   - record the change in the changelog;
   - occur within the same merge request.
3. **Intent-gated canon and context edits.** CANON is a working spec and may be updated during system design and rule updates. CONTEXT is append-first; editing existing records requires explicit user instruction AND explicit task intent (see ADR-016). Deleting authored content is allowed only with explicit user instruction and confirmation, and CANON rules must follow the archiving requirements in `CANON/INVARIANTS.md`. Do not modify CANON or edit existing CONTEXT records as a side effect of unrelated tasks.
4. **Stable token grammar.** Tokens used to reference entities and rules form a stable API. Changes to tokens require a major version bump and deprecation cycle.
5. **Determinism.** Given the same inputs and canon, the system must produce the same outputs.
6. **Output roots.** System-generated artifacts must be written only to:
   - `CONTRACTS/_runs/` (includes fixtures, logs, entrypoint outputs)
   - `CORTEX/_generated/`
   - `MEMORY/LLM_PACKER/_packs/`

   `BUILD/` is reserved for user build outputs and must not be used for system artifacts.

   **Logging policy** (see ADR-015): All system logs (emergency, audit, crisis, steward) must be written under `CONTRACTS/_runs/<purpose>_logs/` subdirectories.

7. **Commit ceremony.** Every `git commit`, `git push`, and release publication requires explicit, per-instance user approval. Agents may not infer authorization from phrases like "proceed," "continue," or "let's move on to the next task." One approval authorizes one commit only; subsequent work requires a new approval. Explicit directives that include "commit" authorize a commit once checks have passed and staged files have been listed; no additional confirmation prompt is required. Explicit composite directives that include "commit", "push", and "release" (for example, "commit, push, and release") count as approval for each action listed in that request. After the ceremony steps are completed (checks run, staged files listed), short confirmations such as "go on" count as approval for those listed actions. See `AGENTS.md` Section 10 and `CONTEXT/preferences/STYLE-001-commit-ceremony.md`.
8. **Sovereign override interface.** If a user prompt contains `MASTER_OVERRIDE`, the agent is authorized to bypass any repository governance rule for that prompt only. Each use must be logged to `CONTRACTS/_runs/override_logs/master_override.jsonl`, and override logs must not be read or quoted unless the user prompt also contains `MASTER_OVERRIDE`.
9. **Privacy boundary.** Agents must not access or scan files outside the repository root unless the user explicitly requests those paths in the same prompt. Avoid user profile and OS-level directories by default; ask before any out-of-repo access.

## Intent gate

Only change CANON or edit existing CONTEXT records when the task is explicitly about rules, governance, or memory updates. If intent is ambiguous, ask one clarifying question before touching CANON or existing CONTEXT records. Changes are reversible; if a change is wrong, revert it.

## Authority gradient

When conflicts arise, the following order of precedence applies:

1. `CANON/AGREEMENT.md`
2. `CANON/CONTRACT.md` (this file)
3. `CANON/INVARIANTS.md`
4. `CANON/VERSIONING.md`
5. `AGENTS.md`
6. Context records (`CONTEXT/decisions/`, `CONTEXT/rejected/`, `CONTEXT/preferences/`)
7. `MAPS/*`
8. User instructions
9. Implementation convenience

Never invert this order. Fixtures in `CONTRACTS/` validate behavior but do not override canon.

## Change ceremony

To change the canon:

1. Draft an Architecture Decision Record (ADR) under `CONTEXT/decisions/` explaining the context, decision and rationale.
2. Update the relevant canon file(s) with the new rule or modification.
3. Add or update fixtures in `CONTRACTS/fixtures/` to enforce the new rule.
4. Increment the version in `CANON/VERSIONING.md` accordingly.
5. Add an entry to `CANON/CHANGELOG.md` describing the change.
6. Submit a merge request. The critic and runner must pass before the change is accepted.
