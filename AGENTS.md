# AGENTS.md

Agent Operating Contract for the Agent Governance System (AGS)

This file defines how autonomous or semi-autonomous agents operate inside this
repository. It is procedural authority. If unclear, defer to CANON.

## 1. Required startup sequence (non-negotiable)

Before taking any action, an agent MUST:

1. Read:
   - CANON/CONTRACT.md
   - CANON/INVARIANTS.md
   - CANON/VERSIONING.md
2. Read this file (AGENTS.md) in full
3. Identify the current canon_version
4. Identify whether the task is:
   - governance change
   - skill implementation
   - build execution
   - documentation only

If any of the above steps are skipped, the agent must stop.

## 1A. Question-first gate (no-write)

If the user is asking questions, requesting analysis, or requesting a strategy without explicitly approving implementation, the agent MUST:

- answer first (no edits)
- avoid creating, modifying, deleting, or committing files
- avoid running commands that write artifacts
- ask for explicit approval before making changes (example: "Do you want me to implement this now?")

## 2. Authority gradient

If instructions conflict, obey in this order:

1. CANON/CONTRACT.md
2. CANON/INVARIANTS.md
3. CANON/VERSIONING.md
4. AGENTS.md
5. CONTEXT records (ADRs, rejections, preferences)
6. MAPS/*
7. User instructions
8. Implementation convenience

Never invert this order.

## 3. Mutation rules

Agents MAY:
- create or modify files under:
  - SKILLS/
  - CONTRACTS/
  - CORTEX/ (implementation), and `CORTEX/_generated/` (generated)
  - MEMORY/ (implementation), and `MEMORY/LLM-PACKER-1.1/_packs/` (generated)
  - BUILD/ (user build outputs only)
- append new records under CONTEXT/
- consult CONTEXT/research as optional, non-binding input

Agents MAY NOT:
- modify CANON/* unless explicitly instructed
- delete authored content
- rewrite history in CONTEXT/*
- touch generated artifacts outside:
  - CONTRACTS/_runs/
  - CORTEX/_generated/
  - MEMORY/LLM-PACKER-1.1/_packs/

Generated files must be clearly marked as generated.

Research under CONTEXT/research is opt-in and non-binding. It must not be
treated as canon.

## 4. Build output rules

System-generated artifacts MUST be written only to:

- CONTRACTS/_runs/
- CORTEX/_generated/
- MEMORY/LLM-PACKER-1.1/_packs/

`BUILD/` is reserved for user build outputs. It must not be used for system artifacts.

- BUILD/ is disposable
- BUILD/ is the dist equivalent
- BUILD/ may be wiped at any time
- No authored content belongs in BUILD/

If a task requires writing elsewhere, the agent must stop and ask.

## 5. Skills-first execution

Agents must not perform arbitrary actions.

All non-trivial work must be performed via a skill:

- If a suitable skill exists, use it.
- If no suitable skill exists:
  - propose a new skill
  - write SKILL.md first
  - define fixtures
  - then implement

Direct ad-hoc scripting is forbidden.

## 6. Fixtures as law

If an agent changes behavior, it MUST:

1. Add or update fixtures
2. Run CONTRACTS/runner.py
3. Ensure all fixtures pass
4. Update CANON if constraints change

If fixtures fail, the change does not exist.

## 7. Uncertainty protocol

If any of the following are true, the agent must stop and ask:

- intent is ambiguous
- multiple canon interpretations exist
- change would affect invariants
- output location is unclear
- irreversible action is required

Guessing is forbidden.

## 8. Determinism requirement

Agent actions must be:

- deterministic
- reproducible
- explainable via canon and context

No randomness.
No hidden state.
No silent side effects.

## 9. Exit conditions

An agent should stop when:
- the requested task is complete
- fixtures pass
- outputs are written to the allowed artifact roots
- any blocking uncertainty appears

Agents must not continue "optimizing" beyond scope.
