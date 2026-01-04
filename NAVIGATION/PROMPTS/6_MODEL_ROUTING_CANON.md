---
title: MODEL_ROUTING_CANON
version: 1.0
status: CANONICAL
generated_on: 2026-01-04
scope: Model routing decision ladder for optimal token efficiency
---
<!-- CANON_HASH: db13df040be8f37535ce3db87631a2fb4251af6cfb0fadbdde226fc5c9933636 -->

# MODEL_ROUTING_CANON

## Purpose
Route work across models to minimize wasted tokens and maximize correctness.

Models are execution units. Prefer **procedural execution** over “designing in the prompt.”

## Global defaults
- **Thinking OFF by default** on every model.
- Turn thinking ON only when the task requires invention or a true design fork that cannot be resolved from repo canon/contracts/packs.
- Escalate models **only** after acceptance checks fail or coherence requirements exceed the current model.

## Authority Tiers (Policy)
- **Planner-capable Models**: Claude Sonnet (Thinking), Claude Opus (Thinking/Non-thinking). Designated for planning, repository navigation, and complex decomposition.
- **Non-planner Models**: Gemini Flash, Gemini Pro Low, GPT Codex, Grok Code Fast. Restricted to mechanical execution only.
- **Enforcement**: Non-planner models MUST carry a `plan_ref` artifact or identifier from a planner-capable model. Execution without a valid reference is a policy breach.

## Decision ladder (strict)
1) **Mechanical from an explicit checklist/spec?**  
   → Use the smallest reliable model.

2) **Large/high-risk but fully specified?**  
   → Use a high-authority execution model (non-thinking).

3) **Spec incomplete/contradictory/ambiguous?**  
   → Tighten the spec or pack first. Use a thinking model only as last resort.

## Routing matrix

### A) Mechanical edits, migrations, normalization
Use when:
- File moves/renames, formatting normalization
- Regex refactors with strict boundaries
- “Do exactly X on these files”

Model order:
1) Gemini Flash (thinking OFF)  
2) Gemini Pro Low (thinking OFF)  
3) Claude Sonnet non-thinking  
4) Grok Code Fast only for **low-risk single-file** mechanical edits (no git chores, no multi-file invariants)

Acceptance pattern:
- Explicit file list / globs
- Explicit rules (do / do not touch)
- Explicit validation commands and expected outcomes

### B) Implementation from a complete spec
Use when:
- Algorithms, formats, paths, and acceptance tests are specified
- You want code produced fast with minimal invention

Model order:
1) GPT Codex Mini for small bounded functions  
2) GPT Codex for normal implementation  
3) GPT Codex Max only when change spans many files and needs more retention  
4) Claude Sonnet non-thinking when multi-file coherence is more important than speed

Thinking:
- OFF

Acceptance pattern:
- Steps as bullets
- Exact file paths to create/edit
- Determinism rules (sorting, canonical JSON, stable hashes)
- Runnable tests + exit criteria

### C) Multi-file build with high correctness requirements
Use when:
- Large change set across many modules
- Determinism and invariants are critical
- Spec is complete, but surface area is big

Model:
- Claude Opus non-thinking

Thinking:
- OFF unless truly blocked by a repo-backed design fork

Acceptance pattern:
- Non-negotiable constraints at top
- Forbidden edits list
- Required outputs (paths)
- Validation commands
- Commit plan (1–2 commits)

### D) Design forks, new primitives, ambiguous requirements
Use when:
- Must invent a new primitive
- Repo contracts conflict
- Multiple viable designs exist and a choice must be made

Model order:
1) Claude Sonnet Thinking  
2) Claude Opus Thinking only if unresolved and stakes are extreme

Output requirements:
- Define options (2–4 max)
- Evaluation criteria (what matters)
- Choose one design
- Minimal plan to implement without scope creep

## Escalation rules (avoid token waste)
Do not escalate because a task “feels important.” Escalate only when:
1) Acceptance checks fail, or
2) The task requires broader coherence than the current model can maintain, or
3) The task is underspecified and cannot be clarified by reading repo canon/contracts/packs.

## Quick presets
- **Fastest safe:** Gemini Flash (thinking OFF)
- **Coding workhorse:** Claude Sonnet (thinking OFF)
- **Big specified change, do not mess up:** Claude Opus (thinking OFF)
- **I need invention:** Sonnet Thinking → Opus Thinking only if needed

## Minimal ticket template (paste into any agent)
- Goal:
- Scope:
- Forbidden edits:
- Inputs (paths):
- Outputs (paths):
- Algorithm (steps):
- Determinism rules:
- Validation commands:
- Exit criteria:
- Commit message(s):

## Opus constraint (critical)
Opus is a **high-authority execution model**, not an exploration model.
Prompts to Opus must be fully specified, procedural, and bounded.

## Lint precondition (hard stop)
If lint status is missing or FAIL, routing to any execution model is forbidden.
The only allowed action is to run the canonical linter or repair the prompt pack to pass lint.
