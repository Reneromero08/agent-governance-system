---
name: hermes-harness
description: Use Hermes Agent as an orchestration harness that decomposes tasks, delegates focused work to subagents, and synthesizes verified results.
version: 0.1.0
status: Active
required_canon_version: ">=3.0.0"
platforms: [macos, linux]
metadata:
  hermes:
    tags: [orchestration, delegation, subagents, repo, automation]
    category: orchestration
    requires_toolsets: [delegation]
    config:
      - key: hermes_harness.max_parallel_subagents
        description: Default maximum number of concurrent leaf subagents.
        default: 3
        prompt: Maximum concurrent subagents
      - key: hermes_harness.max_spawn_depth
        description: Delegation depth. Keep at 1 unless nested orchestration is explicitly needed.
        default: 1
        prompt: Maximum delegation depth
      - key: hermes_harness.default_mode
        description: Default routing mode for task decomposition.
        default: auto
        prompt: Default harness mode
      - key: hermes_harness.require_parent_synthesis
        description: Require the parent to synthesize all child outputs before replying.
        default: true
        prompt: Require parent synthesis
---
# Hermes Harness

## When to Use

Use this skill when the task is too wide for a single reasoning thread and can be split into independent subtasks, such as repo audits, parallel research, test triage, migration planning, feature decomposition, multi-file code review, documentation passes, benchmark review, or synthesis across several evidence streams.

Do not use this skill for trivial single-step work, questions that need one local answer, or tasks where the user explicitly forbids subagents.

## Core Model

The parent agent is the harness. It owns decomposition, context packaging, task assignment, final verification, and synthesis. Subagents are temporary workers with isolated context. They should receive complete task packets and return structured summaries.

The parent must never assume a subagent remembers anything from the parent conversation. Put all required context into the `goal` and `context` fields.

## Procedure

1. **Classify the task.** Choose one mode: `plan`, `research`, `audit`, `code`, `debug`, `docs`, `synthesis`, or `auto`.
2. **Collect context.** Identify workspace path, relevant files, constraints, user goals, acceptance criteria, and forbidden actions.
3. **Decompose.** Split only along clean boundaries. Good boundaries are file groups, research questions, subsystems, hypotheses, test categories, or independent implementation tickets.
4. **Choose worker count.** Default to 1-3 leaf workers. More than 3 needs an explicit reason.
5. **Build task packets.** Each subagent packet must include goal, context, allowed toolsets, expected output format, verification steps, and stop conditions.
6. **Delegate.** Use `delegate_task` for reasoning-heavy subtasks. Use `execute_code` only for mechanical processing.
7. **Supervise.** Watch for contradictory findings, missing evidence, overbroad edits, and incomplete verification.
8. **Synthesize.** Combine child outputs into one coherent answer. Resolve conflicts. State what changed, what was verified, and what remains uncertain.
9. **Persist learning.** If this workflow reveals a repeatable procedure, update or create a narrower skill with `skill_manage`.

## Delegation Patterns

### Single focused worker

Use when one isolated expert pass is enough.

```python
delegate_task(
    goal="Review the auth module for brittle error handling",
    context="""
Workspace: /absolute/path/to/repo
Files: src/auth/login.py, src/auth/session.py
Constraints: read-only review unless a minimal fix is obvious.
Return: findings, severity, evidence, recommended patch.
""",
    toolsets=["terminal", "file"]
)
```

### Parallel leaf workers

Use when subtasks are independent.

```python
delegate_task(tasks=[
    {
        "goal": "Audit test failures and identify root causes",
        "context": "Workspace: /repo. Run pytest. Do not edit files. Return failing tests, causes, and likely fixes.",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Review public API docs for gaps",
        "context": "Workspace: /repo. Inspect README and docs/. Return missing setup, usage, and migration notes.",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Map code ownership by subsystem",
        "context": "Workspace: /repo. Inspect src/. Return subsystem map and high-risk files.",
        "toolsets": ["terminal", "file"]
    }
])
```

### Orchestrator child

Use only when a child must run its own small fan-out. Keep this rare. Require the user task to justify nested decomposition and ensure `delegation.max_spawn_depth` permits it.

```python
delegate_task(
    goal="Survey three implementation strategies and recommend one",
    role="orchestrator",
    context="Workspace: /repo. Compare minimal patch, adapter layer, and rewrite. Return recommendation with tradeoffs.",
    toolsets=["delegation", "terminal", "file"]
)
```

## Toolset Presets

| Preset | Toolsets | Use |
|---|---|---|
| `repo-read` | `terminal`, `file` | Inspect repo without making edits. |
| `repo-edit` | `terminal`, `file` | Make contained code/doc edits and run checks. |
| `web-research` | `web` | Current public research. |
| `hybrid-research` | `web`, `terminal`, `file` | Research plus local synthesis or artifact writing. |
| `orchestrator` | `delegation`, `terminal`, `file`, `web` | Nested planning. Use sparingly. |

## Subagent Brief Contract

Every delegated task must include:

```text
GOAL: one sentence with a concrete deliverable.
WORKSPACE: absolute path or "none".
SCOPE: included files, directories, topics, or hypotheses.
CONTEXT: all facts the worker needs. No references to "above" or "previous".
CONSTRAINTS: allowed edits, forbidden edits, budget, style, user preferences.
TOOLS: explicit toolsets.
OUTPUT: exact sections required.
VERIFY: commands, checks, citations, or reasoning checks.
STOP: when to stop and what to return if blocked.
```

## Required Parent Synthesis Format

Return final answers in this shape unless the user asked for another format:

```markdown
# Result

## What I delegated
- Worker 1: ...
- Worker 2: ...

## Findings
- ...

## Changes made
- ...

## Verification
- Command/check: result

## Conflicts or uncertainty
- ...

## Next move
- ...
```

## Verification

Before final reply, the parent agent must check:

- Every subagent had a complete context packet.
- No child made edits outside its scope.
- All claims are supported by child evidence, local files, commands, or cited sources.
- Conflicting child results were reconciled.
- Final response is shorter and clearer than the combined raw child outputs.

## Pitfalls

- **Vague goal:** `Fix the error` fails because the child knows nothing. Include stack trace, file path, commands, and expected behavior.
- **Overlapping edits:** Two workers should not edit the same files unless the parent sequences them.
- **Premature fan-out:** Do not delegate before collecting enough context.
- **Nested explosion:** Default to leaf workers. Orchestrator children multiply cost fast.
- **Raw paste synthesis:** The parent must transform results into a final answer, not concatenate logs.
- **Durability confusion:** Delegation runs inside the current parent turn. It is not a persistent job queue.

## External Agent Handoff

When another agent wants Hermes to act as the harness, it should send a request using `scripts/hermes_harness.py` or the `templates/external_agent_prompt.md` format. The external agent should provide the task, workspace, constraints, allowed write scope, desired output, and maximum worker count.
