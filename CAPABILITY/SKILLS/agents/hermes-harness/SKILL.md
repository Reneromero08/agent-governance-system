---
name: hermes-harness
description: Use Hermes Agent as an orchestration harness that decomposes tasks, delegates focused work to subagents, and synthesizes verified results.
version: 0.1.0
status: Active
required_canon_version: ">=3.0.0"
platforms: [windows, macos, linux]
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

> **For autonomous goal loops, use the Worker API control plane** — persistent
> workers keyed by `worker_id` + `conversation` + `session_key`, with the
> persistent reasoning lane (`/v1/responses` named conversations, server-side
> memory) as canonical memory, an opt-in `/v1/runs` execution lane (summarized
> back), `marker`/`judge` completion, and a postflight scope audit. See
> **[WORKER_API.md](WORKER_API.md)** and Entry Mode 3 below. The `delegate_task`
> content below is the original loaded-skill path, still valid but not the
> recommended way to run goal loops.

## When to Use

Use this skill when the task is too wide for a single reasoning thread and can be split into independent subtasks, such as repo audits, parallel research, test triage, migration planning, feature decomposition, multi-file code review, documentation passes, benchmark review, or synthesis across several evidence streams.

Do not use this skill for trivial single-step work, questions that need one local answer, or tasks where the user explicitly forbids subagents.

## Entry Modes

There are three entry paths. **Use Mode 3 (Worker API) for autonomous goal
loops** — it is the current, validated path. Modes 1 and 2 are **LEGACY**: they
predate the Worker API and have no judge loop, no execution-with-approval, and
no postflight scope audit. They still work for their original purposes (one-shot
delegation / prompt generation) but should not be used to run goal loops.

Decision: autonomous goal loop / "keep working until done" → **Mode 3**.
Disposable one-shot fan-out inside Hermes → Mode 1. Just generate a prompt or a
single governed turn → Mode 2.

### 1. Loaded Skill (opencode `skill` tool) — LEGACY
The parent agent loads the skill instructions, then follows the procedure below using its own delegation tools (`task` in opencode, `delegate_task` in Hermes). The parent IS the harness. No goal-judge loop; the parent decides when to stop.

### 2. Governance Pipeline (`skill_run` / CLI) — LEGACY
Calls `run.py` which routes to the Hermes API server (default: `http://127.0.0.1:8643/v1`). Uses `/v1/responses` with named `conversation`. NOTE: `run.py` is intentionally **prompt-only** (it never calls the live agent — see run.py); it does not run a goal loop. For live autonomous runs use Mode 3.

Input JSON shape for skill_run:
```json
{
  "task": "string (required)",
  "mode": "audit|research|code|debug|docs|plan|synthesis|auto|persistent_worker|persistent_worker_verify",
  "workspace": "/absolute/path (optional, defaults to repo root)",
  "max_workers": 3,
  "toolsets": ["terminal", "file"],
  "constraints": "read-only, prefer evidence from commands",
  "timeout": 900,
  "conversation": "ags:catcas-auditor",
  "conversation_new": false,
  "session_key": "ccc:ags:main"
}
```

The Hermes API server must be running (check: `curl http://127.0.0.1:8643/v1/models`).
Uses `HERMES_API_KEY` or `API_SERVER_KEY` environment variable for authentication.

### 3. Worker API Control Plane (`worker_api.py` / `worker_control.py`) — RECOMMENDED

The two paths above make the *manager* the harness. The Worker API inverts this:
a small control plane becomes the harness, and any manager (OpenCode, a script,
a cron) is just a client. It owns a persistent worker registry, scoped task
packets, a harness-managed goal loop, artifact manifests, logs, and worker state.

Native Hermes `/goal` is **not** used: no HTTP path dispatches it. The goal loop
is owned by the control plane.

**Persistent worker identity** = `worker_id` + `conversation` + `session_key`.
By default a worker runs on the **persistent reasoning lane** -- Hermes
`/v1/responses` named conversations (`persistent_transport="responses"`),
server-side memory. Every goal-loop turn reuses the same `conversation` +
`session_key`; client-side transcript is NOT the canonical memory. The
**execution lane** (`/v1/runs`, used only when `execution_required=True`) lets
the agent run approval-gated code/tests, then its result is **summarized back**
into the persistent conversation -- runs are never the memory layer.

Completion: **`completion_mode="marker"` (default)** -- the worker emits
`GOAL_COMPLETE: true` / `GOAL_BLOCKED: true`. **`completion_mode="judge"`** uses
an external judge model (default deepseek-v4-flash); if the judge is unavailable
the loop **fails fast** (status `error`), it does not silently burn the budget.
Optional layers: `verify_command` (deterministic gate) and
`judgment_mode="manager"` (pause for the dispatcher's verdict).

```text
Manager -> Worker API -> Worker Registry -> Goal loop -> Hermes /v1/responses
                      -> Artifact manifests / logs / state
```

Full architecture, endpoints, identity-field semantics, and usage:
**[WORKER_API.md](WORKER_API.md)**. This is the recommended path for persistent
delegated cognition (long-lived specialists), as opposed to disposable subagents.

### Sessions (named conversations)

Uses Hermes `/v1/responses` with named `conversation`. Hermes manages conversation state server-side — no client-side history replay. The caller sends only the new task.

**Fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conversation` | string | (none) | Named conversation. Hermes auto-chains to latest stored response. Use deterministic names like `ccc:ags:catcas-auditor`. |
| `conversation_new` | bool | false | If true, requires `conversation` and creates a new unique conversation name by appending a UTC timestamp. Does not reuse or instruct-away the old conversation. |
| `session_key` | string | (none) | `X-Hermes-Session-Key` header for long-term memory scoping (e.g., `agent:ags:catcas`). Independent of conversation transcript. |

**Example workflow:**
```json
{"task": "We are building a TEP solver. The tape is 256 bytes...", "conversation": "ccc:ags:tep-solver"}

{"task": "Now implement the XOR forward pass.", "conversation": "ccc:ags:tep-solver"}

{"task": "Verify the SHA-256 restoration.", "conversation": "ccc:ags:tep-solver"}
```

**Without a conversation name** each call is stateless (fresh turn).

**Limitation:** Hermes stores up to 100 responses per named conversation (LRU eviction). Not infinite archival memory.

### Parent Prompt Construction Contract

The parent agent must convert vague follow-ups into scoped task packets before calling Hermes. Never send bare prompts like "Harden results" — always resolve them against the prior goal's artifact set.

**Scope resolution:**
| Vague phrase | Resolve to |
|-------------|-----------|
| `results`, `work`, `output` | files from the previous goal |
| `double check`, `verify` | read-only audit of artifact set, then in-scope fixes |
| `harden`, `fix`, `cleanup` | modify only the artifact set |
| `integrity`, `engineering` | quality of the artifact set, not the whole repo |

**Use `persistent_worker_verify` for follow-ups.** It injects STRICT SCOPE LOCK into the worker prompt. Always pass `--write-root`, `--read-root`, and `--search-policy artifact_only`.

> Prompt-level scope only. The Worker API adds a postflight scope audit; runtime enforcement (pre-write firewall, search limiter, auto-revert) is not yet implemented.

### Architecture: Three Layers

Use Hermes native mechanisms instead of repo-local JSON memory files:

```text
Layer 1: Named conversation (/v1/responses + conversation)
    Purpose: reusable multi-phase worker context
    Example: conversation="ccc:ags:catcas-auditor"

Layer 2: Session search (Hermes FTS5 session search)
    Purpose: recover old specific messages when needed
    Note: search is on-demand, no LLM calls required

Layer 3: MEMORY.md / USER.md
    Purpose: compact durable facts injected at session start
    Not for phase transcripts — keep under ~2K chars
```

### Decision Table

```text
Need isolated one-off research? → stateless call (no conversation)
Need phase 5 to remember phases 1-4? → named conversation
Need recall something from weeks ago? → session search
Need durable project facts every session? → MEMORY.md / USER.md
Need cheap mechanical processing? → scripts, not LLM session replay
```

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
