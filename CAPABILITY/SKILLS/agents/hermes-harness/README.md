# Hermes Harness Skill Folder

> **Current / recommended path:** the **Worker API control plane** — a faithful
> `/goal` replica over the HTTP API: persistent workers that execute their own
> code, an independent **deepseek-v4-flash** judge driving done/continue, an
> optional deterministic `verify_command` gate, and a git write-firewall. See
> **[WORKER_API.md](WORKER_API.md)**. The sections below document the original
> loaded-skill (`delegate_task`) and `/v1/responses` paths, which still work but
> are not the recommended way to run autonomous goal loops.

Drop-in repo folder that lets Hermes Agent act as a task harness for other agents.

> **The modes and transports in this section are LEGACY.** They predate the
> Worker API and have no goal-judge loop / execution-with-approval / write-firewall.
> For autonomous goal loops use the **[Worker API control plane](#worker-api-control-plane-recommended)**
> (next section). The content here remains accurate for its original uses
> (one-shot `delegate_task` fan-out, prompt generation, single governed turns).

It gives you three (legacy) operating modes and two transports:

**Transports (legacy):**
| Transport | Endpoint | Key | Dispatches /goal |
|-----------|----------|-----|-----------------|
| `responses` (default) | `POST /v1/responses` | `conversation` | No |
| `session_chat` | `POST /api/sessions/{id}/chat` | `session_id` | No — source confirmed |

Real Hermes `/goal` only dispatches through the CLI (`/goal` slash command in `hermes chat`) and messaging gateway (Telegram, Discord, etc.). The API server's `_handle_session_chat` at gateway/platforms/api_server.py:1560 routes `user_message` directly to `_run_agent()` without command intercept. The CLI path goes through `CLICommandsMixin` and the gateway path through `GatewaySlashCommandsMixin` — both contain `/goal` handlers. The API path does not.

**session_id, conversation, and session_key are different:**
- `session_id` — Hermes SessionDB session. Required for `/api/sessions/{id}/chat`.
- `conversation` — `/v1/responses` named conversation chain. Stateless persistence.
- `session_key` — `X-Hermes-Session-Key` header. Long-term memory scope.

**Modes (legacy — for goal loops use the Worker API instead):**

1. **Inside Hermes**: install `skills/hermes-harness/` as a Hermes skill and invoke `/hermes-harness`. Use `persistent_worker` mode with named conversations for multi-phase continuity, or use `delegate_task` for disposable isolated subagents.
2. **From another agent or script**: call `skills/hermes-harness/scripts/hermes_harness.py`. It sends a structured task request via `/v1/responses` to a local Hermes API server with named `conversation` support for persistent multi-turn context.
3. **Persistent workers**: Set `mode: persistent_worker` and a `conversation` name. The worker continues from prior context without spawning `delegate_task`. Use this for phase 1 → 2 → 3 → pause → 5 continuity.

This folder is intentionally conservative: named conversations for persistent workers, flat delegation for disposable subagents, explicit context packets, and final synthesis by the parent agent.

## Worker API control plane (recommended)

The modes above make the *manager* the harness. For persistent delegated
cognition, invert it: a small **Worker API is the harness**, and any manager
(OpenCode, a script, a cron) is just a client. See **[WORKER_API.md](WORKER_API.md)**
for the full architecture.

```text
Manager Agent -> Worker API -> Persistent Worker Registry
              -> Harness-managed goal loop -> Hermes /v1/responses (named conversation)
              -> Artifact manifests / logs / state
```

- `scripts/worker_control.py` — `WorkerController`: registry, goal loop, judge, manifests, logs, CLI.
- `scripts/worker_api.py` — stdlib HTTP API over the controller.
- `scripts/hermes_run_transport.py` — async-run transport (executes code) + the judge.
- The loop is a faithful **Hermes `/goal` replica** (native `/goal` isn't on the
  HTTP API): each turn an aux judge model (**deepseek-v4-flash** by default)
  returns done/continue; stop on done, `max_turns`, or a manager verdict.
  Optional `verify_command` (deterministic gate), `judgment_mode="manager"`, and
  a postflight git **write-firewall** (`auto_revert`). See WORKER_API.md.

```bash
python scripts/worker_api.py --port 8770          # start the API
python scripts/worker_control.py worker-create --worker-id catcas-auditor ...
python scripts/worker_control.py task-submit --worker-id catcas-auditor --task "..." --max-turns 6
```

## Design principles

- Use `persistent_worker` mode + named conversation for multi-phase continuity.
- Use `delegate_task` only for disposable, isolated subtasks.
- Subagents receive explicit context only. Never assume they know the parent conversation.
- Parent synthesis is mandatory. Never paste raw subagent outputs as the final answer.
- Durable long-running work should use Hermes cron or background terminal jobs, not synchronous delegation.
- `--model` is a cosmetic label — actual model selection is server-side in Hermes config.

## Scope-Locked Execution

Use `persistent_worker_verify` mode for follow-up tasks (verify, harden, audit, fix, double check, clean up). This mode injects a STRICT SCOPE LOCK into the worker prompt with explicit write/read/search scope boundaries.

> **Prompt-level only.** Runtime enforcement (write firewall, search limiter, postflight diff audit, auto-revert) is not yet implemented. Do not treat this as a hard filesystem sandbox. This mode reduces scope drift but does not replace runtime path validation.

```bash
python scripts/hermes_harness.py run \
  --task "Double check logic, engineering, and integrity. Harden results." \
  --mode persistent_worker_verify \
  --conversation "ccc:ags:phase58" \
  --write-root "THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY" \
  --read-root "THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY" \
  --search-policy artifact_only \
  --branch-policy forbidden
```

**CLI scope args:**
| Arg | Description |
|-----|-------------|
| `--write-root PATH,...` | Only these paths may be modified |
| `--read-root PATH,...` | Default search scope |
| `--deny-write-root PATH,...` | Explicit deny list (defaults: CAPABILITY/, TOOLS/, .git/, .hermes/) |
| `--search-policy` | `artifact_only` \| `dependency_only` \| `repo_explicit` |
| `--branch-policy` | `forbidden` \| `allowed` |

## Parent Prompt Construction Contract

The parent agent is responsible for converting user intent into a scoped task packet before calling Hermes Harness. **Never send vague follow-up prompts directly to a persistent worker.**

Bad:
```
Double check logic, engineering, and integrity. Harden results.
```

Good:
```
TASK: Double check logic, engineering, and integrity for the previous deliverable only.

WRITE_SCOPE: THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY
READ_SCOPE: THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY
SEARCH_POLICY: artifact_only
BRANCH_POLICY: forbidden

Treat "results", "work", "output", "logic", "engineering", "integrity",
"harden", "verify", "audit", "fix", and "clean up" as referring only
to the prior goal's artifact set. Do not inspect, audit, modify,
refactor, or improve unrelated repository files. Do not create branches.
Do not create future-goal proposals. If an outside issue is noticed but
does not block the in-scope task, ignore it.
```

### Scope Resolution Rules

The parent agent must resolve vague nouns before sending the task:

| User phrase | Parent must resolve as |
|-------------|----------------------|
| `results` | files created or modified by the previous goal |
| `work` | current named conversation artifact set |
| `output` | previous deliverable files |
| `double check` | read-only audit of the artifact set, then in-scope fixes only |
| `harden` | improve only the artifact set |
| `integrity` | integrity of the artifact set, not the whole repository |
| `engineering` | engineering quality of the artifact set, not repo infrastructure |
| `fix` | modify only files inside the declared write scope |
| `cleanup` | cleanup only inside declared write scope |

### Default Follow-Up Template

```text
TASK: {user_request}

TARGET: The previous deliverable in this named conversation only.

WRITE_SCOPE: {paths from previous goal}
READ_SCOPE: {same as WRITE_SCOPE, plus explicit dependency files if known}
SEARCH_POLICY: artifact_only
BRANCH_POLICY: forbidden

FORBIDDEN:
- Do not modify files outside WRITE_SCOPE.
- Do not audit the whole repository.
- Do not refactor shared infrastructure.
- Do not touch CAPABILITY/, TOOLS/, GOVERNANCE/, .git/, .hermes/, or
  unrelated project roots unless those paths are explicitly in WRITE_SCOPE.
- Do not create branches.
- Do not create future-goal proposals.
- Do not report unrelated issues.

OUT_OF_SCOPE_POLICY: Ignore unrelated out-of-scope issues. If an external
file directly blocks the in-scope task, report BLOCKED_EXTERNAL_DEPENDENCY
only. Do not edit it.

POSTFLIGHT: After edits, run a changed-file audit. If any changed path is
outside WRITE_SCOPE, revert it automatically and report SCOPE_ESCAPE_REVERTED.
```

### Artifact Manifest

At the end of every goal, produce an artifact manifest so follow-ups can deterministically resolve scope:

```json
{
  "goal_id": "phase58",
  "conversation": "ccc:ags:phase58",
  "created_files": ["THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY/..."],
  "modified_files": ["THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/SSH_ROADMAP.md"],
  "write_scope": ["THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY"],
  "read_scope": ["THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY"],
  "forbidden_roots": ["CAPABILITY", "TOOLS", ".git", ".hermes"]
}
```

The next follow-up must use this manifest to determine scope.

### Core Invariant

> The subagent may only spend tokens and mutate files inside the declared task surface. External context is allowed only when required to complete the declared task. External mutation is never allowed. External suggestions are noise unless they block the declared task.

## Usage examples

```bash
# Persistent worker — multi-phase audit with named conversation
python scripts/hermes_harness.py run \
  --task "Continue phase 5 of the CAT_CAS audit." \
  --mode persistent_worker \
  --conversation "ccc:ags:catcas-auditor"

# New conversation from scratch (appends timestamp)
python scripts/hermes_harness.py run \
  --task "Start a fresh audit of the roadmap." \
  --mode persistent_worker \
  --conversation "ccc:ags:new-audit" \
  --conversation-new

# Stateless disposable task — no conversation, no persistence
python scripts/hermes_harness.py run \
  --task "What files changed in src/ today?" \
  --mode audit

# Dry-run — print the prompt without calling the API
python scripts/hermes_harness.py run \
  --task "Audit the repo" \
  --mode audit \
  --dry-run

# Session chat transport — turn on existing Hermes session
python scripts/hermes_harness.py run \
  --transport session_chat \
  --session-id "a7876494-2178-4ab4-942f-2f77e9f4344e" \
  --task "Double check logic, engineering, and integrity. Harden results." \
  --mode persistent_worker_verify \
  --write-root "THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/phase5_9" \
  --read-root "THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/phase5_9" \
  --search-policy artifact_only \
  --branch-policy forbidden

# Prompt-only — generate the harness prompt for manual review
python scripts/hermes_harness.py prompt \
  --task-file examples/task.audit.json
```

## Folder map

```text
skills/hermes-harness/
├── SKILL.md
├── README.md
├── config/
│   └── hermes-harness.yaml
├── examples/
│   ├── task.audit.json
│   ├── task.research.json
│   └── commands.md
├── scripts/
│   ├── hermes_harness.py
│   └── hermes_task.sh
├── templates/
│   ├── delegation_brief.md
│   ├── external_agent_prompt.md
│   ├── synthesis_report.md
│   └── task_matrix.yaml
└── tests/
    └── test_contracts.py
```

## References

- Hermes skills use `SKILL.md`, references, templates, scripts, and assets, and can be loaded as slash commands.
- Hermes `delegate_task` creates isolated child agents whose only inherited context is what the parent sends in `goal` and `context`.
- Hermes API server exposes `/v1/responses` (stateful, named conversations) and `/v1/chat/completions` (stateless). This skill uses `/v1/responses` with `conversation` for persistent worker context.

## Runtime Enforcement (TODO)

The harness currently provides prompt-level scope locking only. Hard runtime containment requires:

- **Write firewall:** block any file write outside `--write-root` before it executes.
- **Search root limiter:** deny grep/glob/search outside `--read-root` unless `--search-policy` allows it.
- **Branch command deny:** block `git branch`, `git checkout -b`, etc. when `--branch-policy forbidden`.
- **Postflight diff audit:** run `git diff --name-only` after the task completes.
- **Auto-revert out-of-scope changes:** revert any changed file outside `WRITE_SCOPE` and report `SCOPE_ESCAPE_REVERTED`.
- **Fail-on-escape:** if unauthorized changes occurred and cannot be auto-reverted, fail the run.
