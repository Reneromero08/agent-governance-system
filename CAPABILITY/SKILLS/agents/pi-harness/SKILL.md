---
name: pi-harness
description: Run the local Pi coding agent as a persistent, headless worker with stable session IDs, background tasks, status polling, result collection, cancellation, and follow-up prompts in the same session. Use when an agent or local automation needs to delegate work to Pi, let it continue in the background, check progress later, or resume the same Pi conversation across multiple turns.
version: 0.3.0
status: Active
required_canon_version: ">=3.0.0"
---

# Pi Harness

Use Pi's headless JSON mode with an explicit `--session-id`. Keep worker state,
Pi sessions, logs, and task receipts under `LAW/CONTRACTS/_runs/pi-harness/`.

Pi workers load **no project context files, skills, prompt templates, or default
Pi system prompt automatically**. The parent must manually select and inject
every context block needed for each task. The harness always passes
`--no-context-files`, `--no-skills`, `--no-prompt-templates`, and an explicitly
empty `--system-prompt`. It transports the complete task packet through an
audited generated prompt file so Windows command shims cannot truncate
multiline instructions.

## Choose an entry path

- Use root `run.py` through `skill_run` to build and validate an offline task
  packet. It never launches Pi or spends model tokens.
- Use `scripts/worker_control.py` for live local workers.

Always run Python through the repository virtual environment.

## Create a worker

```powershell
.\.venv\Scripts\python.exe CAPABILITY\SKILLS\agents\pi-harness\scripts\worker_control.py worker-create `
  --worker-id reviewer `
  --workspace "D:\path\to\repo" `
  --read-root "src,tests" `
  --write-root "src,tests" `
  --allow-write
```

Worker identity and Pi conversation identity are separate:

- `worker_id` routes harness commands.
- `session_id` is the stable Pi session UUID. Omit it to derive a deterministic
  UUID from the worker ID and workspace.

Workers are read-only by default (`read,grep,find,ls`). `--allow-write`
explicitly enables `edit,write`; always provide narrow write roots. Shell access
is separate because shell commands cannot be path-confined mechanically:

```powershell
... worker-create --worker-id builder --write-root "src,tests" --allow-write --allow-shell `
  --shell-program git `
  --shell-program "python=.\.venv\Scripts\python.exe"
```

The governed shell overrides Pi's built-in `bash` tool. It accepts only a
program alias plus a literal argument array—never a shell command string—and
resolves every allowed program to an absolute native executable when the worker
is created. It confines `cwd` to the workspace, filters credentials from the
child environment, applies timeout/output caps, and records structured calls in
Pi JSONL. On Windows, `.cmd`, `.bat`, and PowerShell scripts are rejected.

Pi extensions are disabled by default because they execute code inside the
worker process. Add `--allow-extensions` only when the required extensions are
trusted.

Optional model selection:

```powershell
... worker-create --worker-id reviewer --provider openai --model gpt-5
```

## Submit and supervise work

Submit a background turn:

```powershell
... worker_control.py task-submit --worker-id reviewer --task "Review the auth flow and report concrete risks."
```

Select manual context independently for that task:

```powershell
... worker_control.py task-submit --worker-id reviewer `
  --task "Review the auth flow." `
  --context-file "src/auth/contract.md" `
  --context-file "tests/auth_cases.md" `
  --context-text "Focus on session fixation." `
  --context-token-budget 6000 `
  --context-tokenizer cl100k_base
```

Context files must be inside the worker's declared read roots. Sources are
packed deterministically in command order (files, then text), truncated to the
selected task budget using the selected tiktoken encoding, and recorded with
source hashes and included/original token counts. With no `--context-file` or
`--context-text`, the task receives no extra context and the token budget may be
zero. Every later `prompt` can choose a different context set and budget while
reusing the same Pi session. The selected budget applies only to manually
injected context; the task packet, Pi system prompt, tool schemas, and prior
session messages also consume the model's context window.

The command returns immediately with a deterministic `task_id`. Then use:

```powershell
... worker_control.py task-status --task-id reviewer-000001
... worker_control.py task-result --task-id reviewer-000001
... worker_control.py task-log --task-id reviewer-000001
```

Submit a new turn to the same Pi session after the prior task settles:

```powershell
... worker_control.py prompt --worker-id reviewer --message "Now fix the two high-severity findings."
```

`prompt` is an alias for another background task on the worker's unchanged
`session_id`. A worker accepts only one running task at a time.

Cancel a running task:

```powershell
... worker_control.py task-cancel --task-id reviewer-000001
```

## Task-packet contract

Every live prompt includes:

- the concrete goal;
- absolute workspace;
- read and write scope;
- tool policy;
- a branch/commit prohibition;
- a required final result containing findings, changed files, and verification.

Treat Pi as an untrusted external executable. Verify its claims and inspect the
repository diff before accepting work. The harness records process exit status
and the final assistant text; it does not declare the work correct.

## Operational constraints

- Never use `--no-session`; persistence is the point of this skill.
- Never assume Pi discovered repository instructions or governance. Manually
  include exactly what the task requires through task context options.
- Never reuse one worker ID for unrelated workspaces.
- Do not modify a worker while a task is running.
- Do not commit, push, merge, or release from Pi. Those actions remain subject
  to the parent session's explicit ceremonies.
- Use task logs for progress. Do not spawn visible terminal windows.
- When shell access is enabled, inspect `integrity.shell_scope_verifiable` and
  independently review the workspace diff; shell side effects cannot be
  confined by inspecting Pi's structured `edit`/`write` events.
- Set `PI_COMMAND` only when `pi` is not on `PATH`.

## Output shape

`task-status` and `task-result` return JSON. Important fields are `status`
(`queued`, `running`, `complete`, `failed`, or `cancelled`), `session_id`,
`pid`, `exit_code`, `result`, `integrity`, `receipt_path`, `stdout_path`, and
`stderr_path`. A task can be `complete` only when Pi exits successfully, emits
strict JSONL, reaches the current Pi `agent_end` event (or legacy
`agent_settled` event), returns a non-empty assistant result, and has no observed
out-of-scope `edit` or `write` call. Each task receipt hashes the prompt, stdout,
stderr, and final result.
