# Worker API Control Plane

The Worker API is the harness. OpenCode is only one possible manager client.
**Persistent worker identity is based on `worker_id` + `conversation` +
`session_key`.** Hermes `/v1/responses` named conversations are the default
**persistent reasoning lane** (the canonical worker memory). Hermes `/v1/runs`
is an **execution lane**, not the canonical memory layer, unless proven
otherwise. Native Hermes `/goal` is not available through tested API routes and
is not used. The harness goal loop is our own API-controlled continuation loop.

## Two lanes (read this first)

A persistent worker keeps its own context across turns and tasks -- *persistent
delegated cognition*, not a fresh subagent each time. That continuity lives in
the **persistent reasoning lane**, never in client-side transcript replay.

```text
Manager Agent
  -> Worker API
       -> Persistent Worker Registry        (_state/workers/*.json)
       -> PERSISTENT REASONING LANE          (canonical memory)
            default: POST /v1/responses, conversation=<worker.conversation>,
            X-Hermes-Session-Key=<worker.session_key>, store=true  (server-side)
       -> OPTIONAL EXECUTION LANE            (NOT memory)
            /v1/runs for approval-gated command/test execution; the result is
            SUMMARIZED BACK into the persistent conversation
       -> Artifact Manifest / Logs / State
```

- The goal loop sends **every turn to the same `conversation` + `session_key`**;
  continuity is **server-side**. Client-side `conversation_history` is a
  fallback/debug cache only -- never the canonical memory.
- `/v1/runs` is used only when `execution_required=True` (or a worker's
  `execution_transport`); its output is recorded back into the persistent
  conversation via `record_execution_result_to_persistent_worker`.

## Identity truth table

| ID             | Meaning                              | Used for                  |
|----------------|--------------------------------------|---------------------------|
| `worker_id`    | Worker API identity                  | manager routing           |
| `conversation` | Hermes /v1/responses named conversation | persistent worker context |
| `session_key`  | Hermes memory scope header (`X-Hermes-Session-Key`) | long-term memory |
| `session_id`   | Hermes SessionDB session             | `session_chat` only       |
| `run_id`       | Hermes run job id (`/v1/runs`)       | execution tracking        |
| `task_id`      | Worker API task id                   | logs / status             |

`conversation` is NOT `session_id`, and `session_id` is NOT a substitute for
`conversation` (rules enforced in `create_worker`).

## Worker identity example

```json
{
  "worker_id": "catcas-phase58",
  "role": "CAT_CAS Phase 5.8 worker",
  "persistent_transport": "responses",
  "conversation": "ccc:ags:phase58",
  "session_key": "agent:ags:phase58",
  "execution_transport": "runs",
  "workspace": "D:/CCC 2.0/AI/agent-governance-system",
  "read_roots": ["THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY"],
  "write_roots": ["THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/PHASE5_8_BARE_METAL_BOUNDARY"],
  "search_policy": "artifact_only",
  "branch_policy": "forbidden"
}
```

The Worker API is the real harness. Hermes is one backend runtime that executes
model turns inside named conversations. The control plane owns routing, scope,
goal loops, logs, and state. OpenCode (or any other manager) is just a client.

## Why native Hermes `/goal` is not used

Native Hermes `/goal` only dispatches through two paths:

- the CLI slash-command mixin (`hermes chat` -> `CLICommandsMixin`)
- the messaging gateway (`GatewaySlashCommandsMixin`: Telegram, Discord, ...)

The HTTP API does **not** intercept slash commands:

- `POST /v1/responses` treats `/goal ...` as plain input text.
- `POST /api/sessions/{id}/chat` (`_handle_session_chat`) routes `user_message`
  straight to `_run_agent()` with no command intercept.

So there is no tested API path that triggers real `/goal`. Rather than fake it,
this control plane implements the goal loop itself on top of the one persistent,
tested transport: `/v1/responses` with named conversations.

> The control plane never builds a `/goal` URL, never posts to a goal path, and
> never claims native `/goal` support. `test_no_goal_endpoint_in_source` and
> `test_no_native_goal_in_prompts` enforce this.

## Mental model

```text
Manager Agent (OpenCode, a script, a cron, ...)
  -> Worker API  (scripts/worker_api.py  /  WorkerController)
       -> Persistent Worker Registry   (_state/workers/*.json)
       -> Harness-managed goal loop     (control plane owns parse/stop/continue)
            -> Hermes /v1/responses, named conversation (one turn per call)
       -> Artifact manifests / logs / task records  (_state/)
```

- **Manager owns the roadmap.** It decides which worker gets which task.
- **Workers own domain-local context.** Each is a named Hermes conversation that
  persists across tasks and turns.
- **Worker API owns routing, scope, loops, logs, and state.**
- **Hermes owns model execution only.**

This is not "subagents". It is persistent delegated cognition: a worker is a
long-lived specialist, not a disposable fan-out child.

## Identity fields are distinct

| Field | Owner | Purpose |
|-------|-------|---------|
| `conversation` | `/v1/responses` | Turn chain for one persistent worker. The loop reuses it every turn. |
| `session_key` | `X-Hermes-Session-Key` | Long-term memory scope. Independent of the transcript. |
| `session_id` | Hermes SessionDB | Only used by the `session_chat` transport. **Not** stored on a worker record. |

A persistent worker record never carries a `session_id` — conflating the three
is the classic failure mode, so the registry keeps them separate by construction.

## Goal loop semantics

The harness runs its own API-controlled continuation loop (native `/goal` is not
dispatchable through the raw HTTP API). Every turn goes to the **persistent
reasoning lane** -- the same `conversation` + `session_key` -- so the worker
keeps server-side continuity.

1. Build a scope-locked task packet (STRICT SCOPE LOCK identical to the rest of
   the skill) + a goal-loop framing. In marker mode the agent is told to emit a
   marker; in judge mode it is told a separate judge decides (no marker).
2. Send the turn on the persistent lane (`conversation`, `session_key`,
   `store=true`). **Continuity is server-side**, not client-side replay.
3. Decide completion by `completion_mode`:
   - **`marker` (default):** the worker emits `GOAL_COMPLETE: true` (->complete)
     or `GOAL_BLOCKED: true` (->blocked). Cheap, deterministic.
   - **`judge`:** an external judge model returns `{done, reason}`. `done` ->
     complete; otherwise the reason is fed back as a continuation. If the judge
     is **unavailable**, the loop **fails fast** with status `error` -- it does
     NOT silently fail-open and burn the budget.
4. Optional layers when set:
   - **`verify_command`** — a deterministic gate the harness runs (exit 0 ==
     pass); a failing check is fed back. Use for objective goals.
   - **`judgment_mode="manager"`** — the loop PAUSES (`awaiting_judgment`) and
     hands the deliverable to the dispatcher, who calls `judge()` to accept
     (complete) or reject (resume with feedback). Expires after `judgment_timeout`.
5. Stop at `max_turns` -> **budget_exhausted**. Backend failure -> **error**.
6. Every turn / verdict / verification / scope audit is logged; a manifest is
   emitted on finish. `goal_loop=False` = single-shot, reports `complete`.

### The judge model (judge mode only)

When `completion_mode="judge"`, the judge is **`deepseek-v4-flash`** by default,
called *directly* at `https://api.deepseek.com` — independent of the worker model
(no shared-blind-spot self-certification). Key auto-discovered from Hermes' own
`.env`. Override via `HERMES_JUDGE_MODEL` / `HERMES_JUDGE_BASE_URL` /
`HERMES_JUDGE_API_KEY` or `--judge-model`. Marker mode never contacts a judge.

## Transports (two lanes)

| Lane | field | endpoint | role |
|------|-------|----------|------|
| Persistent reasoning (**canonical memory**) | `persistent_transport` = `responses` (default) | `POST /v1/responses` (named conversation) | every task turn; server-side memory |
| Persistent reasoning (alt) | `persistent_transport` = `session_chat` | `POST /api/sessions/{id}/chat` | needs `session_id`; server-side session memory |
| Execution (**NOT memory**) | `execution_transport` = `runs` (default), used only when `execution_required=True` | `POST /v1/runs` + events + approval | run approval-gated code/tests; result summarized back |

`/v1/runs` is **not** the canonical memory layer. The run API has no named
conversation, so a task run on the execution lane keeps an *execution-local*
client-side transcript only; on completion the harness calls
`record_execution_result_to_persistent_worker`, which writes a compact summary
into the worker's persistent `conversation` (+ `session_key`) so the canonical
memory retains what happened.

The execution lane (`hermes_run_transport.py`) starts an async run, streams
`GET /v1/runs/{run_id}/events`, and AUTO-ANSWERS every `approval.request` via
`POST /v1/runs/{run_id}/approval` with `choice="once"` (lets the agent run its
own code/tests while writing **nothing** persistent to your Hermes config).

> **Never use `choice="always"` or `"session"`.** `"always"` permanently writes
> the tool into the server's global `command_allowlist` (a cross-platform
> approval loosening). `"once"` is re-answered per call and leaves zero trace.
> The global `approvals.mode` is never touched, so the CLI / Telegram / etc.
> keep manual approval.

## External verification (trustworthy completion)

A self-reported `GOAL_COMPLETE` is not trustworthy — an agent can claim "tests
passed" without running them. Pass a `verify_command` and the harness runs the
real check itself; the agent's marker only *requests* verification.

```bash
python scripts/worker_control.py task-submit --worker-id w1 \
  --task "Implement X with tests, run them, iterate until green." \
  --verify-command "python -m pytest path/to/test_x.py -q" --max-turns 6
```

The manifest then carries `harness_verified` (`true`/`false`/`null`):
`null` = no check configured (self-declared); `true`/`false` = an external check
actually ran. Every verification attempt is recorded in `rec["verifications"]`.

> **Spaces in paths:** `verify_command` runs via `shell=True`. If your repo path
> contains a space (e.g. `D:\CCC 2.0\...`), use a **PATH-resolved** executable
> (`python -m pytest ...`, `pytest ...`) or **quote** any absolute path. An
> unquoted `D:\CCC 2.0\...\python.exe` is parsed as the command `D:\CCC` and
> fails. The `cwd` (workspace) may contain spaces safely.

## Scope locking

Every packet carries, via the shared SCOPE_LOCK block:

- explicit write scope, read scope, deny-write roots
- search policy (`artifact_only` | `dependency_only` | `repo_explicit`)
- branch policy (`forbidden` | `allowed`)
- no commits unless explicitly requested
- no future-goal proposals
- no unrelated-issue reporting
- no out-of-scope mutation

> **Prompt-level scope lock + POSTFLIGHT SCOPE AUDIT (not a hard firewall).**
> The scope lock above is prompt-level. The audit runs *after* the turn -- it
> observes and (optionally) reverts; it does NOT block writes before they happen.
> When the workspace is a git repo, the postflight audit (`git status
> --porcelain -uall`, diffed against a baseline captured at task start)
> cross-checks the agent's *reported* files against git. Because a real repo is
> constantly dirty from concurrent work, a global diff can't attribute changes,
> so the manifest separates them:
>
> - `agent_confirmed` — agent said it changed it **and** git agrees (real work)
> - `agent_missing` — agent **claimed** it but it's **not on disk** at all (**fabrication** — the "40 tests passed" lie)
> - `agent_unchanged` — claimed, **exists** on disk but unchanged this run (pre-existing / identical no-op rewrite; benign)
> - `agent_escapes` — confirmed files **outside `write_scope`** (the real breach)
> - `agent_clean` — did the agent itself stay in scope?
> - `unattributed_changes` — git-changed but the agent never claimed them: almost
>   always **concurrent external edits**; never auto-reverted, only surfaced.
>
> `auto_revert=True` (opt-in, off by default) reverts **only `agent_escapes`** —
> your parallel work is never touched. Assumes `workspace` is the git repo root.
> **Unattributed dirty files may remain and require manual review.** A true
> pre-write firewall (block before the write), a command-approval policy, and an
> isolated worktree/sandbox remain future work.

## Artifact manifest

Emitted on every completed task and written to `_state/manifests/{worker_id}.json`:

```json
{
  "worker_id": "...", "task_id": "...", "conversation": "...",
  "created_files": [], "modified_files": [],
  "read_scope": [], "write_scope": [],
  "verification": "", "harness_verified": true,
  "harness_verification": {"command": "...", "exit_code": 0, "passed": true, "output": "..."},
  "manager_judgment": {"verdict": "accept", "by": "manager", "...": "..."},
  "changed_files_source": "git_observed",
  "agent_confirmed": ["..."], "agent_missing": [], "agent_unchanged": [], "agent_escapes": [],
  "agent_clean": true, "unattributed_changes": ["...concurrent edits..."], "scope_reverted": [],
  "status": "complete", "next_recommended_task": "", "generated_at": "..."
}
```

Trust the **harness-observed** fields over self-reported ones:
- `harness_verified` (`true`/`false`/`null`) — did a `verify_command` actually run+pass?
- `manager_judgment` — the dispatcher's accept/reject verdict (if `judgment_mode="manager"`).
- `agent_confirmed` / `agent_missing` / `agent_unchanged` / `agent_escapes` /
  `agent_clean` / `unattributed_changes` — from the git audit (see the postflight
  scope-audit note above); `changed_files_source: "worker_reported"` only when the workspace
  isn't a git repo.

`created_files`/`modified_files`/`verification` are still self-reported (from the
agent's `ARTIFACT_MANIFEST` block). `next_recommended_task` (optional) must stay in scope.

## HTTP API

Stdlib `http.server`, no external framework. Auth is an optional bearer token via
`WORKER_API_KEY` (omit for localhost-only use).

| Method | Path | Action |
|--------|------|--------|
| GET | `/health` | Liveness; reports `native_goal: false` |
| GET | `/workers` | List workers |
| POST | `/workers` | Register a worker |
| GET | `/workers/{id}` | Worker record |
| GET | `/workers/{id}/state` | Status + last task summary |
| POST | `/workers/{id}/tasks` | Submit a task, run the goal loop |
| GET | `/workers/{id}/tasks/{task_id}` | Task record |
| POST | `/workers/{id}/continue` | Resume last task, same conversation |
| POST | `/workers/{id}/judge` | Manager verdict (`judgment_mode="manager"`): `{verdict, feedback}` |
| GET | `/tasks/{task_id}/log` | Structured JSONL log |

Start it:

```bash
python scripts/worker_api.py --host 127.0.0.1 --port 8770
# or
python scripts/worker_control.py serve --port 8770
```

## Usage

### From a manager (direct HTTP)

```bash
# 1. register a persistent specialist
curl -s localhost:8770/workers -d '{
  "worker_id": "catcas-auditor",
  "conversation": "ccc:ags:catcas-auditor",
  "session_key": "agent:ags:catcas",
  "workspace": "/repo",
  "write_roots": ["THOUGHT/LAB/CAT_CAS/phase5_9"],
  "read_roots":  ["THOUGHT/LAB/CAT_CAS/phase5_9"]
}'

# 2. run a bounded autonomous task loop
curl -s localhost:8770/workers/catcas-auditor/tasks -d '{
  "task": "Double check logic, engineering, and integrity. Harden results.",
  "acceptance_criteria": "All phase 5.9 artifacts verified; no open gaps.",
  "max_turns": 6
}'

# 3. resume if it hit the turn budget
curl -s localhost:8770/workers/catcas-auditor/continue -d '{"max_turns": 4}'
```

### From OpenCode (manager harness)

OpenCode stays a client. Instead of loading the `hermes-harness` skill and
delegating, the manager POSTs scoped task packets to the Worker API and reads
back structured results:

```text
OpenCode (manager, owns roadmap)
  -> POST /workers/{specialist}/tasks   (scoped packet, goal_loop=true)
  <- {status, turns, final_result, artifact_manifest}
  -> POST /workers/{specialist}/continue (if budget_exhausted)
```

The manager never needs Hermes internals; the API is the stable seam.

### From the CLI (no server)

```bash
python scripts/worker_control.py worker-create --worker-id catcas-auditor \
  --conversation ccc:ags:catcas-auditor --session-key agent:ags:catcas \
  --workspace /repo --write-roots THOUGHT/LAB/CAT_CAS/phase5_9 \
  --read-roots THOUGHT/LAB/CAT_CAS/phase5_9

python scripts/worker_control.py task-submit --worker-id catcas-auditor \
  --task "Harden phase 5.9 results." --max-turns 6 \
  --acceptance-criteria "All artifacts verified."

python scripts/worker_control.py continue --worker-id catcas-auditor --max-turns 4
python scripts/worker_control.py log --task-id <task_id>
```

No transport invokes real Hermes `/goal`. The goal loop is the control plane's,
not Hermes'. See the **Transports** section above for `runs` vs `responses`.

## Remaining risks / next hardening steps

- **No pre-write firewall.** The current scope control is a *postflight scope
  audit* (observe + optional revert after the turn) -- NOT a write firewall. A
  true pre-write block (deny the write before it lands) and a search-root limiter
  are still future work. Postflight `auto_revert` also requires a git workspace
  and only reverts agent-reported files.
- **Judge shares the worker's provider.** The judge is an independent *model*
  (deepseek-v4-flash vs deepseek-v4-pro) but same vendor; a different vendor would
  diversify blind spots further.
- **Synchronous loop.** `submit_task` blocks until the loop ends. For long runs,
  move to a background task table with polling, or Hermes cron.
- **Single-process state.** JSON files under `_state/` are fine for one host;
  concurrent writers would need locking or a real store.
- **Auth is a single shared bearer token.** No per-worker scoping or rotation.
