#!/usr/bin/env python3
"""Worker API control plane for the Hermes Harness.

This module is the real harness. Hermes is only one backend runtime that
executes model turns inside named conversations. The control plane owns:

    - the persistent worker registry
    - scoped task packets
    - a harness-managed goal loop (NOT native Hermes /goal)
    - artifact manifests
    - structured per-task logs
    - worker state and completion/block markers

Why not native Hermes /goal?
    Native Hermes /goal only dispatches through the CLI slash-command mixin
    and the messaging gateway (Telegram, Discord, ...). The API server routes
    /v1/responses and /api/sessions/{id}/chat straight to a normal agent turn
    with no slash-command intercept. So /goal cannot be driven through the API.
    This control plane therefore implements the goal loop itself, on top of the
    one transport that is real and tested: /v1/responses with named
    conversations. See README.md for the full rationale.

Storage:
    Worker records, task records, and JSONL logs live under a state directory
    (default: <skill>/_state, override with HERMES_WORKER_STATE_DIR). This is
    the control plane's own data. It never touches Hermes state.db and never
    treats Hermes runtime databases as durable project memory.

The controller takes an injectable `caller` so tests can drive the goal loop
deterministically without a live Hermes server. The default caller wraps
hermes_harness.call_hermes_responses (the /v1/responses transport).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from hermes_harness import (  # noqa: E402
    DEFAULT_BASE,
    DEFAULT_KEY,
    DEFAULT_MODEL,
    build_harness_prompt,
    call_hermes_responses,
)
from hermes_run_transport import call_hermes_run, call_hermes_judge  # noqa: E402

SKILL_ROOT = _SCRIPTS_DIR.parent

DEFAULT_DONE_MARKER = "GOAL_COMPLETE: true"
DEFAULT_BLOCKED_MARKER = "GOAL_BLOCKED: true"
DEFAULT_MAX_TURNS = 6

VALID_BACKENDS = {"hermes_responses"}
VALID_STATUSES = {"idle", "running", "complete", "blocked", "budget_exhausted", "error",
                  "awaiting_judgment", "judgment_expired"}

# Status values a goal loop run can terminate with.
TERMINAL_RUN_STATUSES = {"complete", "blocked", "budget_exhausted", "error"}

_MANIFEST_RE = re.compile(
    r"ARTIFACT_MANIFEST\s*[:=]?\s*```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Caller abstraction (the only place that talks to a backend runtime)
# ---------------------------------------------------------------------------

# A caller is: (prompt, conversation, session_key, model) -> {text, response_id, usage}
Caller = Callable[..., Dict[str, Any]]
# A judge is: (goal, response) -> {done: bool, reason: str, blocked?: bool}
Judge = Callable[..., Dict[str, Any]]


# Independent judge endpoint. Defaults to DeepSeek's fast model. The Hermes API
# server exposes only its single main model ("hermes-win" = deepseek-v4-pro), so
# selecting a *different*/cheaper judge model requires calling the provider
# directly. Override via env (or pass through default_judge args):
#   HERMES_JUDGE_MODEL     (default: deepseek-flash)
#   HERMES_JUDGE_BASE_URL  (default: https://api.deepseek.com)
#   HERMES_JUDGE_API_KEY   (or DEEPSEEK_API_KEY)
JUDGE_DEFAULT_MODEL = "deepseek-v4-flash"
JUDGE_DEFAULT_BASE = "https://api.deepseek.com"


def _resolve_hermes_secret(name: str) -> str:
    """Find a secret (e.g. DEEPSEEK_API_KEY) from env or Hermes' .env files.

    Hermes stores provider keys in its own .env (under %LOCALAPPDATA%/hermes or
    ~/.hermes), not the process env. This lets the judge reuse the deepseek key
    Hermes already holds, with no duplication.
    """
    v = os.environ.get(name)
    if v:
        return v
    candidates = [Path.home() / ".hermes" / ".env"]
    la = os.environ.get("LOCALAPPDATA")
    if la:
        candidates.append(Path(la) / "hermes" / ".env")
    for f in candidates:
        try:
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, val = line.split("=", 1)
                if k.strip() == name:
                    return val.strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def default_judge(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = 60,
) -> Judge:
    """Default goal judge: an INDEPENDENT model returning {done, reason}.

    This is the Hermes /goal mechanism -- a separate model decides done/continue
    from the agent's last response, not the worker (no self-certification) and
    not a human. Defaults to DeepSeek flash (a cheap, independent judge, distinct
    from the deepseek-v4-pro worker). Fail-open: judge error -> continue.
    """
    mdl = model or os.environ.get("HERMES_JUDGE_MODEL", JUDGE_DEFAULT_MODEL)
    base = base_url or os.environ.get("HERMES_JUDGE_BASE_URL", JUDGE_DEFAULT_BASE)
    key = (api_key or os.environ.get("HERMES_JUDGE_API_KEY")
           or _resolve_hermes_secret("DEEPSEEK_API_KEY"))

    def _judge(goal: str, response: str) -> Dict[str, Any]:
        v = call_hermes_judge(goal, response, model=mdl, base_url=base,
                              api_key=key, timeout=timeout)
        return {"done": v["done"], "reason": v["reason"], "blocked": False}
    return _judge


def default_caller(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Caller:
    """Build the default backend caller: Hermes /v1/responses transport.

    This never calls /goal and never uses session_chat. It chains turns by
    reusing the worker's named conversation, which is the only persistent,
    tested Hermes API path.
    """
    base = base_url or os.environ.get("HERMES_API_BASE", DEFAULT_BASE)
    key = api_key or os.environ.get("HERMES_API_KEY") or os.environ.get("API_SERVER_KEY") or DEFAULT_KEY

    def _call(prompt: str, conversation: str = "", session_key: str = "",
              model: str = DEFAULT_MODEL, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        return call_hermes_responses(
            prompt=prompt,
            conversation=conversation,
            session_key=session_key,
            store=bool(conversation),
            base_url=base,
            api_key=key,
            model=model or DEFAULT_MODEL,
            timeout=timeout,
        )

    return _call


def default_run_caller(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = 600,
    approve_choice: str = "once",
) -> Caller:
    """Backend caller using the async run API with harness-side approval.

    Unlike the /v1/responses caller, this answers `approval.request` events so
    the agent can execute its own code/tests autonomously. Context is threaded
    client-side via `conversation_history` (the run API has no named
    conversation). Raises on a run.failed error so the goal loop records it.
    """
    base = base_url or os.environ.get("HERMES_API_BASE", DEFAULT_BASE)
    key = api_key or os.environ.get("HERMES_API_KEY") or os.environ.get("API_SERVER_KEY") or DEFAULT_KEY

    def _call(prompt: str, conversation: str = "", session_key: str = "",
              model: str = DEFAULT_MODEL, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        res = call_hermes_run(
            prompt,
            session_key=session_key,
            conversation_history=history or None,
            approve_choice=approve_choice,
            base_url=base, api_key=key, timeout=timeout,
        )
        if res.get("error"):
            raise RuntimeError(res["error"])
        return {
            "text": res.get("text", ""),
            "response_id": res.get("run_id", ""),
            "usage": res.get("usage", {}),
            "approvals": res.get("approvals", 0),
        }

    return _call


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seconds_since(ts: str) -> float:
    """Seconds elapsed since an _utc_now() timestamp (inf if unparseable)."""
    try:
        then = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - then).total_seconds()
    except (ValueError, TypeError):
        return float("inf")


def _new_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


# ---------------------------------------------------------------------------
# Goal-loop prompt framing
# ---------------------------------------------------------------------------

def build_goal_loop_contract(
    target: str,
    acceptance_criteria: str,
    done_marker: str,
    blocked_marker: str,
) -> str:
    """The marker / acceptance contract appended to a scope-locked packet."""
    ac = acceptance_criteria.strip() or "The declared task is fully and verifiably done."
    return (
        "GOAL LOOP CONTRACT:\n"
        f"TARGET: {target.strip() or 'The declared task and its artifact scope only.'}\n"
        f"ACCEPTANCE CRITERIA: {ac}\n"
        "\n"
        "This is a harness-managed goal loop owned by the control plane.\n"
        "Do the in-scope work this turn, then report status with an explicit marker.\n"
        "\n"
        f"When the acceptance criteria are fully met, emit on its own line:\n"
        f"  {done_marker}\n"
        f"If you cannot proceed because of a blocker, emit on its own line:\n"
        f"  {blocked_marker}\n"
        "  reason: <one line>\n"
        "\n"
        f"Do not emit '{done_marker}' unless the acceptance criteria are met.\n"
        "If more in-scope work remains, do not emit either marker; the harness\n"
        "will send a continuation turn in this same conversation.\n"
        "\n"
        "End every turn with a machine-readable manifest of in-scope changes:\n"
        "ARTIFACT_MANIFEST:\n"
        "```json\n"
        '{\"created_files\": [], \"modified_files\": [], \"verification\": \"\"}\n'
        "```\n"
    )


def build_judge_loop_contract(target: str, acceptance_criteria: str) -> str:
    """Goal-loop framing for judge mode: no markers (an aux judge decides done)."""
    ac = acceptance_criteria.strip() or "The declared task is fully and verifiably done."
    return (
        "AUTONOMOUS GOAL LOOP:\n"
        f"TARGET: {target.strip() or 'The declared task and its artifact scope only.'}\n"
        f"ACCEPTANCE CRITERIA: {ac}\n"
        "\n"
        "Keep working toward the goal across turns. You do NOT need to declare\n"
        "completion -- a separate judge decides when the goal is met from your\n"
        "response, and the harness will send a continuation turn if more work\n"
        "remains. If the goal is genuinely impossible, say so plainly with the\n"
        "reason. Do the in-scope work this turn, then end with a machine-readable\n"
        "manifest of in-scope changes:\n"
        "ARTIFACT_MANIFEST:\n"
        "```json\n"
        '{\"created_files\": [], \"modified_files\": [], \"verification\": \"\"}\n'
        "```\n"
    )


def build_task_packet(
    worker: Dict[str, Any],
    task: str,
    *,
    mode: str = "persistent_worker_verify",
    constraints: str = "",
    output_contract: str = "",
    acceptance_criteria: str = "",
    target: str = "",
    goal_loop: bool = True,
    use_judge: bool = True,
    done_marker: str = DEFAULT_DONE_MARKER,
    blocked_marker: str = DEFAULT_BLOCKED_MARKER,
    write_roots: Optional[List[str]] = None,
    read_roots: Optional[List[str]] = None,
    deny_roots: Optional[List[str]] = None,
    search_policy: Optional[str] = None,
    branch_policy: Optional[str] = None,
) -> str:
    """Compose the first-turn scope-locked packet for a worker.

    Reuses hermes_harness.build_harness_prompt so the STRICT SCOPE LOCK block
    (write/read/deny scope, search policy, branch policy, no-commit,
    no-future-goals, no-unrelated-issues, no out-of-scope mutation) is identical
    to the rest of the skill. Then appends the goal-loop marker contract.
    """
    base = build_harness_prompt(
        task=task,
        workspace=worker.get("workspace", "."),
        mode=mode,
        max_workers=0,  # persistent worker: never fan out
        constraints=constraints,
        output=output_contract,
        write_roots=write_roots if write_roots is not None else worker.get("write_roots"),
        read_roots=read_roots if read_roots is not None else worker.get("read_roots"),
        deny_roots=deny_roots if deny_roots is not None else worker.get("deny_write_roots"),
        search_policy=search_policy or worker.get("search_policy", "artifact_only"),
        branch_policy=branch_policy or worker.get("branch_policy", "forbidden"),
    )
    if not goal_loop:
        return base
    if use_judge:
        contract = build_judge_loop_contract(target, acceptance_criteria)
    else:
        contract = build_goal_loop_contract(target, acceptance_criteria, done_marker, blocked_marker)
    return f"{base}\n\n{contract}"


def build_continuation_packet(
    done_marker: str = DEFAULT_DONE_MARKER,
    blocked_marker: str = DEFAULT_BLOCKED_MARKER,
    turn: int = 2,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> str:
    """Compact continuation turn. Scope lock is preserved by the conversation."""
    return (
        f"CONTINUE (harness goal loop turn {turn} of {max_turns}).\n"
        "Same STRICT SCOPE LOCK and GOAL LOOP CONTRACT from this conversation "
        "still apply. Do not widen scope. Do not commit. Do not create branches.\n"
        "Continue the in-scope work from where you stopped.\n"
        f"Emit '{done_marker}' when acceptance criteria are met, or "
        f"'{blocked_marker}' with a reason if blocked.\n"
        "End with the ARTIFACT_MANIFEST json block.\n"
    )


# ---------------------------------------------------------------------------
# Marker / manifest parsing
# ---------------------------------------------------------------------------

def detect_marker(text: str, done_marker: str, blocked_marker: str) -> Optional[str]:
    """Return 'complete', 'blocked', or None based on markers in worker output."""
    if not text:
        return None
    if done_marker and done_marker in text:
        return "complete"
    if blocked_marker and blocked_marker in text:
        return "blocked"
    return None


def parse_reported_manifest(text: str) -> Dict[str, Any]:
    """Best-effort parse of a worker-emitted ARTIFACT_MANIFEST json block."""
    if not text:
        return {}
    m = _MANIFEST_RE.search(text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def run_verification(command: str, cwd: Optional[str] = None, timeout: int = 120) -> Dict[str, Any]:
    """Run an EXTERNAL acceptance check, owned by the harness (not the agent).

    This is the source of truth for completion: the agent's GOAL_COMPLETE marker
    only requests verification; this decides it. Returns exit code 0 == passed.
    The output tail is fed back to the agent on failure so it can fix and retry.

    NOTE: runs via `shell=True`. Any path with a SPACE must be quoted, or use a
    PATH-resolved executable. Prefer `python -m pytest ...` / `pytest ...` over an
    absolute interpreter path -- e.g. an unquoted `D:\\CCC 2.0\\...\\python.exe`
    breaks ('D:\\CCC' is not recognized). The workspace `cwd` may contain spaces
    safely (it is not shell-parsed).
    """
    import subprocess
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd or None,
            capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout or "")
        if proc.stderr:
            out += ("\n[stderr]\n" + proc.stderr)
        return {"passed": proc.returncode == 0, "exit_code": proc.returncode,
                "output": out[-4000:], "command": command}
    except subprocess.TimeoutExpired:
        return {"passed": False, "exit_code": -1,
                "output": f"verification timed out after {timeout}s", "command": command}
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "exit_code": -1,
                "output": f"verification could not run: {exc}", "command": command}


# ---------------------------------------------------------------------------
# Write firewall: postflight git-diff scope audit (harness-observed changes)
# ---------------------------------------------------------------------------

def git_status_porcelain(workspace: str) -> Optional[Dict[str, str]]:
    """Return {repo-relative path: status} from `git status --porcelain`.

    None if `workspace` is not a git repo (or git is unavailable) -- the audit
    then degrades gracefully to worker-reported manifests.
    """
    import subprocess
    try:
        r = subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain", "-uall"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
    except Exception:  # noqa: BLE001
        return None
    out: Dict[str, str] = {}
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line[:2], line[3:]
        if " -> " in path:  # rename: take the destination
            path = path.split(" -> ", 1)[1]
        out[path.strip().strip('"').replace("\\", "/")] = status
    return out


def _git_oneline(workspace: str, *args: str) -> Optional[str]:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(workspace), *args],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _norm_rel(path: str, workspace: str, prefix: str = "", root: Optional[str] = None) -> str:
    """Normalize a path to REPO-ROOT-relative (matching `git status` output).

    `prefix` is the workspace's location within the repo (`git rev-parse
    --show-prefix`), `root` the repo toplevel. Handles workspaces that are a
    subdirectory of the repo, not just the repo root.
    """
    p = Path(path)
    if p.is_absolute():
        base = Path(root) if root else Path(workspace)
        try:
            return str(p.resolve().relative_to(base.resolve())).replace("\\", "/")
        except (ValueError, OSError):
            return str(path).replace("\\", "/")
    rel = str(path).replace("\\", "/").lstrip("./")
    return f"{prefix}{rel}" if prefix else rel


def git_revert_path(workspace: str, path: str, status: str) -> bool:
    """Revert one out-of-scope change: delete if untracked, else `git checkout`."""
    import subprocess
    try:
        if status.startswith("??"):
            target = Path(workspace) / path
            if target.exists():
                target.unlink()
        else:
            subprocess.run(["git", "-C", str(workspace), "checkout", "--", path],
                           capture_output=True, text=True, timeout=30)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class WorkerController:
    """Owns the persistent worker registry, goal loops, manifests, and logs."""

    def __init__(
        self,
        state_dir: Optional[str | Path] = None,
        caller: Optional[Caller] = None,
        judge: Optional[Judge] = None,
    ) -> None:
        sd = state_dir or os.environ.get("HERMES_WORKER_STATE_DIR") or (SKILL_ROOT / "_state")
        self.state_dir = Path(sd).expanduser().resolve()
        self.workers_dir = self.state_dir / "workers"
        self.tasks_dir = self.state_dir / "tasks"
        self.logs_dir = self.state_dir / "logs"
        self.manifests_dir = self.state_dir / "manifests"
        for d in (self.workers_dir, self.tasks_dir, self.logs_dir, self.manifests_dir):
            d.mkdir(parents=True, exist_ok=True)
        # Injectable for tests. If injected, it is used for every worker
        # regardless of transport. In production, the caller is chosen per
        # worker by its `transport` field (see _caller_for):
        #   "runs"      -> async run API + harness-side approval (can execute)
        #   "responses" -> /v1/responses named conversation (no autonomous exec)
        self._injected_caller = caller
        self._caller = caller or default_run_caller()  # back-comat default
        self._resp_caller: Optional[Caller] = None
        self._run_caller: Optional[Caller] = None
        # Goal judge (Hermes /goal-style). Injectable for tests; lazily built
        # in production so a judge model is only contacted when actually used.
        self._injected_judge = judge
        self._judge: Optional[Judge] = judge

    def _judge_for(self, worker: Dict[str, Any]) -> Judge:
        if self._judge is None:
            # Independent judge (DeepSeek flash by default), NOT the worker model.
            self._judge = default_judge()
        return self._judge

    def _caller_for(self, worker: Dict[str, Any]) -> Caller:
        if self._injected_caller is not None:
            return self._injected_caller
        transport = worker.get("transport", "runs")
        if transport == "responses":
            if self._resp_caller is None:
                self._resp_caller = default_caller()
            return self._resp_caller
        if self._run_caller is None:
            self._run_caller = default_run_caller()
        return self._run_caller

    # -- registry ----------------------------------------------------------

    def _worker_path(self, worker_id: str) -> Path:
        return self.workers_dir / f"{worker_id}.json"

    @contextlib.contextmanager
    def _worker_lock(self, worker_id: str, stale_seconds: int = 7200):
        """Atomic per-worker advisory lock (closes the concurrent-submit race).

        Uses O_CREAT|O_EXCL (atomic on the filesystem) so only one caller can hold
        a worker at a time, even across threads/processes. A lock left behind by a
        crash is stolen once older than `stale_seconds`.
        """
        lockp = self.workers_dir / f"{worker_id}.lock"
        acquired = False
        try:
            try:
                fd = os.open(str(lockp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, _utc_now().encode("utf-8"))
                os.close(fd)
                acquired = True
            except FileExistsError:
                try:
                    age = _seconds_since(lockp.read_text(encoding="utf-8").strip())
                except OSError:
                    age = 0.0
                if age <= stale_seconds:
                    raise ValueError(f"Worker {worker_id!r} is busy (locked).")
                # Stale lock from a crash -> steal it.
                with contextlib.suppress(OSError):
                    lockp.unlink()
                fd = os.open(str(lockp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, _utc_now().encode("utf-8"))
                os.close(fd)
                acquired = True
            yield
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    lockp.unlink()

    def list_workers(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in sorted(self.workers_dir.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        p = self._worker_path(worker_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def save_worker(self, worker: Dict[str, Any]) -> Dict[str, Any]:
        worker["updated_at"] = _utc_now()
        self._worker_path(worker["worker_id"]).write_text(
            json.dumps(worker, indent=2), encoding="utf-8"
        )
        return worker

    def create_worker(
        self,
        worker_id: str,
        role: str = "specialist",
        conversation: str = "",
        session_key: str = "",
        model: str = DEFAULT_MODEL,
        workspace: str = ".",
        backend: str = "hermes_responses",
        read_roots: Optional[List[str]] = None,
        write_roots: Optional[List[str]] = None,
        deny_write_roots: Optional[List[str]] = None,
        search_policy: str = "artifact_only",
        branch_policy: str = "forbidden",
        transport: str = "runs",
    ) -> Dict[str, Any]:
        if not worker_id or not str(worker_id).strip():
            raise ValueError("worker_id is required")
        if backend not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend {backend!r}. Valid: {sorted(VALID_BACKENDS)}")
        if transport not in ("runs", "responses"):
            raise ValueError(f"Invalid transport {transport!r}. Valid: runs, responses")
        if self.get_worker(worker_id):
            raise ValueError(f"Worker {worker_id!r} already exists")
        # conversation, session_key, and session_id are three DISTINCT things.
        # The control plane uses `conversation` for the /v1/responses turn chain
        # and `session_key` for long-term memory scope. session_id (a Hermes
        # SessionDB id) is intentionally NOT part of a persistent worker record:
        # the responses transport does not use it.
        worker = {
            "worker_id": worker_id,
            "role": role,
            "backend": backend,
            "conversation": conversation or f"ccc:ags:{worker_id}",
            "session_key": session_key or f"agent:ags:{worker_id}",
            "model": model or DEFAULT_MODEL,
            "workspace": str(workspace or "."),
            "read_roots": _as_list(read_roots),
            "write_roots": _as_list(write_roots),
            "deny_write_roots": _as_list(deny_write_roots) or ["CAPABILITY/", "TOOLS/", ".git/", ".hermes/"],
            "search_policy": search_policy,
            "branch_policy": branch_policy,
            "transport": transport,
            # Client-side rolling context for the "runs" transport (the async
            # run API has no server-side named conversation). Trimmed in
            # _finalize. Unused by the "responses" transport.
            "history": [],
            "status": "idle",
            "last_task_id": None,
            "artifact_manifest": str((self.manifests_dir / f"{worker_id}.json")),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        return self.save_worker(worker)

    def get_state(self, worker_id: str) -> Dict[str, Any]:
        worker = self.get_worker(worker_id)
        if not worker:
            raise KeyError(f"Unknown worker {worker_id!r}")
        if self._expire_if_stale(worker):
            worker = self.get_worker(worker_id)  # reload after expiry
        last_task = None
        if worker.get("last_task_id"):
            last_task = self.get_task(worker["last_task_id"], summary=True)
        return {
            "worker_id": worker_id,
            "status": worker.get("status"),
            "conversation": worker.get("conversation"),
            "session_key": worker.get("session_key"),
            "last_task_id": worker.get("last_task_id"),
            "last_task": last_task,
            "artifact_manifest": worker.get("artifact_manifest"),
        }

    # -- logs --------------------------------------------------------------

    def _log_path(self, task_id: str) -> Path:
        return self.logs_dir / f"{task_id}.jsonl"

    def _log(self, task_id: str, event: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": _utc_now(), "event": event, **payload}
        with self._log_path(task_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def get_log(self, task_id: str) -> List[Dict[str, Any]]:
        p = self._log_path(task_id)
        if not p.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # -- tasks -------------------------------------------------------------

    def _task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def get_task(self, task_id: str, summary: bool = False) -> Optional[Dict[str, Any]]:
        p = self._task_path(task_id)
        if not p.exists():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        if summary:
            return {
                "task_id": rec.get("task_id"),
                "worker_id": rec.get("worker_id"),
                "status": rec.get("status"),
                "turns": len(rec.get("turns", [])),
                "completed_at": rec.get("completed_at"),
            }
        return rec

    def _save_task(self, rec: Dict[str, Any]) -> None:
        self._task_path(rec["task_id"]).write_text(json.dumps(rec, indent=2), encoding="utf-8")

    def submit_task(
        self,
        worker_id: str,
        task: str,
        *,
        mode: str = "persistent_worker_verify",
        goal_loop: bool = True,
        max_turns: int = DEFAULT_MAX_TURNS,
        constraints: str = "",
        output_contract: str = "",
        acceptance_criteria: str = "",
        target: str = "",
        done_marker: str = DEFAULT_DONE_MARKER,
        blocked_marker: str = DEFAULT_BLOCKED_MARKER,
        write_roots: Optional[List[str]] = None,
        read_roots: Optional[List[str]] = None,
        deny_roots: Optional[List[str]] = None,
        search_policy: Optional[str] = None,
        branch_policy: Optional[str] = None,
        verify_command: str = "",
        verify_timeout: int = 120,
        verify_cwd: str = "",
        judgment_mode: str = "auto",
        use_judge: bool = True,
        judgment_timeout: int = 3600,
        auto_revert: bool = False,
    ) -> Dict[str, Any]:
        """Submit a task to a persistent worker and run the harness goal loop.

        Completion (default) works exactly like Hermes /goal: after each turn an
        auxiliary judge model returns done/continue from the agent's response.
        `use_judge=False` falls back to the legacy marker (`GOAL_COMPLETE`)
        check. `verify_command` (deterministic gate) and `judgment_mode="manager"`
        still layer on top when set.

        Completion authority, in order:
        - `verify_command` (if set): the harness runs it (exit 0 == pass). A
          failing check is fed back to the agent and the loop continues. This is
          the deterministic gate.
        - `judgment_mode="manager"`: after any deterministic gate passes, the
          loop PAUSES and returns status `awaiting_judgment` with the worker's
          deliverable. The dispatching manager inspects it and calls `judge()`
          to accept (-> complete) or reject (-> resume with feedback). This makes
          the manager the judge -- an independent reviewer, not the worker.
        - otherwise (`judgment_mode="auto"`, no verify_command): the
          `done_marker` ends the loop (self-declared completion).
        """
        if judgment_mode not in ("auto", "manager"):
            raise ValueError(f"Invalid judgment_mode {judgment_mode!r}. Valid: auto, manager")
        worker = self.get_worker(worker_id)
        if not worker:
            raise KeyError(f"Unknown worker {worker_id!r}")
        if not str(task).strip():
            raise ValueError("task is required")

        task_id = _new_id("task")
        # Atomically claim the worker. A worker runs one task at a time over a
        # single conversation; the lock + status check close the concurrent-submit
        # race (the HTTP API is threaded) so two loops can't corrupt the shared
        # conversation/history. A worker stuck after a crash is reclaimed once the
        # lock goes stale (see _worker_lock).
        with self._worker_lock(worker_id):
            fresh = self.get_worker(worker_id) or worker
            if fresh.get("status") in ("running", "awaiting_judgment"):
                raise ValueError(
                    f"Worker {worker_id!r} is busy (status={fresh.get('status')}); "
                    f"resolve or wait for the current task before submitting another."
                )
            worker = fresh
            worker["status"] = "running"
            worker["last_task_id"] = task_id
            self.save_worker(worker)
        effective_max = max(1, int(max_turns)) if goal_loop else 1
        # Scope overrides default to the worker's declared scope.
        eff_write = write_roots if write_roots is not None else worker.get("write_roots")
        eff_read = read_roots if read_roots is not None else worker.get("read_roots")
        eff_deny = deny_roots if deny_roots is not None else worker.get("deny_write_roots")
        eff_search = search_policy or worker.get("search_policy", "artifact_only")
        eff_branch = branch_policy or worker.get("branch_policy", "forbidden")

        rec: Dict[str, Any] = {
            "task_id": task_id,
            "worker_id": worker_id,
            "conversation": worker.get("conversation"),
            "session_key": worker.get("session_key"),
            "task": task,
            "mode": mode,
            "goal_loop": bool(goal_loop),
            "max_turns": effective_max,
            "constraints": constraints,
            "output_contract": output_contract,
            "acceptance_criteria": acceptance_criteria,
            "done_marker": done_marker,
            "blocked_marker": blocked_marker,
            "write_scope": _as_list(eff_write),
            "read_scope": _as_list(eff_read),
            "search_policy": eff_search,
            "branch_policy": eff_branch,
            "verify_command": verify_command,
            "verify_timeout": int(verify_timeout),
            "verify_cwd": verify_cwd or worker.get("workspace", ""),
            "judgment_mode": judgment_mode,
            "judgment_timeout": int(judgment_timeout),
            "use_judge": bool(use_judge),
            "scope_auto_revert": bool(auto_revert),
            # Git state at task start, so the postflight audit attributes only
            # files THIS task touched (not pre-existing / concurrent edits).
            "scope_baseline": list((git_status_porcelain(worker.get("workspace") or ".") or {}).keys()),
            "goal": (task + (f"\n\nAcceptance criteria: {acceptance_criteria}" if acceptance_criteria else "")),
            "status": "running",
            "turns": [],
            "verifications": [],
            "judgments": [],
            "judge_verdicts": [],
            "final_result": "",
            "artifact_manifest": None,
            "created_at": _utc_now(),
            "completed_at": None,
        }
        self._save_task(rec)
        self._log(task_id, "request", {
            "worker_id": worker_id, "task": task, "mode": mode,
            "goal_loop": goal_loop, "max_turns": effective_max,
            "conversation": worker.get("conversation"),
        })
        # (worker already claimed + marked running under the lock above)

        first_prompt = build_task_packet(
            worker, task, mode=mode, constraints=constraints,
            output_contract=output_contract, acceptance_criteria=acceptance_criteria,
            target=target, goal_loop=goal_loop, use_judge=use_judge, done_marker=done_marker,
            blocked_marker=blocked_marker, write_roots=eff_write, read_roots=eff_read,
            deny_roots=eff_deny, search_policy=eff_search, branch_policy=eff_branch,
        )

        status = self._run_loop(
            worker, rec, first_prompt, effective_max, goal_loop,
            done_marker, blocked_marker,
            verify_command=verify_command,
            verify_timeout=int(verify_timeout),
            verify_cwd=rec["verify_cwd"],
            judgment_mode=judgment_mode,
            use_judge=bool(use_judge),
            goal=rec["goal"],
        )

        if status == "awaiting_judgment":
            self._pause_for_judgment(worker, rec)
        else:
            self._finalize(worker, rec, status)
        return rec

    def continue_worker(
        self,
        worker_id: str,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        nudge: str = "",
    ) -> Dict[str, Any]:
        """Resume the worker's last task with extra goal-loop turns.

        Reuses the same named conversation, so the worker keeps full context.
        """
        worker = self.get_worker(worker_id)
        if not worker:
            raise KeyError(f"Unknown worker {worker_id!r}")
        last_id = worker.get("last_task_id")
        if not last_id:
            raise ValueError(f"Worker {worker_id!r} has no task to continue")
        rec = self.get_task(last_id)
        if not rec:
            raise ValueError(f"Task {last_id!r} not found")

        rec["status"] = "running"
        rec["completed_at"] = None
        done_marker = rec.get("done_marker", DEFAULT_DONE_MARKER)
        blocked_marker = rec.get("blocked_marker", DEFAULT_BLOCKED_MARKER)
        start_turn = len(rec.get("turns", [])) + 1
        cap = start_turn + max(1, int(max_turns)) - 1
        rec["max_turns"] = cap
        self._save_task(rec)
        self._log(last_id, "continue", {"from_turn": start_turn, "max_turns": cap, "nudge": nudge})

        worker["status"] = "running"
        self.save_worker(worker)

        first = build_continuation_packet(done_marker, blocked_marker, start_turn, cap)
        if nudge:
            first = f"{first}\nADDITIONAL IN-SCOPE GUIDANCE: {nudge}\n"

        status = self._run_loop(
            worker, rec, first, cap, True, done_marker, blocked_marker,
            start_turn=start_turn,
            verify_command=rec.get("verify_command", ""),
            verify_timeout=int(rec.get("verify_timeout", 120)),
            verify_cwd=rec.get("verify_cwd", ""),
            judgment_mode=rec.get("judgment_mode", "auto"),
            use_judge=bool(rec.get("use_judge", True)),
            goal=rec.get("goal", rec.get("task", "")),
        )
        if status == "awaiting_judgment":
            self._pause_for_judgment(worker, rec)
        else:
            self._finalize(worker, rec, status)
        return rec

    # -- manager judgment --------------------------------------------------

    def _expire_if_stale(self, worker: Dict[str, Any]) -> bool:
        """Expire a manager-judgment pause that no one resolved in time.

        Prevents paused workers from leaking forever. Returns True if it expired.
        """
        if worker.get("status") != "awaiting_judgment":
            return False
        rec = self.get_task(worker.get("last_task_id")) if worker.get("last_task_id") else None
        if not rec or rec.get("status") != "awaiting_judgment":
            return False
        timeout = int(rec.get("judgment_timeout", 3600))
        if _seconds_since(rec.get("awaiting_since", "")) <= timeout:
            return False
        rec.setdefault("judgments", []).append({
            "verdict": "expired", "by": "harness",
            "reason": f"no verdict within {timeout}s", "at": _utc_now(),
        })
        self._log(rec["task_id"], "judgment", {"verdict": "expired"})
        self._finalize(worker, rec, "judgment_expired")
        return True

    def _pause_for_judgment(self, worker: Dict[str, Any], rec: Dict[str, Any]) -> None:
        """Persist a paused task awaiting the manager's verdict (no finalize)."""
        rec["status"] = "awaiting_judgment"
        rec["awaiting_since"] = _utc_now()
        # Build an interim manifest so the manager can see the candidate result.
        rec["artifact_manifest"] = self._build_manifest(worker, rec, "awaiting_judgment")
        self._save_task(rec)
        worker["status"] = "awaiting_judgment"
        self.save_worker(worker)
        self._log(rec["task_id"], "status", {"status": "awaiting_judgment"})

    def judge(
        self,
        worker_id: str,
        verdict: str,
        feedback: str = "",
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> Dict[str, Any]:
        """The dispatching MANAGER's verdict on a paused worker's deliverable.

        verdict="accept" -> task completes (manager-judged).
        verdict="reject" -> the worker resumes with the manager's feedback and
                            works toward another GOAL_COMPLETE (then pauses again).

        This is how the agent that dispatched the task becomes the goal judge:
        an independent reviewer of the worker's output, not the worker itself.
        """
        if verdict not in ("accept", "reject"):
            raise ValueError(f"Invalid verdict {verdict!r}. Valid: accept, reject")
        worker = self.get_worker(worker_id)
        if not worker:
            raise KeyError(f"Unknown worker {worker_id!r}")
        if self._expire_if_stale(worker):
            worker = self.get_worker(worker_id)  # reload; will fail the check below
        last_id = worker.get("last_task_id")
        rec = self.get_task(last_id) if last_id else None
        if not rec:
            raise ValueError(f"Worker {worker_id!r} has no task to judge")
        if rec.get("status") != "awaiting_judgment":
            raise ValueError(f"Task {rec.get('task_id')} is not awaiting judgment "
                             f"(status={rec.get('status')})")

        judged_turn = len(rec.get("turns", []))
        rec.setdefault("judgments", []).append({
            "verdict": verdict, "feedback": feedback, "by": "manager",
            "turn": judged_turn, "at": _utc_now(),
        })
        self._log(rec["task_id"], "judgment", {"verdict": verdict, "turn": judged_turn})

        if verdict == "accept":
            self._finalize(worker, rec, "complete")
            return rec

        # reject -> resume the worker with the manager's feedback.
        done_marker = rec.get("done_marker", DEFAULT_DONE_MARKER)
        blocked_marker = rec.get("blocked_marker", DEFAULT_BLOCKED_MARKER)
        start_turn = len(rec.get("turns", [])) + 1
        cap = start_turn + max(1, int(max_turns)) - 1
        rec["max_turns"] = cap
        rec["status"] = "running"
        self._save_task(rec)
        worker["status"] = "running"
        self.save_worker(worker)

        first = build_continuation_packet(done_marker, blocked_marker, start_turn, cap)
        first += (
            f"\n\nMANAGER REVIEW: your '{done_marker}' was REJECTED by the manager.\n"
            f"Required changes: {feedback or '(see acceptance criteria)'}\n"
            f"Address this, then emit '{done_marker}' again for re-review."
        )
        status = self._run_loop(
            worker, rec, first, cap, True, done_marker, blocked_marker,
            start_turn=start_turn,
            verify_command=rec.get("verify_command", ""),
            verify_timeout=int(rec.get("verify_timeout", 120)),
            verify_cwd=rec.get("verify_cwd", ""),
            judgment_mode=rec.get("judgment_mode", "manager"),
            use_judge=bool(rec.get("use_judge", True)),
            goal=rec.get("goal", rec.get("task", "")),
        )
        if status == "awaiting_judgment":
            self._pause_for_judgment(worker, rec)
        else:
            self._finalize(worker, rec, status)
        return rec

    # -- goal loop ---------------------------------------------------------

    def _run_loop(
        self,
        worker: Dict[str, Any],
        rec: Dict[str, Any],
        first_prompt: str,
        max_turns: int,
        goal_loop: bool,
        done_marker: str,
        blocked_marker: str,
        start_turn: int = 1,
        verify_command: str = "",
        verify_timeout: int = 120,
        verify_cwd: str = "",
        judgment_mode: str = "auto",
        use_judge: bool = True,
        goal: str = "",
    ) -> str:
        """Harness-owned goal loop. Returns a terminal run status.

        Never calls /goal. The loop control (parse, verify, stop, continue) lives
        here, in the control plane. When `verify_command` is set, a GOAL_COMPLETE
        marker triggers an EXTERNAL check the harness runs; only exit 0 ends the
        loop as complete. A failed check is fed back and the loop continues.
        """
        conversation = worker.get("conversation", "")
        session_key = worker.get("session_key", "")
        model = worker.get("model", DEFAULT_MODEL)
        done_marker = done_marker or DEFAULT_DONE_MARKER
        blocked_marker = blocked_marker or DEFAULT_BLOCKED_MARKER
        caller = self._caller_for(worker)
        # "runs" transport carries context client-side via conversation_history;
        # "responses" relies on the server-side named conversation instead.
        uses_history = worker.get("transport", "runs") == "runs"
        history: Optional[List[Dict[str, str]]] = list(worker.get("history", [])) if uses_history else None

        status = "running"
        pending_feedback = ""
        turn = start_turn
        while turn <= max_turns:
            if turn == start_turn:
                prompt = first_prompt
            else:
                prompt = build_continuation_packet(done_marker, blocked_marker, turn, max_turns)
            if pending_feedback:
                prompt = f"{prompt}\n\n{pending_feedback}"
                pending_feedback = ""
            self._log(rec["task_id"], "prompt", {"turn": turn, "prompt": prompt})
            try:
                resp = caller(
                    prompt, conversation=conversation,
                    session_key=session_key, model=model,
                    history=history,
                )
            except Exception as exc:  # backend/transport failure
                self._log(rec["task_id"], "error", {"turn": turn, "error": str(exc)})
                rec.setdefault("turns", []).append({
                    "turn": turn, "prompt": prompt, "output": "",
                    "error": str(exc), "usage": {},
                })
                status = "error"
                rec["error"] = str(exc)
                break

            text = resp.get("text", "") if isinstance(resp, dict) else str(resp)
            usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
            rec.setdefault("turns", []).append({
                "turn": turn, "prompt": prompt, "output": text,
                "usage": usage, "response_id": (resp.get("response_id") if isinstance(resp, dict) else ""),
                "approvals": (resp.get("approvals", 0) if isinstance(resp, dict) else 0),
            })
            rec["final_result"] = text
            if history is not None:
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": text})
            self._log(rec["task_id"], "turn_output", {
                "turn": turn, "output": text, "usage": usage,
            })
            self._save_task(rec)

            if not goal_loop:
                status = "complete"  # single-shot: one turn is the whole task
                break

            # Decide done / blocked / continue for this turn.
            if use_judge:
                # Hermes /goal mechanism: an aux model judges the response.
                try:
                    verdict = self._judge_for(worker)(goal or rec.get("task", ""), text)
                except Exception as exc:  # fail-open like /goal
                    verdict = {"done": False, "blocked": False,
                               "reason": f"judge error, continuing (fail-open): {exc}"}
                rec.setdefault("judge_verdicts", []).append({"turn": turn, **verdict})
                self._log(rec["task_id"], "judge", {
                    "turn": turn, "done": verdict.get("done"), "reason": verdict.get("reason"),
                })
                self._save_task(rec)
                is_complete = bool(verdict.get("done"))
                is_blocked = bool(verdict.get("blocked"))
                judge_reason = str(verdict.get("reason", ""))
            else:
                marker = detect_marker(text, done_marker, blocked_marker)
                is_complete = marker == "complete"
                is_blocked = marker == "blocked"
                judge_reason = ""

            if is_blocked:
                status = "blocked"
                break
            if is_complete:
                # 1. Deterministic gate first (if configured).
                if verify_command:
                    vr = run_verification(verify_command, cwd=verify_cwd or None, timeout=verify_timeout)
                    vr["turn"] = turn
                    rec.setdefault("verifications", []).append(vr)
                    self._save_task(rec)
                    self._log(rec["task_id"], "verify", {
                        "turn": turn, "passed": vr["passed"],
                        "exit_code": vr["exit_code"], "command": verify_command,
                    })
                    if not vr["passed"]:
                        # Claimed done but the check failed. Reject and feed back.
                        pending_feedback = (
                            f"VERIFICATION REJECTED. You emitted '{done_marker}', but the harness "
                            f"ran the acceptance check and it FAILED (exit {vr['exit_code']}).\n"
                            f"Command: {verify_command}\nOutput (tail):\n{vr['output']}\n"
                            f"Do NOT emit '{done_marker}' again until this command exits 0. Fix "
                            f"the in-scope work and continue."
                        )
                        turn += 1
                        continue
                # 2. Manager judgment: pause and hand the deliverable back.
                if judgment_mode == "manager":
                    self._log(rec["task_id"], "awaiting_judgment", {"turn": turn})
                    status = "awaiting_judgment"
                    break
                # 3. Otherwise complete (judge-verified, command-verified, or
                #    self-declared, depending on what was configured).
                status = "complete"
                break
            # Not done -> continue. Feed the judge's reason forward (like /goal's
            # "Continuing toward goal: <reason>" continuation).
            if use_judge and judge_reason:
                pending_feedback = (
                    "CONTINUING TOWARD THE GOAL. The goal judge says it is not yet "
                    f"complete: {judge_reason}\nKeep working on the in-scope task."
                )
            turn += 1
        else:
            status = "budget_exhausted"

        if history is not None:
            # Keep the last 40 messages (20 turns) of rolling context so a
            # persistent worker remembers across tasks without unbounded growth.
            worker["history"] = history[-40:]

        return status

    # -- finalize / manifest ----------------------------------------------

    def _scope_audit(self, worker: Dict[str, Any], rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Postflight git audit, designed for an ALWAYS-DIRTY / busy tree.

        In a repo with constant concurrent edits, a global git diff cannot
        attribute changes to this task. So the reliable signal is the agent's
        OWN reported files, cross-checked against git:
          - agent_confirmed : agent said it changed it AND git agrees (real work)
          - agent_unconfirmed: agent claimed it but git shows NO change (fabrication)
          - agent_escapes    : agent_confirmed files outside write_scope (real breach)
          - unattributed     : git-changed, agent never claimed -> concurrent edits
                               OR a hidden agent change; CANNOT be told apart in a
                               busy tree, so never auto-reverted, only surfaced.
        Auto-revert (opt-in) touches ONLY agent_escapes.
        """
        ws = worker.get("workspace") or "."
        after = git_status_porcelain(ws)
        if after is None:
            return None  # not a git repo -> fall back to worker-reported manifest
        baseline = set(rec.get("scope_baseline", []))
        touched = {p for p in after if p not in baseline}
        # Normalize everything to REPO-ROOT-relative (git status' frame), so a
        # workspace that's a subdirectory of the repo still attributes correctly.
        prefix = _git_oneline(ws, "rev-parse", "--show-prefix") or ""
        root = _git_oneline(ws, "rev-parse", "--show-toplevel")
        write_scope = [
            _norm_rel(r, ws, prefix, root).rstrip("/")
            for r in rec.get("write_scope", []) if r
        ]

        def in_scope(p: str) -> bool:
            return any(p == r or p.startswith(r + "/") for r in write_scope)

        reported = parse_reported_manifest(rec.get("final_result", ""))
        agent_files = {
            _norm_rel(f, ws, prefix, root)
            for f in _as_list(reported.get("created_files")) + _as_list(reported.get("modified_files"))
        }
        base = Path(root) if root else Path(ws)

        def on_disk(relpath: str) -> bool:
            try:
                return (base / relpath).exists()
            except OSError:
                return False

        agent_confirmed = sorted(f for f in agent_files if f in touched)  # claimed AND changed this run
        # Claimed but no git change since baseline -> split by whether it exists:
        #   missing on disk  -> FABRICATION (the real lie signal)
        #   exists on disk   -> pre-existing / identical no-op rewrite (benign)
        agent_missing = sorted(f for f in agent_files if f not in touched and not on_disk(f))
        agent_unchanged = sorted(f for f in agent_files if f not in touched and on_disk(f))
        agent_escapes = [f for f in agent_confirmed if write_scope and not in_scope(f)]
        unattributed = sorted(p for p in touched if p not in agent_files)  # likely concurrent edits

        reverted: List[str] = []
        if rec.get("scope_auto_revert") and agent_escapes:
            for p in agent_escapes:
                if git_revert_path(ws, p, after.get(p, "")):
                    reverted.append(p)
                    self._log(rec["task_id"], "scope_revert", {"path": p})
        return {
            "source": "git_observed",
            "agent_confirmed": agent_confirmed,
            "agent_missing": agent_missing,        # claimed, not on disk -> fabrication
            "agent_unchanged": agent_unchanged,    # claimed, on disk, no change this run (pre-existing/no-op)
            "agent_escapes": agent_escapes,
            "unattributed": unattributed,
            "reverted": reverted,
            "agent_clean": not agent_escapes,
        }

    def _finalize(self, worker: Dict[str, Any], rec: Dict[str, Any], status: str) -> None:
        rec["status"] = status
        rec["completed_at"] = _utc_now()
        rec["scope_audit"] = self._scope_audit(worker, rec)
        if rec["scope_audit"]:
            self._log(rec["task_id"], "scope_audit", rec["scope_audit"])
        manifest = self._build_manifest(worker, rec, status)
        rec["artifact_manifest"] = manifest
        self._save_task(rec)
        # Persist worker-level manifest pointer (latest).
        Path(worker["artifact_manifest"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._log(rec["task_id"], "status", {"status": status})
        self._log(rec["task_id"], "manifest", {"manifest": manifest})
        self._log(rec["task_id"], "final", {"result": rec.get("final_result", "")})

        worker["status"] = status if status in VALID_STATUSES else "idle"
        if status not in ("running",):
            # Worker goes back to idle for routing, but keeps last_task_id and
            # the terminal status is preserved on the task record.
            worker["status"] = "idle"
        self.save_worker(worker)

    def _build_manifest(self, worker: Dict[str, Any], rec: Dict[str, Any], status: str) -> Dict[str, Any]:
        reported = parse_reported_manifest(rec.get("final_result", ""))
        created = _as_list(reported.get("created_files"))
        modified = _as_list(reported.get("modified_files"))
        next_task = reported.get("next_recommended_task") or ""
        verifications = rec.get("verifications", [])
        last_verify = verifications[-1] if verifications else None
        # Harness-observed truth: did an external acceptance check pass?
        # None == no external check was configured (completion is self-declared).
        harness_verified: Optional[bool] = last_verify["passed"] if last_verify else None
        judgments = rec.get("judgments", [])
        last_judgment = judgments[-1] if judgments else None
        audit = rec.get("scope_audit")
        return {
            "worker_id": worker["worker_id"],
            "task_id": rec["task_id"],
            "conversation": worker.get("conversation"),
            "created_files": created,
            "modified_files": modified,
            "read_scope": rec.get("read_scope", []),
            "write_scope": rec.get("write_scope", []),
            "verification": reported.get("verification", ""),
            # Harness-observed completion truth. None == completion was self-
            # declared (no verify_command). True/False == an external acceptance
            # check actually ran and passed/failed.
            "harness_verified": harness_verified,
            "harness_verification": last_verify,
            # Manager verdict (judgment_mode="manager"): the dispatching agent's
            # independent accept/reject. None == no manager judgment was used.
            "manager_judgment": last_judgment,
            # Harness-observed via git, attribution-aware for a busy/dirty tree.
            # Trust these over the self-reported created_files/modified_files above:
            "changed_files_source": ("git_observed" if audit else "worker_reported"),
            "agent_confirmed": (audit["agent_confirmed"] if audit else None),   # claimed AND changed this run
            "agent_missing": (audit["agent_missing"] if audit else None),       # claimed but NOT on disk (FABRICATION)
            "agent_unchanged": (audit["agent_unchanged"] if audit else None),   # claimed, on disk, unchanged (pre-existing/no-op)
            "agent_escapes": (audit["agent_escapes"] if audit else None),      # confirmed AND out of scope
            "agent_clean": (audit["agent_clean"] if audit else None),          # agent stayed in scope
            "unattributed_changes": (audit["unattributed"] if audit else None),  # likely concurrent edits; not the agent
            "scope_reverted": (audit["reverted"] if audit else None),
            "status": status,
            "next_recommended_task": next_task if isinstance(next_task, str) else "",
            "generated_at": _utc_now(),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Worker API control plane (Hermes Harness).")
    parser.add_argument("--state-dir", default=None, help="State directory (default: <skill>/_state).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("worker-create", help="Register a persistent worker.")
    p_create.add_argument("--worker-id", required=True)
    p_create.add_argument("--role", default="specialist")
    p_create.add_argument("--conversation", default="")
    p_create.add_argument("--session-key", default="")
    p_create.add_argument("--model", default=DEFAULT_MODEL)
    p_create.add_argument("--workspace", default=".")
    p_create.add_argument("--read-roots", "--read-root", dest="read_roots", default="")
    p_create.add_argument("--write-roots", "--write-root", dest="write_roots", default="")
    p_create.add_argument("--deny-write-roots", default="")
    p_create.add_argument("--search-policy", default="artifact_only")
    p_create.add_argument("--branch-policy", default="forbidden")
    p_create.add_argument("--transport", default="runs", choices=["runs", "responses"],
                          help="runs (default, can execute code) or responses (named conversation, no exec).")

    sub.add_parser("worker-list", help="List registered workers.")

    p_get = sub.add_parser("worker-get", help="Show one worker record.")
    p_get.add_argument("--worker-id", required=True)

    p_state = sub.add_parser("worker-state", help="Show worker state.")
    p_state.add_argument("--worker-id", required=True)

    p_task = sub.add_parser("task-submit", help="Submit a task and run the goal loop.")
    p_task.add_argument("--worker-id", required=True)
    p_task.add_argument("--task", required=True)
    p_task.add_argument("--mode", default="persistent_worker_verify")
    p_task.add_argument("--no-goal-loop", action="store_true")
    p_task.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p_task.add_argument("--constraints", default="")
    p_task.add_argument("--output-contract", default="")
    p_task.add_argument("--acceptance-criteria", default="")
    p_task.add_argument("--target", default="")
    p_task.add_argument("--done-marker", default=DEFAULT_DONE_MARKER)
    p_task.add_argument("--blocked-marker", default=DEFAULT_BLOCKED_MARKER)
    p_task.add_argument("--write-root", "--write-roots", dest="write_root", default=None)
    p_task.add_argument("--read-root", "--read-roots", dest="read_root", default=None)
    p_task.add_argument("--verify-command", default="", help="External acceptance check the harness runs on GOAL_COMPLETE (exit 0 == pass).")
    p_task.add_argument("--verify-timeout", type=int, default=120)
    p_task.add_argument("--judgment-mode", default="auto", choices=["auto", "manager"],
                        help="'manager' pauses on completion and returns awaiting_judgment for the dispatcher to judge.")
    p_task.add_argument("--no-judge", action="store_true",
                        help="Disable the aux judge loop; fall back to literal GOAL_COMPLETE marker.")
    p_task.add_argument("--judge-model", default="",
                        help="Override the goal-judge model (default: deepseek-v4-flash via DeepSeek).")
    p_task.add_argument("--auto-revert", action="store_true",
                        help="Write firewall: auto-revert agent-made changes outside write scope (git workspace only).")

    p_judge = sub.add_parser("judge", help="Manager verdict on a worker awaiting judgment.")
    p_judge.add_argument("--worker-id", required=True)
    p_judge.add_argument("--verdict", required=True, choices=["accept", "reject"])
    p_judge.add_argument("--feedback", default="")
    p_judge.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)

    p_cont = sub.add_parser("continue", help="Continue a worker's last task.")
    p_cont.add_argument("--worker-id", required=True)
    p_cont.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p_cont.add_argument("--nudge", default="")

    p_tget = sub.add_parser("task-get", help="Show a task record.")
    p_tget.add_argument("--task-id", required=True)

    p_log = sub.add_parser("log", help="Show a task's structured log.")
    p_log.add_argument("--task-id", required=True)

    p_serve = sub.add_parser("serve", help="Start the HTTP Worker API.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8770)

    args = parser.parse_args(argv)
    # Optional judge-model override (task-submit only).
    _judge = None
    if getattr(args, "judge_model", ""):
        _judge = default_judge(model=args.judge_model)
    ctl = WorkerController(state_dir=args.state_dir, judge=_judge)

    if args.command == "worker-create":
        w = ctl.create_worker(
            worker_id=args.worker_id, role=args.role, conversation=args.conversation,
            session_key=args.session_key, model=args.model, workspace=args.workspace,
            read_roots=_as_list(args.read_roots), write_roots=_as_list(args.write_roots),
            deny_write_roots=_as_list(args.deny_write_roots),
            search_policy=args.search_policy, branch_policy=args.branch_policy,
            transport=args.transport,
        )
        _print(w)
    elif args.command == "worker-list":
        _print(ctl.list_workers())
    elif args.command == "worker-get":
        _print(ctl.get_worker(args.worker_id) or {"error": "not found"})
    elif args.command == "worker-state":
        _print(ctl.get_state(args.worker_id))
    elif args.command == "task-submit":
        rec = ctl.submit_task(
            args.worker_id, args.task, mode=args.mode,
            goal_loop=not args.no_goal_loop, max_turns=args.max_turns,
            constraints=args.constraints, output_contract=args.output_contract,
            acceptance_criteria=args.acceptance_criteria, target=args.target,
            done_marker=args.done_marker, blocked_marker=args.blocked_marker,
            write_roots=_as_list(args.write_root) if args.write_root else None,
            read_roots=_as_list(args.read_root) if args.read_root else None,
            verify_command=args.verify_command, verify_timeout=args.verify_timeout,
            judgment_mode=args.judgment_mode, use_judge=not args.no_judge,
            auto_revert=args.auto_revert,
        )
        _print(rec)
    elif args.command == "judge":
        _print(ctl.judge(args.worker_id, args.verdict, feedback=args.feedback, max_turns=args.max_turns))
    elif args.command == "continue":
        _print(ctl.continue_worker(args.worker_id, max_turns=args.max_turns, nudge=args.nudge))
    elif args.command == "task-get":
        _print(ctl.get_task(args.task_id) or {"error": "not found"})
    elif args.command == "log":
        _print(ctl.get_log(args.task_id))
    elif args.command == "serve":
        from worker_api import serve  # noqa: E402
        serve(host=args.host, port=args.port, state_dir=args.state_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
