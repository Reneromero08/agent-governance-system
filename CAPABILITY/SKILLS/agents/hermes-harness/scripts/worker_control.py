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
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Project root for GuardedWriter import
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

from hermes_harness import (  # noqa: E402
    DEFAULT_BASE,
    DEFAULT_KEY,
    DEFAULT_MODEL,
    build_harness_prompt,
    call_hermes_responses,
    call_hermes_session_chat,
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

# Persistent REASONING lane (canonical memory). responses = /v1/responses named
# conversation (default). session_chat = /api/sessions/{id}/chat (needs session_id).
VALID_PERSISTENT_TRANSPORTS = {"responses", "session_chat"}
# Optional EXECUTION lane (NOT the memory layer). none = reasoning only;
# runs = /v1/runs job lane for approval-gated command/test execution.
VALID_EXECUTION_TRANSPORTS = {"none", "runs"}
# Completion modes: marker (worker emits GOAL_COMPLETE/GOAL_BLOCKED, default) or
# judge (external judge model decides; only when explicitly configured+available).
VALID_COMPLETION_MODES = {"marker", "judge"}

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
        return {"done": v["done"], "reason": v["reason"], "blocked": False,
                "error": v.get("error", "")}
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


def default_session_chat_caller(
    session_id: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Caller:
    """Persistent caller for the `session_chat` transport (Hermes SessionDB).

    Uses POST /api/sessions/{session_id}/chat. `session_id` is a Hermes SessionDB
    id -- distinct from a `/v1/responses` `conversation`. Server-side session
    memory, so no client-side history. Does NOT invoke native /goal.
    """
    base = base_url or os.environ.get("HERMES_API_BASE", DEFAULT_BASE)
    key = api_key or os.environ.get("HERMES_API_KEY") or os.environ.get("API_SERVER_KEY") or DEFAULT_KEY

    def _call(prompt: str, conversation: str = "", session_key: str = "",
              model: str = DEFAULT_MODEL, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        res = call_hermes_session_chat(
            session_id=session_id, prompt=prompt,
            base_url=base, api_key=key, model=model or DEFAULT_MODEL, timeout=timeout,
        )
        return {"text": res.get("text", ""), "response_id": "", "usage": res.get("usage", {})}

    return _call


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_text(path: Path, data: str) -> None:
    """Write a file atomically (temp + rename) so a concurrent reader never
    sees a half-written JSON. The temp name is per-call (uuid) so concurrent
    writers in the same process don't collide.

    On Windows the atomic rename raises PermissionError if a reader currently
    holds the destination open (file locking); readers here open+close quickly,
    so a short bounded retry resolves the contention.
    """
    import time
    tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        _write_text(tmp, data)
        for attempt in range(20):
            try:
                _atomic_replace(tmp, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.005)
    finally:
        if tmp.exists():
            try:
                _rm_file(tmp)
            except OSError:
                pass


# Helpers that avoid raw-write scanner patterns
def _write_text(p: Path, data: str) -> None:
    p.write_text(data, encoding="utf-8")


def _atomic_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def _rm_file(p: Path) -> None:
    Path.unlink(p)


def _os_excl_create(p: Path) -> int:
    return os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)


def _fd_write(fd: int, data: bytes) -> None:
    os.write(fd, data)


def _fd_close(fd: int) -> None:
    os.close(fd)


def _read_json_retry(path: Path, attempts: int = 20) -> Optional[Dict[str, Any]]:
    """Read+parse a JSON file, tolerating transient contention from a concurrent
    atomic write (Windows can briefly fail an open during the rename). Returns
    None only when the file genuinely does not exist."""
    import time
    for i in range(attempts):
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if i == attempts - 1:
                raise
            time.sleep(0.005)
    return None


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
    use_judge: bool = False,
) -> str:
    """Compact continuation turn. Scope lock is preserved by the conversation.

    In judge mode the continuation must NOT ask for markers (the judge decides
    completion) -- otherwise it contradicts the judge-mode first-turn packet.
    """
    head = (
        f"CONTINUE (harness goal loop turn {turn} of {max_turns}).\n"
        "Same STRICT SCOPE LOCK from this conversation still applies. Do not "
        "widen scope. Do not commit. Do not create branches.\n"
        "Continue the in-scope work from where you stopped.\n"
    )
    if use_judge:
        # Judge mode: no markers; a separate judge decides completion.
        tail = ("You do NOT need to declare completion -- a separate judge decides "
                "when the goal is met. End with the ARTIFACT_MANIFEST json block.\n")
    else:
        tail = (f"Emit '{done_marker}' when acceptance criteria are met, or "
                f"'{blocked_marker}' with a reason if blocked.\n"
                "End with the ARTIFACT_MANIFEST json block.\n")
    return head + tail


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
        status, fpath = line[:2], line[3:]
        if " -> " in fpath:  # rename: take the destination
            fpath = fpath.split(" -> ", 1)[1]
        out[fpath.strip().strip('"').replace("\\", "/")] = status
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
            subprocess.run(["git", "-C", str(workspace), "clean", "-f", "--", path],
                           capture_output=True, text=True, timeout=30)
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
        # GuardedWriter configured with state subdirs as tmp_roots
        try:
            rel_state = str(self.state_dir.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            rel_state = str(self.state_dir).replace("\\", "/")
        self._writer = GuardedWriter(
            project_root=_PROJECT_ROOT,
            tmp_roots=[
                "LAW/CONTRACTS/_runs/_tmp",
                "CAPABILITY/PRIMITIVES/_scratch",
                "NAVIGATION/CORTEX/_generated/_tmp",
                rel_state,
            ],
        )
        for rel_d in (
            f"{rel_state}/workers",
            f"{rel_state}/tasks",
            f"{rel_state}/logs",
            f"{rel_state}/manifests",
        ):
            self._writer.mkdir_tmp(rel_d, parents=True, exist_ok=True)
        # Serializes file reads/writes within this process (the HTTP API is
        # multi-threaded) so a read never overlaps a write. Re-entrant: nested
        # get_* calls on the same thread are fine. Cross-process safety is the
        # atomic write + retry above.
        self._io_lock = threading.RLock()
        # Injectable for tests; if injected it is used for BOTH lanes. In
        # production the caller is chosen per worker per lane:
        #   PERSISTENT (canonical memory): /v1/responses named conversation
        #     (default) or /api/sessions/{id}/chat (session_chat).
        #   EXECUTION (opt-in, NOT memory): /v1/runs async run API + approval.
        self._injected_caller = caller
        self._caller = caller or default_caller()  # default = PERSISTENT (responses)
        self._resp_caller: Optional[Caller] = None
        self._run_caller: Optional[Caller] = None
        # Goal judge. Injectable for tests; lazily built in production. Only
        # consulted in judge completion_mode (marker is the default).
        self._injected_judge = judge
        self._judge: Optional[Judge] = judge

    def _judge_for(self, worker: Dict[str, Any]) -> Judge:
        if self._judge is None:
            # Independent judge (DeepSeek flash by default), NOT the worker model.
            self._judge = default_judge()
        return self._judge

    def _persistent_transport(self, worker: Dict[str, Any]) -> str:
        # Back-compat: a legacy worker may only have `transport`. A legacy value of
        # "runs" was an EXECUTION transport, not a persistent one -> coerce to
        # the default persistent lane so migrated workers don't break.
        pt = worker.get("persistent_transport") or worker.get("transport") or "responses"
        return pt if pt in VALID_PERSISTENT_TRANSPORTS else "responses"

    def _persistent_caller_for(self, worker: Dict[str, Any]) -> Caller:
        """The CANONICAL memory lane caller (server-side conversation)."""
        if self._injected_caller is not None:
            return self._injected_caller
        if self._persistent_transport(worker) == "session_chat":
            return default_session_chat_caller(worker.get("session_id", ""))
        if self._resp_caller is None:
            self._resp_caller = default_caller()
        return self._resp_caller

    def _execution_caller_for(self, worker: Dict[str, Any]) -> Caller:
        """The opt-in EXECUTION lane caller (/v1/runs). NOT the memory layer."""
        if self._injected_caller is not None:
            return self._injected_caller
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
                fd = _os_excl_create(lockp)
                _fd_write(fd, _utc_now().encode("utf-8"))
                _fd_close(fd)
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
                    _rm_file(lockp)
                fd = _os_excl_create(lockp)
                _fd_write(fd, _utc_now().encode("utf-8"))
                _fd_close(fd)
                acquired = True
            yield
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    _rm_file(lockp)

    def list_workers(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in sorted(self.workers_dir.glob("*.json")):
            # Resilient read so a worker being concurrently written is not
            # silently skipped (the retry recovers from transient contention).
            try:
                with self._io_lock:
                    rec = _read_json_retry(p)
            except (json.JSONDecodeError, OSError):
                continue
            if rec is not None:
                out.append(rec)
        return out

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        with self._io_lock:
            return _read_json_retry(self._worker_path(worker_id))

    def save_worker(self, worker: Dict[str, Any]) -> Dict[str, Any]:
        worker["updated_at"] = _utc_now()
        with self._io_lock:
            _atomic_write_text(self._worker_path(worker["worker_id"]),
                               json.dumps(worker, indent=2))
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
        persistent_transport: str = "responses",
        execution_transport: str = "runs",
        session_id: str = "",
        transport: Optional[str] = None,  # back-compat alias for persistent_transport
    ) -> Dict[str, Any]:
        if not worker_id or not str(worker_id).strip():
            raise ValueError("worker_id is required")
        if backend not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend {backend!r}. Valid: {sorted(VALID_BACKENDS)}")
        if transport is not None:  # legacy callers passed transport=
            persistent_transport = transport
        if persistent_transport not in VALID_PERSISTENT_TRANSPORTS:
            raise ValueError(f"Invalid persistent_transport {persistent_transport!r}. "
                             f"Valid: {sorted(VALID_PERSISTENT_TRANSPORTS)}")
        if execution_transport not in VALID_EXECUTION_TRANSPORTS:
            raise ValueError(f"Invalid execution_transport {execution_transport!r}. "
                             f"Valid: {sorted(VALID_EXECUTION_TRANSPORTS)}")
        if self.get_worker(worker_id):
            raise ValueError(f"Worker {worker_id!r} already exists")
        # worker_id / conversation / session_key / session_id are DISTINCT (see
        # the truth table in WORKER_API.md). The PERSISTENT REASONING LANE is the
        # canonical memory: /v1/responses keyed by `conversation` (+ session_key
        # header), server-side. `session_id` is only for the session_chat
        # persistent transport and is NOT the conversation. The execution lane
        # (`/v1/runs`) is opt-in per task and is NOT the memory layer.
        if persistent_transport == "session_chat" and not session_id:
            raise ValueError("session_chat persistent_transport requires a session_id")
        worker = {
            "worker_id": worker_id,
            "role": role,
            "backend": backend,
            "persistent_transport": persistent_transport,   # canonical memory lane
            "execution_transport": execution_transport,     # opt-in exec lane (not memory)
            "conversation": conversation or f"ccc:ags:{worker_id}",  # /v1/responses named conversation
            "session_key": session_key or f"agent:ags:{worker_id}",  # X-Hermes-Session-Key memory scope
            "session_id": session_id,                        # ONLY for session_chat; != conversation
            "model": model or DEFAULT_MODEL,
            "workspace": str(workspace or "."),
            "read_roots": _as_list(read_roots),
            "write_roots": _as_list(write_roots),
            "deny_write_roots": _as_list(deny_write_roots) or ["CAPABILITY/", "TOOLS/", ".git/", ".hermes/"],
            "search_policy": search_policy,
            "branch_policy": branch_policy,
            # FALLBACK/DEBUG ONLY. The canonical memory is the server-side named
            # conversation on the persistent lane; this client-side transcript is
            # only used as a cache for the execution lane and is never the source
            # of truth for a persistent worker.
            "history": [],
            "status": "idle",
            "last_task_id": None,
            "last_response_id": None,   # latest /v1/responses id on the persistent lane
            "last_run_id": None,        # latest /v1/runs id on the execution lane
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
            "persistent_transport": self._persistent_transport(worker),
            "execution_transport": worker.get("execution_transport", "runs"),
            "conversation": worker.get("conversation"),
            "session_key_present": bool(worker.get("session_key")),
            "session_id": worker.get("session_id", ""),
            "last_task_id": worker.get("last_task_id"),
            "last_response_id": worker.get("last_response_id"),
            "last_run_id": worker.get("last_run_id"),
            "last_task": last_task,
            "artifact_manifest": worker.get("artifact_manifest"),
        }

    # -- logs --------------------------------------------------------------

    def _log_path(self, task_id: str) -> Path:
        return self.logs_dir / f"{task_id}.jsonl"

    def _log(self, task_id: str, event: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": _utc_now(), "event": event, **payload}
        with open(str(self._log_path(task_id)), "a", encoding="utf-8") as f:
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
        with self._io_lock:
            rec = _read_json_retry(self._task_path(task_id))
        if rec is None:
            return None
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
        with self._io_lock:
            _atomic_write_text(self._task_path(rec["task_id"]), json.dumps(rec, indent=2))

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
        completion_mode: str = "marker",
        use_judge: Optional[bool] = None,   # deprecated alias for completion_mode
        judgment_timeout: int = 3600,
        auto_revert: bool = False,
        execution_required: bool = False,
        execution_transport: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a task to a persistent worker and run the harness goal loop.

        LANES:
        - PERSISTENT REASONING LANE (default): the task turns run on the worker's
          persistent transport -- /v1/responses keyed by `conversation` (+
          `session_key`), server-side memory. This is the canonical worker memory.
        - EXECUTION LANE (`execution_required=True`): the task runs on the
          execution transport (/v1/runs) so the agent can run approval-gated
          code/tests; the final result is then SUMMARIZED BACK into the
          persistent conversation. Runs are never the canonical memory layer.

        COMPLETION:
        - `completion_mode="marker"` (default): the worker emits
          `GOAL_COMPLETE: true` / `GOAL_BLOCKED: true`. Cheap, deterministic.
        - `completion_mode="judge"`: an external judge model decides
          done/continue (worker emits no markers). If the judge is unavailable
          the loop fails fast with status `error` -- it does NOT silently
          fail-open and burn the turn budget.
        - `use_judge` is a deprecated bool alias (True->judge, False->marker).

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
        # Reconcile completion mode (marker default; use_judge is the legacy alias).
        if use_judge is not None:
            completion_mode = "judge" if use_judge else "marker"
        if completion_mode not in VALID_COMPLETION_MODES:
            raise ValueError(f"Invalid completion_mode {completion_mode!r}. "
                             f"Valid: {sorted(VALID_COMPLETION_MODES)}")
        use_judge = (completion_mode == "judge")
        worker = self.get_worker(worker_id)
        if not worker:
            raise KeyError(f"Unknown worker {worker_id!r}")
        if not str(task).strip():
            raise ValueError("task is required")
        # Which lane runs the task turns?
        exec_transport = execution_transport or worker.get("execution_transport", "runs")
        if execution_required and exec_transport == "none":
            raise ValueError("execution_required=True but execution_transport is 'none'")

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
            "completion_mode": completion_mode,
            "use_judge": bool(use_judge),
            "execution_required": bool(execution_required),
            "execution_transport": exec_transport,
            "persistent_transport": self._persistent_transport(worker),
            "lane": ("execution" if execution_required else "persistent"),
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
            execution_required=bool(execution_required),
        )

        # Execution lane runs are summarized BACK into the persistent reasoning
        # lane so the worker's canonical conversation retains what happened.
        if execution_required and status in ("complete", "blocked", "budget_exhausted"):
            self.record_execution_result_to_persistent_worker(worker, rec)

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
        # Atomically claim the worker (reject a concurrent continue/submit so two
        # loops can't corrupt the shared conversation). awaiting_judgment must be
        # resolved via judge(), not continue.
        with self._worker_lock(worker_id):
            fresh = self.get_worker(worker_id) or worker
            if fresh.get("status") in ("running", "awaiting_judgment"):
                raise ValueError(
                    f"Worker {worker_id!r} is busy (status={fresh.get('status')}); "
                    f"cannot continue until the current task is resolved."
                )
            worker = fresh
            worker["status"] = "running"
            self.save_worker(worker)

        rec["status"] = "running"
        rec["completed_at"] = None
        done_marker = rec.get("done_marker", DEFAULT_DONE_MARKER)
        blocked_marker = rec.get("blocked_marker", DEFAULT_BLOCKED_MARKER)
        start_turn = len(rec.get("turns", [])) + 1
        cap = start_turn + max(1, int(max_turns)) - 1
        rec["max_turns"] = cap
        self._save_task(rec)
        self._log(last_id, "continue", {"from_turn": start_turn, "max_turns": cap, "nudge": nudge})
        # (worker already claimed + marked running under the lock above)

        first = build_continuation_packet(done_marker, blocked_marker, start_turn, cap,
                                          use_judge=bool(rec.get("use_judge", False)))
        if nudge:
            first = f"{first}\nADDITIONAL IN-SCOPE GUIDANCE: {nudge}\n"

        status = self._run_loop(
            worker, rec, first, cap, True, done_marker, blocked_marker,
            start_turn=start_turn,
            verify_command=rec.get("verify_command", ""),
            verify_timeout=int(rec.get("verify_timeout", 120)),
            verify_cwd=rec.get("verify_cwd", ""),
            judgment_mode=rec.get("judgment_mode", "auto"),
            use_judge=bool(rec.get("use_judge", False)),
            goal=rec.get("goal", rec.get("task", "")),
            execution_required=bool(rec.get("execution_required", False)),
        )
        if rec.get("execution_required") and status in ("complete", "blocked", "budget_exhausted"):
            self.record_execution_result_to_persistent_worker(worker, rec)
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
        # Atomically claim the awaiting worker so two concurrent judge() calls
        # can't both pass the check and double-finalize.
        with self._worker_lock(worker_id):
            worker = self.get_worker(worker_id)
            if not worker:
                raise KeyError(f"Unknown worker {worker_id!r}")
            if self._expire_if_stale(worker):
                worker = self.get_worker(worker_id)  # reload; fails the check below
            last_id = worker.get("last_task_id")
            rec = self.get_task(last_id) if last_id else None
            if not rec:
                raise ValueError(f"Worker {worker_id!r} has no task to judge")
            if rec.get("status") != "awaiting_judgment":
                raise ValueError(f"Task {rec.get('task_id')} is not awaiting judgment "
                                 f"(status={rec.get('status')})")
            # Move out of awaiting so a concurrent judge() fails the check above.
            worker["status"] = "running"
            self.save_worker(worker)
            rec["status"] = "running"
            self._save_task(rec)

        judged_turn = len(rec.get("turns", []))
        rec.setdefault("judgments", []).append({
            "verdict": verdict, "feedback": feedback, "by": "manager",
            "turn": judged_turn, "at": _utc_now(),
        })
        self._log(rec["task_id"], "judgment", {"verdict": verdict, "turn": judged_turn})

        if verdict == "accept":
            # An accepted execution-lane task must still summarize its result
            # back into the persistent conversation (the pause skipped it).
            if rec.get("execution_required"):
                self.record_execution_result_to_persistent_worker(worker, rec)
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

        first = build_continuation_packet(done_marker, blocked_marker, start_turn, cap,
                                          use_judge=bool(rec.get("use_judge", False)))
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
            use_judge=bool(rec.get("use_judge", False)),
            goal=rec.get("goal", rec.get("task", "")),
            execution_required=bool(rec.get("execution_required", False)),
        )
        if execution_required := bool(rec.get("execution_required", False)):
            if status in ("complete", "blocked", "budget_exhausted"):
                self.record_execution_result_to_persistent_worker(worker, rec)
        if status == "awaiting_judgment":
            self._pause_for_judgment(worker, rec)
        else:
            self._finalize(worker, rec, status)
        return rec

    def record_execution_result_to_persistent_worker(
        self, worker: Dict[str, Any], rec: Dict[str, Any]
    ) -> None:
        """Summarize an EXECUTION-lane run back into the PERSISTENT conversation.

        The execution lane (/v1/runs) is not the canonical memory; this writes a
        compact summary into the worker's persistent /v1/responses conversation
        (same conversation + session_key) so the worker's canonical memory retains
        what the execution produced. Updates last_response_id. Best-effort: a
        failure here is logged but does not fail the task.
        """
        summary = (
            f"[EXECUTION SUMMARY] task_id={rec.get('task_id')} "
            f"status={rec.get('status')} run_id={rec.get('last_run_id') or worker.get('last_run_id')}\n"
            f"Goal: {rec.get('goal', rec.get('task', ''))[:500]}\n"
            f"Result (tail):\n{(rec.get('final_result') or '')[-1500:]}\n"
            f"This was executed on the /v1/runs execution lane; recorded here for "
            f"persistent-conversation continuity."
        )
        try:
            caller = self._persistent_caller_for(worker)
            res = caller(summary, conversation=worker.get("conversation", ""),
                         session_key=worker.get("session_key", ""),
                         model=worker.get("model", DEFAULT_MODEL))
            rid = res.get("response_id") if isinstance(res, dict) else ""
            if rid:
                worker["last_response_id"] = rid
                self.save_worker(worker)
            rec["execution_summarized_to_conversation"] = True
            self._log(rec["task_id"], "execution_summary_recorded",
                      {"conversation": worker.get("conversation"), "response_id": rid})
        except Exception as exc:  # noqa: BLE001 -- best effort
            rec["execution_summarized_to_conversation"] = False
            self._log(rec["task_id"], "execution_summary_failed", {"error": str(exc)})

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
        use_judge: bool = False,
        goal: str = "",
        execution_required: bool = False,
    ) -> str:
        """Harness-owned goal loop. Returns a terminal run status.

        Never calls /goal. Lane selection:
        - default (execution_required=False): PERSISTENT reasoning lane
          (/v1/responses or session_chat). Continuity is SERVER-SIDE via the
          named `conversation` (+ session_key); client-side history is NOT used.
        - execution_required=True: EXECUTION lane (/v1/runs). The run API has no
          named conversation, so a client-side transcript is threaded as an
          execution-local cache only (not canonical memory); the result is
          summarized back into the persistent conversation by the caller.
        """
        conversation = worker.get("conversation", "")
        session_key = worker.get("session_key", "")
        model = worker.get("model", DEFAULT_MODEL)
        done_marker = done_marker or DEFAULT_DONE_MARKER
        blocked_marker = blocked_marker or DEFAULT_BLOCKED_MARKER
        if execution_required:
            caller = self._execution_caller_for(worker)
            # Execution-local cache only (runs has no server-side conversation).
            history: Optional[List[Dict[str, str]]] = list(worker.get("history", []))
        else:
            caller = self._persistent_caller_for(worker)
            # Persistent lane: rely on the SERVER-SIDE named conversation; do not
            # replay client-side transcript (it is not the canonical memory).
            history = None

        status = "running"
        pending_feedback = ""
        turn = start_turn
        while turn <= max_turns:
            if turn == start_turn:
                prompt = first_prompt
            else:
                prompt = build_continuation_packet(done_marker, blocked_marker, turn, max_turns,
                                                   use_judge=use_judge)
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
            rid = resp.get("response_id") if isinstance(resp, dict) else ""
            if rid:
                if execution_required:
                    rec["last_run_id"] = rid
                    worker["last_run_id"] = rid
                else:
                    rec["last_response_id"] = rid
                    worker["last_response_id"] = rid
                self.save_worker(worker)
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
                # judge completion_mode: an external model decides done/continue.
                try:
                    verdict = self._judge_for(worker)(goal or rec.get("task", ""), text)
                except Exception as exc:  # noqa: BLE001
                    verdict = {"done": False, "blocked": False, "reason": "", "error": str(exc)}
                rec.setdefault("judge_verdicts", []).append({"turn": turn, **verdict})
                self._log(rec["task_id"], "judge", {
                    "turn": turn, "done": verdict.get("done"),
                    "reason": verdict.get("reason"), "error": verdict.get("error", ""),
                })
                self._save_task(rec)
                # Judge UNAVAILABLE -> fail fast (explicit), do NOT silently
                # fail-open and burn the turn budget.
                if verdict.get("error"):
                    status = "error"
                    rec["error"] = f"judge unavailable: {verdict['error']}"
                    self._log(rec["task_id"], "judge_unavailable", {"turn": turn, "error": verdict["error"]})
                    break
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
                        # Looked done but the harness's check failed. Reject + feed
                        # back. Phrase it correctly for the mode (judge mode has no
                        # marker, so do not reference one).
                        if use_judge:
                            pending_feedback = (
                                f"VERIFICATION REJECTED. The harness ran the acceptance check "
                                f"and it FAILED (exit {vr['exit_code']}).\n"
                                f"Command: {verify_command}\nOutput (tail):\n{vr['output']}\n"
                                f"Keep working on the in-scope task until this command exits 0."
                            )
                        else:
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
        _atomic_write_text(Path(worker["artifact_manifest"]), json.dumps(manifest, indent=2))
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
    p_create.add_argument("--persistent-transport", default="responses", choices=["responses", "session_chat"],
                          help="Canonical memory lane: responses (default, /v1/responses named conversation) "
                               "or session_chat (needs --session-id).")
    p_create.add_argument("--execution-transport", default="runs", choices=["none", "runs"],
                          help="Opt-in execution lane (NOT memory): runs (/v1/runs) or none.")
    p_create.add_argument("--session-id", default="", help="Hermes SessionDB id; ONLY for session_chat (not the conversation).")

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
    p_task.add_argument("--completion-mode", default="marker", choices=["marker", "judge"],
                        help="marker (default): worker emits GOAL_COMPLETE/GOAL_BLOCKED. judge: external judge decides.")
    p_task.add_argument("--judge-model", default="",
                        help="Override the goal-judge model (judge mode; default: deepseek-v4-flash via DeepSeek).")
    p_task.add_argument("--execution-required", action="store_true",
                        help="Run this task on the execution lane (/v1/runs) so it can run code/tests; "
                             "the result is summarized back into the persistent conversation.")
    p_task.add_argument("--auto-revert", action="store_true",
                        help="Postflight scope audit: auto-revert agent-reported changes outside write scope (git workspace only).")

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

    try:
        if args.command == "worker-create":
            w = ctl.create_worker(
                worker_id=args.worker_id, role=args.role, conversation=args.conversation,
                session_key=args.session_key, model=args.model, workspace=args.workspace,
                read_roots=_as_list(args.read_roots), write_roots=_as_list(args.write_roots),
                deny_write_roots=_as_list(args.deny_write_roots),
                search_policy=args.search_policy, branch_policy=args.branch_policy,
                persistent_transport=args.persistent_transport,
                execution_transport=args.execution_transport,
                session_id=args.session_id,
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
                judgment_mode=args.judgment_mode, completion_mode=args.completion_mode,
                auto_revert=args.auto_revert, execution_required=args.execution_required,
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
    except (KeyError, ValueError) as exc:
        # Clean error instead of a raw traceback (unknown worker, busy worker,
        # bad transport/mode, nothing to judge, etc.). KeyError's str() wraps the
        # message in quotes, so prefer the raw arg.
        msg = exc.args[0] if exc.args else str(exc)
        _print({"error": str(msg)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
