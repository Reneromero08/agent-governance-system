#!/usr/bin/env python3
"""Hermes async-run transport with harness-side approval handling.

Why this exists
---------------
The /v1/responses transport (hermes_harness.call_hermes_responses) is a single
blocking POST. If the agent invokes an approval-gated tool (execute_code,
terminal, ...) under `approvals.mode: manual`, the run emits an
`approval.request` and waits. A blocking client cannot answer it, so the tool
call times out "pending approval" and the agent can never run code or tests.

The async run API solves this:
    POST /v1/runs                       -> {run_id}  (returns immediately)
    GET  /v1/runs/{run_id}/events       -> SSE: message.delta / approval.request
                                            / run.completed / run.failed
    POST /v1/runs/{run_id}/approval     -> {choice, all}  resolve the approval

This module starts a run, streams its events, and AUTO-ANSWERS every
`approval.request` so the agent can execute its own code autonomously.

IMPORTANT: the default choice is "once" (approve this single tool call only).
Do NOT default to "always" or "session": "always" PERMANENTLY writes the tool
into the server's `command_allowlist` (a global, cross-platform approval
loosening persisted to config.yaml); "session" persists for the session scope.
"once" leaves ZERO persistent state -- each approval is answered individually
and nothing is written to the server config. The global `approvals.mode` is
never touched either, so the CLI / Telegram / etc. keep manual approval.

This is portable: it uses only documented public endpoints (listed in the API
server route table), not internal server state. No Hermes config change and no
gateway restart required.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

DEFAULT_BASE = "http://127.0.0.1:8643/v1"
DEFAULT_KEY = "change-me-local-dev"
VALID_CHOICES = {"once", "session", "always", "deny"}


def _headers(api_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return h


def _post(url: str, payload: Dict[str, Any], api_key: str, session_key: str = "",
          timeout: Optional[int] = 60) -> Dict[str, Any]:
    extra = {"X-Hermes-Session-Key": session_key} if session_key else None
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=_headers(api_key, extra), method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _approve(base: str, run_id: str, api_key: str, choice: str, resolve_all: bool) -> None:
    url = f"{base.rstrip('/')}/runs/{run_id}/approval"
    try:
        _post(url, {"choice": choice, "all": resolve_all, "resolve_all": resolve_all},
              api_key, timeout=30)
    except urllib.error.HTTPError as e:
        # 409 "no pending approval" is benign (already resolved / raced).
        if e.code != 409:
            raise


def call_hermes_run(
    prompt: str,
    *,
    session_key: str = "",
    instructions: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    previous_response_id: str = "",
    approve_choice: str = "once",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = 600,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run one agent turn via the async run API, auto-answering approvals.

    Returns {text, usage, run_id, approvals, events_seen, error}.
    """
    base = (base_url or os.environ.get("HERMES_API_BASE", DEFAULT_BASE)).rstrip("/")
    key = (api_key or os.environ.get("HERMES_API_KEY")
           or os.environ.get("API_SERVER_KEY") or DEFAULT_KEY)
    if approve_choice not in VALID_CHOICES:
        raise ValueError(f"approve_choice must be one of {sorted(VALID_CHOICES)}")

    body: Dict[str, Any] = {"input": prompt}
    if instructions:
        body["instructions"] = instructions
    if conversation_history:
        body["conversation_history"] = conversation_history
    if previous_response_id:
        body["previous_response_id"] = previous_response_id

    started = _post(f"{base}/runs", body, key, session_key=session_key, timeout=60)
    run_id = started.get("run_id")
    if not run_id:
        return {"text": "", "usage": {}, "run_id": "", "approvals": 0,
                "events_seen": 0, "error": f"no run_id in start response: {started}"}

    # Stream events. Each event is a single `data: {json}` SSE line.
    events_url = f"{base}/runs/{run_id}/events"
    req = urllib.request.Request(events_url, headers=_headers(key), method="GET")

    text_parts: List[str] = []
    final_output = ""
    usage: Dict[str, Any] = {}
    approvals = 0
    events_seen = 0
    error = ""

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line or line.startswith(":"):
                # blank line (event boundary) or SSE comment / keepalive
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            events_seen += 1
            if on_event:
                try:
                    on_event(event)
                except Exception:
                    pass
            etype = event.get("event") or event.get("type") or ""

            if etype == "approval.request":
                approvals += 1
                # Answer in a worker thread so a slow approval POST never
                # stalls reading the event stream.
                threading.Thread(
                    target=_approve, args=(base, run_id, key, approve_choice, True),
                    daemon=True,
                ).start()
            elif etype == "message.delta":
                d = event.get("delta")
                if d:
                    text_parts.append(d)
            elif etype == "run.completed":
                final_output = event.get("output", "") or ""
                usage = event.get("usage", {}) or {}
            elif etype == "run.failed":
                error = event.get("error", "agent run failed")
            # run.cancelled / tool progress events: ignored for output purposes.

    text = final_output or "".join(text_parts)
    return {
        "text": text, "usage": usage, "run_id": run_id,
        "approvals": approvals, "events_seen": events_seen,
        "error": error, "response_id": run_id,
    }


# ---------------------------------------------------------------------------
# Goal judge -- faithful replica of Hermes /goal's per-turn judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a goal-completion judge for an autonomous agent loop. You are given "
    "a GOAL and the agent's most recent response. Decide whether the goal is "
    "satisfied. Reply with STRICT JSON ONLY, no prose: "
    '{"done": true|false, "reason": "<one sentence>"}. '
    "Mark done=true ONLY when the response explicitly confirms the goal is "
    "complete, the final deliverable is clearly produced, or the goal is "
    "genuinely unachievable/blocked (done=true with the block reason). "
    "Otherwise done=false. Be conservative: prefer false when unsure."
)

def _balanced_objects(s: str):
    """Yield top-level balanced {...} substrings, ignoring braces inside strings."""
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield s[start:i + 1]
                start = None


def _extract_verdict(content: str) -> Dict[str, Any]:
    """Find the verdict JSON in a judge response robustly (prose, code fences, and
    brace-heavy text around it). Prefer the LAST balanced object with a 'done' key.
    """
    found: Dict[str, Any] = {}
    for candidate in _balanced_objects(content):
        try:
            d = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(d, dict) and "done" in d:
            found = d  # keep the last one (judges usually put the verdict last)
    return found


def call_hermes_judge(
    goal: str,
    last_response: str,
    *,
    model: str = "hermes-agent",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = 60,
) -> Dict[str, Any]:
    """Ask an auxiliary model: is the goal done? Returns {done, reason}.

    Mirrors Hermes /goal: conservative JSON judge, FAIL-OPEN (any error ->
    done=False so a broken judge never wedges the loop; the turn budget is the
    real backstop).
    """
    base = (base_url or os.environ.get("HERMES_API_BASE", DEFAULT_BASE)).rstrip("/")
    key = (api_key or os.environ.get("HERMES_API_KEY")
           or os.environ.get("API_SERVER_KEY") or DEFAULT_KEY)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"GOAL:\n{goal}\n\nAGENT'S LAST RESPONSE:\n{last_response[-4000:]}"},
        ],
        "stream": False,
        "temperature": 0,
    }
    try:
        data = _post(f"{base}/chat/completions", payload, key, timeout=timeout)
        content = data["choices"][0]["message"]["content"]
        verdict = _extract_verdict(content)
        done = verdict.get("done", False)
        # Coerce robustly: a judge may return the string "false"/"true" instead
        # of a JSON bool, and bool("false") is True -- that would falsely complete.
        if isinstance(done, str):
            done = done.strip().lower() in ("true", "yes", "1", "done")
        return {"done": bool(done),
                "reason": str(verdict.get("reason", "")) or "(no reason)",
                "error": ""}
    except Exception as exc:  # noqa: BLE001
        # Surface the error so callers decide policy (the goal loop fails fast in
        # judge mode rather than silently continuing and burning the budget).
        return {"done": False, "reason": "", "error": f"judge unavailable: {exc}"}


if __name__ == "__main__":
    # Cheap self-test: ask the agent to run a trivial calculation via its
    # code-execution tool. Proves approval auto-answer + real execution.
    import sys
    from pathlib import Path

    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        for ln in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k = k.strip().upper(); v = v.strip().strip('"').strip("'")
            if k in ("API_SERVER_KEY", "HERMES_API_KEY"):
                os.environ.setdefault("HERMES_API_KEY", v)

    seen_types: List[str] = []
    res = call_hermes_run(
        "Use your code execution tool to compute 6*7 and reply with exactly: ANSWER=<result>. "
        "Do not write any files.",
        session_key="agent:ags:run-transport-selftest",
        on_event=lambda e: seen_types.append(e.get("event", "?")),
        timeout=240,
    )
    print("run_id      :", res["run_id"])
    print("approvals   :", res["approvals"])
    print("events_seen :", res["events_seen"])
    print("event types :", sorted(set(seen_types)))
    print("usage       :", res["usage"])
    print("error       :", res["error"] or "(none)")
    print("text        :", repr(res["text"][:300]))
    ok = ("42" in res["text"]) and not res["error"]
    print("SELFTEST:", "PASS" if ok else "INCONCLUSIVE")
    sys.exit(0 if ok else 1)
