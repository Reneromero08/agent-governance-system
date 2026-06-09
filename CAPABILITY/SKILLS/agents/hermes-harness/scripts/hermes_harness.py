#!/usr/bin/env python3
"""Hermes Harness external caller.

Sends tasks to a local Hermes Agent API server using the /v1/responses endpoint
with named conversations for persistent multi-turn context.

Default endpoint: http://127.0.0.1:8642/v1
Default key env: HERMES_API_KEY or API_SERVER_KEY, fallback change-me-local-dev

Session model:
    conversation="worker-name"  → Hermes auto-chains to the latest stored
                                  response in that named conversation.
    store: true                 → persists across gateway restarts (LRU, 100 max).
    X-Hermes-Session-Key        → stable long-term memory scope per project/worker.

Modes:
    persistent_worker           → named conversation, no delegate_task.
                                  Continue from prior context directly.
    all other modes             → may use delegate_task for decomposition.

Without a conversation name each call is stateless (fresh turn).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BASE = "http://127.0.0.1:8642/v1"
DEFAULT_MODEL = "hermes-agent"
DEFAULT_KEY = "change-me-local-dev"
VALID_MODES = {
    "auto", "plan", "research", "audit", "code", "debug", "docs", "synthesis",
    "persistent_worker",
}

WORKER_PROMPT = textwrap.dedent("""
ACT ON THIS TASK DIRECTLY. Do not mention skills or tools you do not have.

TASK:
{task}

WORKSPACE: {workspace}
MODE: {mode}
CONSTRAINTS: {constraints}

You are the persistent worker for this named conversation.
Continue from prior context in this conversation.
Do NOT spawn delegate_task unless explicitly requested.
Perform the task directly. Answer concisely.

After completing, include a brief output contract section:
{output}
""").strip()

ORCHESTRATOR_PROMPT = textwrap.dedent("""
ACT ON THIS TASK DIRECTLY. Do not mention skills or tools you do not have.

TASK:
{task}

WORKSPACE: {workspace}
MODE: {mode}
CONSTRAINTS: {constraints}

If the task benefits from decomposition, split it and use delegate_task with up to {max_workers} workers.
If the task is atomic, answer directly. Do not fabricate delegation where none is needed.

After completing, include a brief output contract section:
{output}
""").strip()


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def load_task_file(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Task file must contain a JSON object.")
    return data


def normalize_toolsets(value: Optional[str | List[str]]) -> List[str]:
    if value is None:
        return ["terminal", "file"]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in value.split(",") if v.strip()]


def validate_task(task: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not str(task.get("task", "")).strip():
        errors.append("Missing required field: task")
    mode = str(task.get("mode", "auto"))
    if mode not in VALID_MODES:
        errors.append(f"Invalid mode {mode!r}. Valid modes: {', '.join(sorted(VALID_MODES))}")
    if task.get("conversation_new") and not str(task.get("conversation", "")).strip():
        errors.append("conversation_new requires a conversation name")
    return errors


def build_harness_prompt(
    task: str,
    workspace: str = ".",
    mode: str = "auto",
    max_workers: int = 3,
    constraints: str = "",
    output: str = "",
) -> str:
    abs_workspace = str(Path(workspace).expanduser().resolve()) if workspace else "none"
    constraints_block = constraints or 'No extra constraints provided.'
    output_block = output or (
        "Markdown synthesis with delegation summary, findings, changes, "
        "verification, uncertainty, and next move."
    )

    if max_workers <= 0 or mode == "persistent_worker":
        template = WORKER_PROMPT
    else:
        template = ORCHESTRATOR_PROMPT

    return template.format(
        task=task, workspace=abs_workspace, mode=mode,
        max_workers=max_workers, constraints=constraints_block,
        output=output_block,
    )


# ---------------------------------------------------------------------------
# API call using /v1/responses with named conversations
# ---------------------------------------------------------------------------

def call_hermes_responses(
    prompt: str,
    conversation: str = "",
    session_key: str = "",
    store: bool = False,
    base_url: str = DEFAULT_BASE,
    api_key: str = DEFAULT_KEY,
    model: str = DEFAULT_MODEL,
    timeout: Optional[int] = None,
    instructions: str = "",
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/responses"
    payload: Dict[str, Any] = {
        "model": model,
        "input": prompt,
    }
    if conversation:
        payload["conversation"] = conversation
        payload["store"] = store or True
    if instructions:
        payload["instructions"] = instructions

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if session_key:
        headers["X-Hermes-Session-Key"] = session_key

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hermes API HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Hermes API at {url}: {e}") from e

    data = json.loads(body)
    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            texts = [c["text"] for c in item.get("content", []) if c.get("type") == "output_text"]
            text = "\n".join(texts)
            break

    return {
        "text": text or json.dumps(data, indent=2),
        "response_id": data.get("id", ""),
        "usage": data.get("usage", {}),
        "raw": data,
    }


# ---------------------------------------------------------------------------
# High-level run (used by both CLI and run.py skill path)
# ---------------------------------------------------------------------------

def run_task(
    task: str,
    workspace: str = ".",
    mode: str = "auto",
    max_workers: int = 3,
    constraints: str = "",
    output_contract: str = "",
    conversation: str = "",
    conversation_new: bool = False,
    session_key: str = "",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    base = base_url or os.environ.get("HERMES_API_BASE", DEFAULT_BASE)
    key = api_key or os.environ.get("HERMES_API_KEY") or os.environ.get("API_SERVER_KEY") or DEFAULT_KEY
    mdl = model or os.environ.get("HERMES_MODEL", DEFAULT_MODEL)

    if conversation_new and not conversation:
        raise ValueError("conversation_new requires a conversation name.")

    conv = conversation
    if conv and conversation_new:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        conv = f"{conversation}_{ts}"

    prompt = build_harness_prompt(
        task=task, workspace=workspace, mode=mode,
        max_workers=max_workers, constraints=constraints,
        output=output_contract,
    )

    api_result = call_hermes_responses(
        prompt=prompt, conversation=conv,
        session_key=session_key, store=bool(conv),
        base_url=base, api_key=key, model=mdl, timeout=timeout,
    )
    result: Dict[str, Any] = {
        "ok": True, "task": task, "mode": mode,
        "result": api_result["text"],
        "response_id": api_result["response_id"],
        "usage": api_result["usage"],
        "raw": api_result["raw"],
    }
    if conversation_new:
        result["conversation"] = conv
        result["conversation_new"] = True
    elif conv:
        result["conversation"] = conv
    return result


# ---------------------------------------------------------------------------
# CLI argument handling
# ---------------------------------------------------------------------------

def merged_task_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    task: Dict[str, Any] = {}
    if getattr(args, "task_file", None):
        task.update(load_task_file(args.task_file))
    for field in ("task", "workspace", "mode", "max_workers", "constraints",
                  "output_contract", "conversation", "session_key"):
        val = getattr(args, field, None)
        if val is not None:
            task[field] = val
    if getattr(args, "conversation_new", None):
        task["conversation_new"] = True
    if getattr(args, "toolsets", None):
        task["toolsets"] = normalize_toolsets(args.toolsets)
    if "output" in task and "output_contract" not in task:
        task["output_contract"] = task["output"]
    task.setdefault("workspace", ".")
    task.setdefault("mode", "auto")
    task.setdefault("max_workers", 3)
    return task


def cmd_prompt(args: argparse.Namespace) -> int:
    t = merged_task_from_args(args)
    errors = validate_task(t)
    if errors:
        print("Invalid task:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 2
    prompt = build_harness_prompt(
        task=t["task"], workspace=t.get("workspace", "."), mode=t.get("mode", "auto"),
        max_workers=int(t.get("max_workers", 3)),
        constraints=t.get("constraints", ""), output=t.get("output_contract", ""),
    )
    print(prompt)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    t = merged_task_from_args(args)
    errors = validate_task(t)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 2
    print(json.dumps({"ok": True, "task": t}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    t = merged_task_from_args(args)

    if getattr(args, "dry_run", False):
        errors = validate_task(t)
        if errors:
            print("Invalid task:\n- " + "\n- ".join(errors), file=sys.stderr)
            return 2
        prompt = build_harness_prompt(
            task=t.get("task", ""), workspace=t.get("workspace", "."),
            mode=t.get("mode", "auto"), max_workers=int(t.get("max_workers", 3)),
            constraints=t.get("constraints", ""), output=t.get("output_contract", ""),
        )
        conv = t.get("conversation", "")
        if t.get("conversation_new"):
            conv = f"{conv}_DRYRUN_TIMESTAMP"
        dry_base = getattr(args, "base_url", None) or os.environ.get("HERMES_API_BASE", DEFAULT_BASE)
        dry_model = getattr(args, "model", None) or os.environ.get("HERMES_MODEL", DEFAULT_MODEL)
        dry = {
            "endpoint": f"{dry_base.rstrip('/')}/responses",
            "model": dry_model,
            "conversation": conv or None,
            "session_key": t.get("session_key") or None,
            "prompt": prompt,
        }
        print(json.dumps(dry, indent=2))
        return 0

    errors = validate_task(t)
    if errors:
        print("Invalid task:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 2

    result = run_task(
        task=t["task"],
        workspace=t.get("workspace", "."),
        mode=t.get("mode", "auto"),
        max_workers=int(t.get("max_workers", 3)),
        constraints=t.get("constraints", ""),
        output_contract=t.get("output_contract", ""),
        conversation=t.get("conversation", ""),
        conversation_new=t.get("conversation_new", False),
        session_key=t.get("session_key", ""),
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
        model=getattr(args, "model", None),
        timeout=getattr(args, "timeout", None),
    )

    if getattr(args, "output", None):
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote result to {out}")
    else:
        print(result.get("result", json.dumps(result, indent=2)))
    return 0


def add_common_task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", default=None, help="Task to hand to Hermes Harness.")
    p.add_argument("--task-file", default=None, help="JSON task file.")
    p.add_argument("--workspace", default=None, help="Workspace path. Default: current directory.")
    p.add_argument("--mode", default=None, choices=sorted(VALID_MODES), help="Routing mode. Default: auto.")
    p.add_argument("--max-workers", type=int, default=None, help="Max concurrent subagents. Default: 3.")
    p.add_argument("--constraints", default=None, help="Constraints to include in the handoff.")
    p.add_argument("--output-contract", default=None, help="Custom output contract.")
    p.add_argument("--conversation", default=None, help="Named conversation for persistent multi-turn context.")
    p.add_argument("--conversation-new", action="store_true", help="Create a new unique conversation name (appends timestamp).")
    p.add_argument("--session-key", default=None, help="X-Hermes-Session-Key for long-term memory scoping.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send structured tasks to Hermes Agent via /v1/responses.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prompt = sub.add_parser("prompt", help="Print the Hermes Harness prompt only.")
    add_common_task_args(p_prompt)
    p_prompt.set_defaults(func=cmd_prompt)

    p_validate = sub.add_parser("validate", help="Validate a task JSON or CLI task.")
    add_common_task_args(p_validate)
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="Call local Hermes API server (/v1/responses).")
    add_common_task_args(p_run)
    p_run.add_argument("--base-url", default=None, help="API base URL.")
    p_run.add_argument("--api-key", default=None, help="API key.")
    p_run.add_argument("--model", default=None, help="Model label (cosmetic — actual model is server-side).")
    p_run.add_argument("--timeout", type=int, default=None, help="HTTP timeout in seconds.")
    p_run.add_argument("--output", help="Write result to file.")
    p_run.add_argument("--dry-run", action="store_true", help="Print payload without calling the API.")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
