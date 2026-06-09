#!/usr/bin/env python3
"""Hermes Harness external caller.

This script lets any agent, shell, or CI step ask a local Hermes Agent API server
for delegated work. It uses only Python stdlib.

Default endpoint: http://127.0.0.1:8642/v1/chat/completions
Default key env: HERMES_API_KEY or API_SERVER_KEY, fallback change-me-local-dev
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
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BASE = "http://127.0.0.1:8642/v1"
DEFAULT_MODEL = "hermes-agent"
DEFAULT_KEY = "change-me-local-dev"
VALID_MODES = {"auto", "plan", "research", "audit", "code", "debug", "docs", "synthesis"}


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
    try:
        max_workers = int(task.get("max_workers", 3))
        if max_workers < 1 or max_workers > 12:
            errors.append("max_workers must be between 1 and 12. Recommended default is 3.")
    except (TypeError, ValueError):
        errors.append("max_workers must be an integer.")
    return errors


def build_harness_prompt(
    task: str,
    workspace: str = ".",
    mode: str = "auto",
    max_workers: int = 3,
    toolsets: Optional[List[str]] = None,
    constraints: str = "",
    output: str = "Markdown synthesis with delegation summary, findings, changes, verification, uncertainty, and next move.",
) -> str:
    toolsets = toolsets or ["terminal", "file"]
    abs_workspace = str(Path(workspace).expanduser().resolve()) if workspace else "none"
    return textwrap.dedent(f"""
    Use Hermes Agent as the task harness. Load the `hermes-harness` skill if it is installed. If it is not installed, follow this embedded contract exactly.

    PARENT ROLE:
    - You are the coordinator, not the whole workforce.
    - Decompose the task only when the subtasks are independent enough to benefit from delegation.
    - Use `delegate_task` for subtasks that need reasoning, judgment, repo inspection, or research.
    - Keep subagents leaf-only unless nested orchestration is truly necessary.
    - Give every subagent complete context. Subagents know nothing from this parent conversation except what you put in their `goal` and `context`.
    - After workers finish, synthesize. Do not paste raw worker logs as the final answer.

    TASK:
    {task}

    WORKSPACE:
    {abs_workspace}

    MODE:
    {mode}

    MAX CONCURRENT SUBAGENTS:
    {max_workers}

    DEFAULT ALLOWED TOOLSETS:
    {', '.join(toolsets)}

    CONSTRAINTS:
    {constraints or 'No extra constraints provided. Prefer scoped, reversible edits and explicit verification.'}

    OUTPUT CONTRACT:
    {output}

    REQUIRED FINAL SECTIONS:
    # Result
    ## What I delegated
    ## Findings
    ## Changes made
    ## Verification
    ## Conflicts or uncertainty
    ## Next move
    """).strip()


def call_hermes_api(
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    stream: bool = False,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "system",
                "content": "You are Hermes Agent operating as a disciplined subagent orchestration harness. Use available tools and skills, especially delegation, when appropriate.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected Hermes API response: {json.dumps(data, indent=2)[:2000]}") from e


def merged_task_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    task: Dict[str, Any] = {}
    if getattr(args, "task_file", None):
        task.update(load_task_file(args.task_file))
    if getattr(args, "task", None):
        task["task"] = args.task
    if getattr(args, "workspace", None):
        task["workspace"] = args.workspace
    if getattr(args, "mode", None):
        task["mode"] = args.mode
    if getattr(args, "max_workers", None):
        task["max_workers"] = args.max_workers
    if getattr(args, "toolsets", None):
        task["toolsets"] = normalize_toolsets(args.toolsets)
    if getattr(args, "constraints", None):
        task["constraints"] = args.constraints
    if getattr(args, "output_contract", None):
        task["output"] = args.output_contract
    task.setdefault("workspace", ".")
    task.setdefault("mode", "auto")
    task.setdefault("max_workers", 3)
    task.setdefault("toolsets", ["terminal", "file"])
    task.setdefault("constraints", "")
    task.setdefault("output", "Markdown synthesis with delegation summary, findings, changes, verification, uncertainty, and next move.")
    return task


def cmd_prompt(args: argparse.Namespace) -> int:
    task = merged_task_from_args(args)
    errors = validate_task(task)
    if errors:
        print("Invalid task:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 2
    prompt = build_harness_prompt(
        task=task["task"],
        workspace=task.get("workspace", "."),
        mode=task.get("mode", "auto"),
        max_workers=int(task.get("max_workers", 3)),
        toolsets=normalize_toolsets(task.get("toolsets")),
        constraints=task.get("constraints", ""),
        output=task.get("output", ""),
    )
    print(prompt)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    task = merged_task_from_args(args)
    errors = validate_task(task)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 2
    print(json.dumps({"ok": True, "task": task}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    task = merged_task_from_args(args)
    errors = validate_task(task)
    if errors:
        print("Invalid task:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 2

    prompt = build_harness_prompt(
        task=task["task"],
        workspace=task.get("workspace", "."),
        mode=task.get("mode", "auto"),
        max_workers=int(task.get("max_workers", 3)),
        toolsets=normalize_toolsets(task.get("toolsets")),
        constraints=task.get("constraints", ""),
        output=task.get("output", ""),
    )

    base = args.base_url or os.environ.get("HERMES_API_BASE", DEFAULT_BASE)
    key = args.api_key or os.environ.get("HERMES_API_KEY") or os.environ.get("API_SERVER_KEY") or DEFAULT_KEY
    if key == DEFAULT_KEY:
        print("WARNING: Using default insecure API key. Set HERMES_API_KEY or API_SERVER_KEY.", file=sys.stderr)
    model = args.model or os.environ.get("HERMES_MODEL", DEFAULT_MODEL)

    if args.dry_run:
        print(json.dumps({
            "url": base.rstrip("/") + "/chat/completions",
            "model": model,
            "prompt": prompt,
        }, indent=2))
        return 0

    started = time.time()
    result = call_hermes_api(prompt, base, key, model, args.timeout)
    elapsed = time.time() - started

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result + "\n", encoding="utf-8")
        print(f"Wrote Hermes result to {out} ({elapsed:.1f}s)")
    else:
        print(result)
    return 0


def add_common_task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", default=None, help="Task to hand to Hermes Harness.")
    p.add_argument("--task-file", default=None, help="JSON task file.")
    p.add_argument("--workspace", default=None, help="Workspace path passed to Hermes. Default: current directory.")
    p.add_argument("--mode", default=None, choices=sorted(VALID_MODES), help="Routing mode. Default: auto.")
    p.add_argument("--max-workers", type=int, default=None, help="Maximum concurrent subagents to request. Default: 3.")
    p.add_argument("--toolsets", default=None, help="Comma-separated default toolsets, e.g. terminal,file,web.")
    p.add_argument("--constraints", default=None, help="Constraints to include in the handoff.")
    p.add_argument("--output-contract", default=None, help="Custom output contract.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send structured tasks to Hermes Agent as a subagent harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prompt = sub.add_parser("prompt", help="Print the Hermes Harness prompt only.")
    add_common_task_args(p_prompt)
    p_prompt.set_defaults(func=cmd_prompt)

    p_validate = sub.add_parser("validate", help="Validate a task JSON or CLI task.")
    add_common_task_args(p_validate)
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="Call local Hermes API server.")
    add_common_task_args(p_run)
    p_run.add_argument("--base-url", default=None, help=f"Hermes API base. Default env HERMES_API_BASE or {DEFAULT_BASE}")
    p_run.add_argument("--api-key", default=None, help="Hermes API key. Default env HERMES_API_KEY or API_SERVER_KEY.")
    p_run.add_argument("--model", default=None, help=f"Model name. Default {DEFAULT_MODEL}")
    p_run.add_argument("--timeout", type=int, default=900, help="HTTP timeout seconds.")
    p_run.add_argument("--output", help="Write result to file.")
    p_run.add_argument("--dry-run", action="store_true", help="Print payload instead of sending.")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
