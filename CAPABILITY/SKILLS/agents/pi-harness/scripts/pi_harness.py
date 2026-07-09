#!/usr/bin/env python3
"""Pure prompt and Pi command construction for pi-harness."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Mapping

READ_ONLY_TOOLS = ("read", "grep", "find", "ls")
WRITE_TOOLS = ("bash", "edit", "write")
ALL_TOOLS = READ_ONLY_TOOLS + WRITE_TOOLS
MAX_TASK_CHARS = 100_000
MAX_CONSTRAINT_CHARS = 20_000
MAX_RESULT_CHARS = 1_000_000
GOVERNED_SHELL_EXTENSION = Path(__file__).resolve().parents[1] / "extensions" / "governed-shell.ts"


def _items(values: Iterable[str] | str | None) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(",")
    return [str(value).strip() for value in values if str(value).strip()]


def build_task_packet(
    task: str,
    workspace: str,
    read_roots: Iterable[str] | str | None,
    write_roots: Iterable[str] | str | None,
    tools: Iterable[str] | str | None,
    constraints: str = "",
    shell_programs: Mapping[str, object] | None = None,
    manual_context: str = "",
) -> str:
    goal = task.strip()
    if not goal:
        raise ValueError("task is required")
    if len(goal) > MAX_TASK_CHARS:
        raise ValueError(f"task exceeds {MAX_TASK_CHARS} characters")
    if len(constraints) > MAX_CONSTRAINT_CHARS:
        raise ValueError(f"constraints exceed {MAX_CONSTRAINT_CHARS} characters")
    ws = str(Path(workspace).resolve())
    reads = _items(read_roots) or [ws]
    writes = _items(write_roots)
    enabled = _items(tools) or list(READ_ONLY_TOOLS)
    unknown = sorted(set(enabled) - set(ALL_TOOLS))
    if unknown:
        raise ValueError(f"unsupported Pi tools: {', '.join(unknown)}")
    write_enabled = bool(set(enabled) & set(WRITE_TOOLS))
    if write_enabled and not writes:
        raise ValueError("write-capable tools require at least one write root")
    write_text = ", ".join(writes) if writes else "NONE (read-only task)"
    shell_aliases = ", ".join(sorted((shell_programs or {}).keys())) or "NONE"
    extra = constraints.strip() or "None."
    context = manual_context.strip() or "NONE"
    return f"""PI WORKER TASK

GOAL
{goal}

WORKSPACE
{ws}

STRICT SCOPE LOCK
READ_SCOPE: {', '.join(reads)}
WRITE_SCOPE: {write_text}
TOOLS: {', '.join(enabled)}
SHELL_PROGRAMS: {shell_aliases}

RULES
- Work only inside the workspace and the declared scopes.
- Do not modify files outside WRITE_SCOPE.
- Do not create, switch, merge, or delete branches.
- Do not commit, push, publish, or release.
- For bash, use only a SHELL_PROGRAMS alias plus a literal argument array. Never use shell command strings, pipes, redirects, or chaining.
- If blocked, stop and identify the exact blocker.

ADDITIONAL CONSTRAINTS
{extra}

MANUALLY PROVIDED CONTEXT
{context}

FINAL RESPONSE
Return: outcome, findings, created files, modified files, verification commands and results, and blockers.
""".strip()


def build_pi_command(
    worker: Mapping[str, object],
    prompt: str,
    pi_command: str,
    prompt_path: str | None = None,
) -> List[str]:
    command = [
        resolve_executable(pi_command),
        "--mode", "json",
        "--print",
        "--session-id", str(worker["session_id"]),
        "--session-dir", str(worker["session_dir"]),
        "--name", str(worker.get("name") or worker["worker_id"]),
        "--tools", ",".join(_items(worker.get("tools"))),
        "--system-prompt", "",
        "--approve",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
    ]
    provider = str(worker.get("provider") or "").strip()
    model = str(worker.get("model") or "").strip()
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["--model", model])
    if not bool(worker.get("allow_extensions", False)):
        command.append("--no-extensions")
    if bool(worker.get("allow_shell", False)):
        if not GOVERNED_SHELL_EXTENSION.is_file():
            raise ValueError(f"governed shell extension is missing: {GOVERNED_SHELL_EXTENSION}")
        command.extend(["--extension", str(GOVERNED_SHELL_EXTENSION)])
    if prompt_path:
        prompt_file = Path(prompt_path).resolve()
        if not prompt_file.is_file():
            raise ValueError(f"Pi prompt file not found: {prompt_file}")
        command.append(f"@{prompt_file}")
    else:
        command.append(prompt)
    return command


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_executable(command: str) -> str:
    value = command.strip()
    if not value:
        raise ValueError("Pi command is empty")
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ValueError(f"Pi executable not found: {resolved}")
        return str(resolved)
    discovered = shutil.which(value)
    if not discovered:
        raise ValueError(f"Pi executable not found on PATH: {value}")
    return str(Path(discovered).resolve())


def inspect_jsonl(
    jsonl_text: str,
    workspace: str,
    write_roots: Iterable[str] | str | None,
    shell_programs: Mapping[str, object] | None = None,
) -> dict:
    roots = [Path(value).resolve() for value in _items(write_roots)]
    ws = Path(workspace).resolve()
    result = ""
    malformed_lines = 0
    settled = False
    completed = False
    terminal_events: List[str] = []
    stop_reasons: List[str] = []
    write_paths: List[str] = []
    scope_violations: List[str] = []
    shell_used = False
    shell_policy_violations: List[str] = []
    allowed_programs = set((shell_programs or {}).keys())

    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if not isinstance(event, dict):
            malformed_lines += 1
            continue
        event_type = event.get("type")
        if event_type == "agent_settled":
            settled = True
            completed = True
            terminal_events.append("agent_settled")
        elif event_type == "agent_end":
            completed = True
            terminal_events.append("agent_end")
        if event.get("type") == "tool_execution_start":
            tool = event.get("toolName")
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            if tool == "bash":
                shell_used = True
                program = args.get("program")
                raw_cwd = args.get("cwd", ".")
                if not isinstance(program, str) or program not in allowed_programs:
                    shell_policy_violations.append("bash:program-not-allowlisted")
                if not isinstance(raw_cwd, str):
                    shell_policy_violations.append("bash:invalid-cwd")
                else:
                    shell_cwd = (ws / raw_cwd).resolve()
                    if not (shell_cwd == ws or shell_cwd.is_relative_to(ws)):
                        shell_policy_violations.append(f"bash:cwd-escape:{shell_cwd}")
            if tool in {"edit", "write"}:
                raw_path = args.get("path") or args.get("file") or args.get("filePath")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    scope_violations.append(f"{tool}:missing-path")
                else:
                    candidate = Path(raw_path)
                    if not candidate.is_absolute():
                        candidate = ws / candidate
                    candidate = candidate.resolve()
                    normalized = str(candidate)
                    write_paths.append(normalized)
                    if not roots or not any(candidate == root or candidate.is_relative_to(root) for root in roots):
                        scope_violations.append(normalized)
        if event.get("type") != "message_end":
            continue
        message = event.get("message", {})
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        reason = message.get("stopReason")
        if isinstance(reason, str):
            stop_reasons.append(reason)
        chunks = []
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
        if chunks:
            result = "".join(chunks)

    full_result = result
    result_too_large = len(full_result) > MAX_RESULT_CHARS
    if result_too_large:
        result = full_result[:MAX_RESULT_CHARS]
    bad_stop = any(reason in {"error", "aborted", "length"} for reason in stop_reasons)
    integrity_ok = (
        malformed_lines == 0
        and not scope_violations
        and not bad_stop
        and completed
        and bool(full_result.strip())
        and not result_too_large
        and not shell_policy_violations
    )
    return {
        "integrity_ok": integrity_ok,
        "malformed_jsonl_lines": malformed_lines,
        "agent_settled": settled,
        "agent_completed": completed,
        "terminal_events": terminal_events,
        "stop_reasons": stop_reasons,
        "write_paths": sorted(set(write_paths)),
        "scope_violations": sorted(set(scope_violations)),
        "shell_used": shell_used,
        "shell_policy_violations": sorted(set(shell_policy_violations)),
        "shell_scope_verifiable": not shell_used,
        "scope_enforcement": "observed edit/write tool paths",
        "result_chars": len(full_result),
        "result_truncated": result_too_large,
        "result": result,
        "result_sha256": sha256_text(full_result),
    }


def extract_last_assistant_text(jsonl_text: str) -> str:
    return inspect_jsonl(jsonl_text, ".", [str(Path.cwd())])["result"]
