#!/usr/bin/env python3
"""
Commit Manager - Unified commit operations skill.

Consolidates: commit-queue, commit-summary-log, artifact-escape-hatch

Operations:
  - queue: Manage commit queue (enqueue/list/process)
  - summarize: Generate commit summaries or message templates
  - recover: Emergency artifact recovery (escape hatch check)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None

DURABLE_ROOTS = [
    "LAW/CONTRACTS/_runs",
    "NAVIGATION/CORTEX/_generated",
    "MEMORY/LLM_PACKER/_packs",
    "BUILD"
]

QUEUE_ROOT = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "commit_queue"
DEFAULT_LOG_PATH = "LAW/CONTRACTS/_runs/commit_logs/commit_summaries.jsonl"
ALLOWED_TYPES = {"feat", "fix", "docs", "chore", "refactor", "test"}

ALLOWED_ARTIFACT_ROOTS = {
    PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs",
    PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "_generated",
    PROJECT_ROOT / "MEMORY" / "LLM_PACKER" / "_packs",
    PROJECT_ROOT / "BUILD",
}

IGNORE_PATTERNS = {".git", "__pycache__", ".gitkeep", "node_modules", "fixtures", "schemas", "_packs"}


def get_writer() -> GuardedWriter:
    """Get a configured GuardedWriter instance."""
    if not GuardedWriter:
        raise RuntimeError("GuardedWriter not available")
    writer = GuardedWriter(project_root=PROJECT_ROOT, durable_roots=DURABLE_ROOTS)
    writer.open_commit_gate()
    return writer


def write_output(output_path: Path, data: Dict[str, Any], writer: GuardedWriter) -> None:
    """Write JSON output using GuardedWriter."""
    writer.mkdir_durable(str(output_path.parent))
    writer.write_durable(str(output_path), json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_git(args: List[str]) -> str:
    """Run a git command and return output."""
    out = subprocess.check_output(["git", *args], cwd=str(PROJECT_ROOT), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


# ============================================================================
# Operation: queue
# ============================================================================

def _normalize_queue_id(queue_id: str) -> str:
    if not isinstance(queue_id, str) or not queue_id.strip():
        raise ValueError("QUEUE_ID_INVALID")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    queue_id = queue_id.strip()
    if any(ch not in allowed for ch in queue_id):
        raise ValueError("QUEUE_ID_INVALID")
    return queue_id


def _queue_path(queue_id: str) -> Path:
    return QUEUE_ROOT / f"{queue_id}.jsonl"


def _load_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                events.append(obj)
        except Exception:
            continue
    return events


def _append_event(path: Path, event: Dict[str, Any], writer: GuardedWriter) -> None:
    content = ""
    if path.exists():
        content = path.read_text(encoding="utf-8")
    line = json.dumps(event, sort_keys=True, separators=(",", ":"))
    content += line + "\n"
    writer.mkdir_durable(str(path.parent))
    writer.write_durable(str(path), content)


def _entry_id(entry: Dict[str, Any]) -> str:
    payload = {
        "message": entry.get("message"),
        "files": entry.get("files"),
        "author": entry.get("author"),
        "notes": entry.get("notes"),
        "created_at": entry.get("created_at"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _materialize(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for ev in events:
        if ev.get("type") == "enqueue":
            eid = ev.get("id")
            if isinstance(eid, str) and eid:
                entries[eid] = {
                    "id": eid,
                    "message": ev.get("message"),
                    "files": ev.get("files"),
                    "author": ev.get("author"),
                    "notes": ev.get("notes"),
                    "created_at": ev.get("created_at"),
                    "status": "pending",
                    "error": None,
                }
                order.append(eid)
        elif ev.get("type") == "result":
            ref_id = ev.get("ref_id")
            if isinstance(ref_id, str) and ref_id in entries:
                entries[ref_id]["status"] = ev.get("status")
                entries[ref_id]["error"] = ev.get("error")
    return [entries[eid] for eid in order if eid in entries]


def _validate_files(files: List[str]) -> List[str]:
    if not isinstance(files, list) or not files:
        raise ValueError("FILES_REQUIRED")
    out: List[str] = []
    for raw in files:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("FILES_INVALID")
        if raw.startswith("/") or raw.startswith("\\"):
            raise ValueError("FILES_INVALID")
        if ".." in Path(raw).parts:
            raise ValueError("FILES_INVALID")
        out.append(raw)
    return out


def _stage_files(files: List[str]) -> Optional[str]:
    cmd = ["git", "add", "--"] + files
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        return (res.stderr or res.stdout).strip() or "GIT_ADD_FAILED"
    return None


def op_queue(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Manage commit queue."""
    action = payload.get("action")
    queue_id = _normalize_queue_id(payload.get("queue_id", "default"))
    queue_path = _queue_path(queue_id)

    out: Dict[str, Any] = {"ok": False, "queue_id": queue_id, "action": action}

    if action == "enqueue":
        entry = payload.get("entry")
        if not isinstance(entry, dict):
            out["error"] = "ENTRY_REQUIRED"
        else:
            try:
                files = _validate_files(entry.get("files", []))
            except ValueError as e:
                out["error"] = str(e)
                write_output(output_path, out, writer)
                return 0

            message = entry.get("message")
            if not isinstance(message, str) or not message.strip():
                out["error"] = "MESSAGE_REQUIRED"
            else:
                created_at = entry.get("created_at")
                if not isinstance(created_at, str) or not created_at.strip():
                    out["error"] = "CREATED_AT_REQUIRED"
                else:
                    eid = entry.get("id")
                    if not isinstance(eid, str) or not eid.strip():
                        eid = _entry_id(entry)
                    event = {
                        "type": "enqueue",
                        "id": eid,
                        "message": message,
                        "files": files,
                        "author": entry.get("author"),
                        "notes": entry.get("notes"),
                        "created_at": created_at,
                        "status": "pending",
                    }
                    _append_event(queue_path, event, writer)
                    out.update({"ok": True, "entry_id": eid, "status": "pending"})

    elif action == "list":
        max_items = payload.get("max_items")
        if max_items is not None and not isinstance(max_items, int):
            out["error"] = "MAX_ITEMS_INVALID"
        else:
            events = _load_events(queue_path)
            entries = _materialize(events)
            if isinstance(max_items, int) and max_items > 0:
                entries = entries[:max_items]
            out.update({"ok": True, "entries": entries})

    elif action == "process":
        dry_run = bool(payload.get("dry_run", False))
        events = _load_events(queue_path)
        entries = _materialize(events)
        target = next((e for e in entries if e.get("status") == "pending"), None)
        if not target:
            out.update({"ok": True, "status": "empty"})
        else:
            try:
                files = _validate_files(target.get("files", []))
            except ValueError as e:
                out["error"] = str(e)
                write_output(output_path, out, writer)
                return 0

            error = None
            status = "staged"
            if not dry_run:
                error = _stage_files(files)
                if error:
                    status = "error"
            result_event = {
                "type": "result",
                "ref_id": target.get("id"),
                "status": status,
                "error": error,
            }
            _append_event(queue_path, result_event, writer)
            out.update({
                "ok": error is None,
                "status": status,
                "staged": files if error is None else [],
                "error": error,
            })
    else:
        out["error"] = "ACTION_INVALID"

    write_output(output_path, out, writer)
    return 0


# ============================================================================
# Operation: summarize
# ============================================================================

def git_commit_entry(commit_ref: str, note: Optional[str], include_body: bool) -> Dict[str, Any]:
    """Build a commit entry from git."""
    subject = run_git(["show", "-s", "--format=%s", commit_ref]).strip()
    author_date = run_git(["show", "-s", "--format=%aI", commit_ref]).strip()
    committer_date = run_git(["show", "-s", "--format=%cI", commit_ref]).strip()
    commit_hash = run_git(["rev-parse", commit_ref]).strip()
    files = [line.strip() for line in run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_ref]).splitlines() if line.strip()]

    entry: Dict[str, Any] = {
        "commit": commit_hash,
        "subject": subject,
        "author_date": author_date,
        "committer_date": committer_date,
        "files": files,
    }
    if include_body:
        raw_body = run_git(["show", "-s", "--format=%b", commit_ref])
        entry["body"] = raw_body.rstrip("\n") if raw_body else ""
    if note:
        entry["note"] = str(note)
    return entry


def assert_allowed_log_path(log_path: Path) -> None:
    rel = log_path.resolve().relative_to(PROJECT_ROOT.resolve())
    rel_posix = rel.as_posix()
    if not (rel_posix == "LAW/CONTRACTS/_runs" or rel_posix.startswith("LAW/CONTRACTS/_runs/")):
        raise ValueError("log_path must be under LAW/CONTRACTS/_runs/")


def append_jsonl(path: Path, entry: Dict[str, Any], writer: GuardedWriter) -> None:
    content = ""
    if path.exists():
        content = path.read_text(encoding="utf-8")
    line = json.dumps(entry, ensure_ascii=True)
    content += line + "\n"
    writer.mkdir_durable(str(path.parent))
    writer.write_durable(str(path), content)


def build_commit_message(*, type_: str, scope: str, subject: str, normalize: bool) -> str:
    """Build a conventional commit message."""
    if normalize:
        type_ = str(type_).strip().lower()
        scope = str(scope).strip().lower()
        subject = str(subject).strip().lower()
        if subject.endswith("."):
            subject = subject[:-1].rstrip()
    else:
        type_ = str(type_).strip()
        scope = str(scope).strip()
        subject = str(subject).strip()

    if type_ not in ALLOWED_TYPES:
        raise ValueError(f"type must be one of: {sorted(ALLOWED_TYPES)}")
    if not scope or any(ch.isspace() for ch in scope) or "/" in scope:
        raise ValueError("scope must be a short, non-empty token")
    if not subject:
        raise ValueError("subject must be non-empty")

    message = f"{type_}({scope}): {subject}"

    if "\n" in message or "\r" in message:
        raise ValueError("commit message must be single-line")
    if len(message) > 50:
        raise ValueError("commit message must be <= 50 characters")
    return message


def staged_paths() -> List[str]:
    lines = run_git(["diff", "--cached", "--name-only"]).splitlines()
    return [line.strip() for line in lines if line.strip()]


def op_summarize(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Generate commit summaries or message templates."""
    action = str(payload.get("action", "log")).strip().lower()

    if action == "log":
        mode = str(payload.get("mode", "git")).strip().lower()
        append = bool(payload.get("append", True))
        include_body = bool(payload.get("include_body", False))
        log_path_str = str(payload.get("log_path", DEFAULT_LOG_PATH))
        log_path = (PROJECT_ROOT / log_path_str).resolve()
        assert_allowed_log_path(log_path)

        if mode == "manual":
            if "entry" not in payload or not isinstance(payload["entry"], dict):
                raise ValueError("mode=manual requires object field 'entry'")
            entry = payload["entry"]
        elif mode == "git":
            commit_ref = str(payload.get("commit", "")).strip()
            if not commit_ref:
                raise ValueError("mode=git requires 'commit'")
            note = payload.get("note")
            entry = git_commit_entry(commit_ref=commit_ref, note=note, include_body=include_body)
        else:
            raise ValueError("mode must be 'git' or 'manual'")

        if append:
            append_jsonl(log_path, entry, writer)

        result = {
            "ok": True,
            "append": append,
            "log_path": str(Path(log_path_str).as_posix()),
            "entry": entry,
        }
        write_output(output_path, result, writer)
        return 0

    elif action == "template":
        type_ = payload.get("type")
        scope = payload.get("scope")
        subject = payload.get("subject")
        if type_ is None or scope is None or subject is None:
            raise ValueError("action=template requires fields: type, scope, subject")

        normalize = bool(payload.get("normalize", True))
        warn_if_changelog_missing = bool(payload.get("warn_if_changelog_missing", True))

        warnings: List[str] = []
        if warn_if_changelog_missing:
            paths = staged_paths()
            if paths and "CANON/CHANGELOG.md" not in paths:
                warnings.append("staged changes detected but CANON/CHANGELOG.md is not staged")

        message = build_commit_message(type_=str(type_), scope=str(scope), subject=str(subject), normalize=normalize)
        write_output(output_path, {"ok": True, "message": message, "warnings": warnings}, writer)
        return 0

    else:
        raise ValueError("action must be 'log' or 'template'")


# ============================================================================
# Operation: recover (artifact escape hatch)
# ============================================================================

def is_allowed_path(path: Path) -> bool:
    for root in ALLOWED_ARTIFACT_ROOTS:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def should_ignore(path: Path) -> bool:
    return any(part in IGNORE_PATTERNS for part in path.parts)


def is_runtime_artifact(path: Path) -> bool:
    RUNTIME_DIRS = {"_runs", "_generated", "_packs"}
    if any(part in RUNTIME_DIRS for part in path.parts):
        return True
    if path.suffix == ".log":
        return True
    return False


def _git_untracked_files(project_root: Path) -> List[Path]:
    try:
        res = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=str(project_root),
            capture_output=True,
        )
        if res.returncode != 0:
            return []
        out: List[Path] = []
        for raw in res.stdout.split(b"\0"):
            if not raw:
                continue
            rel = os.fsdecode(raw)
            out.append((project_root / rel).resolve())
        return out
    except Exception:
        return []


def find_escaped_artifacts_fast(project_root: Path) -> List[Path]:
    escaped: List[Path] = []
    candidates = _git_untracked_files(project_root)
    for path in candidates:
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if not is_file:
            continue
        if should_ignore(path):
            continue
        if is_runtime_artifact(path) and not is_allowed_path(path):
            escaped.append(path)
    return escaped


def op_recover(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Check for escaped artifacts."""
    escaped = find_escaped_artifacts_fast(PROJECT_ROOT)

    result = {
        **payload,
        "escaped_artifacts": [str(p.relative_to(PROJECT_ROOT)) for p in escaped],
        "escape_check_passed": len(escaped) == 0,
    }
    write_output(output_path, result, writer)

    if escaped:
        print(f"Found {len(escaped)} escaped artifact(s)")
        return 1

    print("Artifact escape hatch check passed")
    return 0


# ============================================================================
# Main dispatcher
# ============================================================================

OPERATIONS = {
    "queue": op_queue,
    "summarize": op_summarize,
    "recover": op_recover,
}


def main(input_path: Path, output_path: Path) -> int:
    """Main entry point."""
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    operation = payload.get("operation")
    if not operation:
        print("Error: 'operation' field is required")
        return 1

    if operation not in OPERATIONS:
        print(f"Error: Unknown operation '{operation}'. Valid: {', '.join(OPERATIONS.keys())}")
        return 1

    try:
        writer = get_writer()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    return OPERATIONS[operation](payload, output_path, writer)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
