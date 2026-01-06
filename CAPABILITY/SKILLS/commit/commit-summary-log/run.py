#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat


try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
DEFAULT_LOG_PATH = "LAW/CONTRACTS/_runs/commit_logs/commit_summaries.jsonl"
ALLOWED_TYPES = {"feat", "fix", "docs", "chore", "refactor", "test"}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any], writer: Any) -> None:
    writer.mkdir_durable(str(path.parent))
    writer.write_durable(str(path), json.dumps(payload, indent=2, sort_keys=True))


def run_git(args: List[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=str(PROJECT_ROOT), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def git_commit_entry(commit_ref: str, note: Optional[str], include_body: bool) -> Dict[str, Any]:
    subject = run_git(["show", "-s", "--format=%s", commit_ref]).strip()
    author_date = run_git(["show", "-s", "--format=%aI", commit_ref]).strip()
    committer_date = run_git(["show", "-s", "--format=%cI", commit_ref]).strip()

    commit_hash = run_git(["rev-parse", commit_ref]).strip()
    files = [line.strip() for line in run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_ref]).splitlines() if line.strip()]

    body = None
    if include_body:
        raw_body = run_git(["show", "-s", "--format=%b", commit_ref])
        body = raw_body.rstrip("\n") if raw_body else ""

    entry: Dict[str, Any] = {
        "commit": commit_hash,
        "subject": subject,
        "author_date": author_date,
        "committer_date": committer_date,
        "files": files,
    }
    if body is not None:
        entry["body"] = body
    if note:
        entry["note"] = str(note)
    return entry


def assert_allowed_log_path(log_path: Path) -> None:
    rel = log_path.resolve().relative_to(PROJECT_ROOT.resolve())
    rel_posix = rel.as_posix()
    if not (rel_posix == "LAW/CONTRACTS/_runs" or rel_posix.startswith("LAW/CONTRACTS/_runs/")):
        raise ValueError("log_path must be under LAW/CONTRACTS/_runs/")


def append_jsonl(path: Path, entry: Dict[str, Any], writer: Any) -> None:
    # Use GuardedWriter for atomic write (simulating append)
    content = ""
    if path.exists():
        content = path.read_text(encoding="utf-8")
        
    line = json.dumps(entry, ensure_ascii=True)
    content += line + "\n"
    
    writer.mkdir_durable(str(path.parent))
    writer.write_durable(str(path), content)


def normalize_commit_line_part(value: str) -> str:
    return str(value).strip().lower()


def normalize_subject(subject: str) -> str:
    subject = str(subject).strip()
    if subject.endswith("."):
        subject = subject[:-1].rstrip()
    return subject.lower()


def validate_commit_message(message: str) -> None:
    message = str(message)
    if "\n" in message or "\r" in message:
        raise ValueError("commit message must be single-line")
    if len(message) > 50:
        raise ValueError("commit message must be <= 50 characters")
    if message != message.lower():
        raise ValueError("commit message must be lowercase (no capitalization)")
    if message.endswith("."):
        raise ValueError("commit message must not end with a period")
    if ":" not in message:
        raise ValueError("commit message must contain ':' separator")


def build_commit_message(
    *,
    type_: str,
    scope: str,
    subject: str,
    normalize: bool,
) -> str:
    if normalize:
        type_ = normalize_commit_line_part(type_)
        scope = normalize_commit_line_part(scope)
        subject = normalize_subject(subject)
    else:
        type_ = str(type_).strip()
        scope = str(scope).strip()
        subject = str(subject).strip()

    if type_ not in ALLOWED_TYPES:
        raise ValueError(f"type must be one of: {sorted(ALLOWED_TYPES)}")
    if not scope or any(ch.isspace() for ch in scope) or "/" in scope:
        raise ValueError("scope must be a short, non-empty token (no spaces, no slashes)")
    if not subject:
        raise ValueError("subject must be non-empty")

    message = f"{type_}({scope}): {subject}"
    validate_commit_message(message)
    return message


def staged_paths() -> List[str]:
    lines = run_git(["diff", "--cached", "--name-only"]).splitlines()
    return [line.strip() for line in lines if line.strip()]


def main() -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = load_json(input_path)

    action = str(payload.get("action", "log")).strip().lower()

    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1

    writer = GuardedWriter(
        project_root=PROJECT_ROOT,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "NAVIGATION/CORTEX/_generated",
            "MEMORY/LLM_PACKER/_packs",
            "BUILD"
        ]
    )
    writer.open_commit_gate()

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
        write_json(output_path, result, writer)
        return 0

    if action == "template":
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
        write_json(output_path, {"ok": True, "message": message, "warnings": warnings}, writer)
        return 0

    raise ValueError("action must be 'log' or 'template'")


if __name__ == "__main__":
    raise SystemExit(main())
