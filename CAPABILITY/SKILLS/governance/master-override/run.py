#!/usr/bin/env python3

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat


def _read_canon_version(project_root: Path) -> str:
    versioning = project_root / "CANON" / "VERSIONING.md"
    if not versioning.exists():
        return "unknown"
    content = versioning.read_text(errors="ignore")
    for line in content.splitlines():
        if "canon_version:" in line:
            return line.split("canon_version:", 1)[1].strip()
    return "unknown"


def _log_path(project_root: Path) -> Path:
    # Return a Path object; formatting to POSIX is done when producing the relative string
    return project_root / "CONTRACTS" / "_runs" / "override_logs" / "master_override.jsonl"


def _append_log(log_path: Path, entry: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _tail_jsonl(log_path: Path, limit: int) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-limit:] if limit > 0 else []
    entries: List[Dict[str, Any]] = []
    for line in tail:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    return entries


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    action = str(payload.get("action", "")).strip().lower()
    token = payload.get("token")
    note = payload.get("note")
    limit = int(payload.get("limit", 20))

    log_path = _log_path(PROJECT_ROOT)
    # Use POSIX separators for the relative log path to match fixture expectations
    log_path_rel = log_path.relative_to(PROJECT_ROOT).as_posix()

    if token != "MASTER_OVERRIDE":
        result = {"ok": False, "action": action or None, "error": "unauthorized", "log_path": log_path_rel}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return 0

    if action == "log":
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "MASTER_OVERRIDE",
            "canon_version": _read_canon_version(PROJECT_ROOT),
            "note": note,
        }
        _append_log(log_path, entry)
        result = {"ok": True, "action": "log", "log_path": log_path_rel}
    elif action == "read":
        entries = _tail_jsonl(log_path, limit)
        result = {"ok": True, "action": "read", "log_path": log_path_rel, "entries": entries}
    else:
        result = {"ok": False, "action": action or None, "error": "unknown_action", "log_path": log_path_rel}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
