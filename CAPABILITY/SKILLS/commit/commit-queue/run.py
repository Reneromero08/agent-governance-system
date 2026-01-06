#!/usr/bin/env python3

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat  # type: ignore

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

QUEUE_ROOT = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "commit_queue"


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


def _append_event(path: Path, event: Dict[str, Any], writer: Any) -> None:
    # Use GuardedWriter for atomic write (simulating append)
    # This is inefficient for large files but compliant with firewall policy.
    content = ""
    if path.exists():
        content = path.read_text(encoding="utf-8")
    
    line = json.dumps(event, sort_keys=True, separators=(",", ":"))
    content += line + "\n"
    
    # Ensure directory created via writer
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


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    action = payload.get("action")
    queue_id = _normalize_queue_id(payload.get("queue_id", "default"))
    queue_path = _queue_path(queue_id)

    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1

    writer = GuardedWriter(
        project_root=PROJECT_ROOT,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "NAVIGATION/CORTEX/_generated",
            "MEMORY/LLM_PACKER/_packs",
            "BUILD"  # Add other potential roots if needed, but queue uses LAW/CONTRACTS/_runs
        ]
    )
    writer.open_commit_gate()

    out: Dict[str, Any] = {
        "ok": False,
        "queue_id": queue_id,
        "action": action,
    }

    if action == "enqueue":
        entry = payload.get("entry")
        if not isinstance(entry, dict):
            out["error"] = "ENTRY_REQUIRED"
        else:
            files = _validate_files(entry.get("files", []))
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
            files = _validate_files(target.get("files", []))
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

    # Write output
    try:
        writer.mkdir_durable(str(output_path.parent))
        writer.write_durable(str(output_path), json.dumps(out, sort_keys=True, separators=(",", ":")))
    except Exception as e:
        print(f"Error writing output: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
