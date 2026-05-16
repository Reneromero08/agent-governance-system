#!/usr/bin/env python3

"""
CodeRabbit Comments Skill - retrieves CodeRabbit review comments from
the VS Code extension's local storage.

Auto-discovers the storage path by scanning AppData\\Code\\User\\workspaceStorage\\
for the workspace matching the current repo root. Alternatively accepts an
explicit storage_path input.

ADR-021 compliant: all operations are deterministic and logged.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat
except ImportError:
    def ensure_canon_compat(skill_dir):
        return True


def _find_storage_path(repo_path: Path) -> Optional[Path]:
    """Auto-discover the CodeRabbit review database file.
    
    Prioritizes workspace.json URI match, but falls back to scanning
    all workspace directories for the most recent coderabbit storage.
    """
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None

    ws_root = Path(appdata) / "Code" / "User" / "workspaceStorage"
    if not ws_root.exists():
        return None

    # Build both URI variants (VSCode uses %5C for Windows backslashes)
    import urllib.parse
    repo_str = str(repo_path.resolve())
    # Variant 1: forward slashes
    posix = repo_str.replace("\\", "/")
    uri_fwd = "file:///" + urllib.parse.quote(posix, safe="/")
    if uri_fwd[8].isalpha():
        uri_fwd = uri_fwd[:8] + uri_fwd[8].lower() + uri_fwd[9:]
    # Variant 2: keep backslashes (VSCode on Windows)
    uri_bs = "file:///" + urllib.parse.quote(repo_str, safe="/\\")
    if uri_bs[8].isalpha():
        uri_bs = uri_bs[:8] + uri_bs[8].lower() + uri_bs[9:]

    candidates = []  # (mod_time, db_path) for fallback
    
    for ws_dir in ws_root.iterdir():
        if not ws_dir.is_dir():
            continue
        ws_json = ws_dir / "workspace.json"
        if ws_json.exists():
            try:
                ws_data = json.loads(ws_json.read_text(encoding="utf-8"))
                folder = ws_data.get("folder", "")
                if folder in (uri_fwd, uri_bs):
                    cr_dir = ws_dir / "coderabbit.coderabbit-vscode"
                    if cr_dir.exists():
                        db = _find_db_in_cr_dir(cr_dir)
                        if db:
                            return db
            except Exception:
                pass
        # Fallback: check for coderabbit storage regardless of match
        cr_dir = ws_dir / "coderabbit.coderabbit-vscode"
        if cr_dir.exists():
            db = _find_db_in_cr_dir(cr_dir)
            if db:
                candidates.append((db.stat().st_mtime, db))
    
    # Fallback: use most recently modified coderabbit database
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    return None


def _find_db_in_cr_dir(cr_dir: Path) -> Optional[Path]:
    """Find the review database JSON in a coderabbit extension directory."""
    best = None
    best_sz = 0
    for f in cr_dir.iterdir():
        if f.suffix == ".json" and f.name != "categories.json":
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    sz = f.stat().st_size
                    if sz > best_sz:
                        best = f
                        best_sz = sz
            except Exception:
                continue
    return best


def _extract_actionable(comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and simplify comments for agent consumption."""
    result = []
    for c in comments:
        severity = c.get("severity", "none")
        if severity == "none":
            continue
        result.append({
            "filename": c.get("filename", ""),
            "startLine": c.get("startLine"),
            "endLine": c.get("endLine"),
            "severity": severity,
            "type": c.get("type", ""),
            "comment": c.get("comment", "").strip(),
            "suggestions": c.get("suggestions", []),
            "resolution": c.get("resolution", ""),
            "codegenInstructions": c.get("codegenInstructions", "").strip(),
        })
    return result


def _action_latest(db_path: Path, file_filter: Optional[str] = None) -> Dict[str, Any]:
    """Return the latest completed review with actionable comments.
    
    If file_filter is provided, only return comments for files matching the pattern.
    """
    try:
        reviews = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"Failed to parse review database: {exc}"}

    completed = [r for r in reviews if r.get("status") == "completed"]
    if not completed:
        return {"ok": True, "reviews": 0, "latest_review": None}

    latest = completed[-1]
    file_review_map = latest.get("fileReviewMap", {})

    all_comments = []
    files = []
    for filename, entry in file_review_map.items():
        if file_filter and file_filter.lower() not in filename.lower():
            continue
        files.append(filename)
        comments = entry.get("comments", [])
        all_comments.extend(_extract_actionable(comments))

    return {
        "ok": True,
        "review_count": len(reviews),
        "latest_review": {
            "id": latest.get("id", ""),
            "startedAt": latest.get("startedAt", ""),
            "endedAt": latest.get("endedAt", ""),
            "comments": all_comments,
            "file_count": len(files),
            "files": sorted(files),
        }
    }


def _action_list(db_path: Path) -> Dict[str, Any]:
    """List all reviews with basic info."""
    try:
        reviews = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"Failed to parse review database: {exc}"}

    result = []
    for r in reviews:
        file_review_map = r.get("fileReviewMap", {})
        result.append({
            "id": r.get("id", ""),
            "status": r.get("status", ""),
            "startedAt": r.get("startedAt", ""),
            "endedAt": r.get("endedAt", ""),
            "file_count": len(file_review_map),
        })

    return {
        "ok": True,
        "reviews": result,
    }


def _action_all(db_path: Path) -> Dict[str, Any]:
    """Return all completed reviews with actionable comments."""
    try:
        reviews = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"Failed to parse review database: {exc}"}

    completed_actions = []
    for r in reviews:
        if r.get("status") != "completed":
            continue
        file_review_map = r.get("fileReviewMap", {})
        all_comments = []
        files = []
        for filename, entry in file_review_map.items():
            files.append(filename)
            all_comments.extend(_extract_actionable(entry.get("comments", [])))

        if all_comments:
            completed_actions.append({
                "id": r.get("id", ""),
                "startedAt": r.get("startedAt", ""),
                "comments": all_comments,
                "file_count": len(files),
            })

    return {
        "ok": True,
        "reviews": completed_actions,
    }


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    action = str(payload.get("action", "latest")).strip().lower()
    storage_path = payload.get("storage_path")
    file_filter = payload.get("file_filter", None)

    if storage_path:
        db_path = Path(storage_path)
        if not db_path.exists():
            result = {"ok": False, "error": f"storage_path not found: {db_path}"}
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
            return 0
    else:
        discovered = _find_storage_path(PROJECT_ROOT)
        if discovered is None:
            result = {"ok": False, "error": "CodeRabbit storage not found. Use the VS Code extension first to generate reviews."}
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
            return 0
        db_path = discovered

    if action == "latest":
        result = _action_latest(db_path, file_filter)
    elif action == "list":
        result = _action_list(db_path)
    elif action == "all":
        result = _action_all(db_path)
    else:
        result = {"ok": False, "error": f"Unknown action: {action}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
