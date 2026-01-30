#!/usr/bin/env python3
"""
Skill: research-cache

Access and manage the persistent research cache.

Contract-style wrapper:
  python run.py <input.json> <output.json>

Deterministic fixture support:
- If `dry_run` is true, returns mock data
- Otherwise, calls the research_cache.py tool
"""

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
except ImportError:
    GuardedWriter = None


def _load_json(path: Path) -> Dict[str, Any]:
    """Load and validate JSON input."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("input must be a JSON object")
    return obj


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON output using GuardedWriter."""
    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        sys.exit(1)

    writer = GuardedWriter(
        PROJECT_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"]
    )

    rel_path = path.resolve().relative_to(PROJECT_ROOT)
    writer.mkdir_auto(str(rel_path.parent))
    content = json.dumps(obj, indent=2, sort_keys=True)
    writer.write_auto(str(rel_path), content)


def run_cache_command(
    action: str,
    url: Optional[str] = None,
    summary: Optional[str] = None,
    tags: Optional[str] = None,
    tag_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Run a research cache command."""
    cache_script = PROJECT_ROOT / "CAPABILITY" / "TOOLS" / "research_cache.py"

    if not cache_script.exists():
        return {
            "error": f"Research cache script not found: {cache_script}",
            "success": False
        }

    cmd = [sys.executable, str(cache_script)]

    if action == "lookup":
        if not url:
            return {"error": "'url' required for lookup", "success": False}
        cmd.extend(["--lookup", url])

    elif action == "save":
        if not url or not summary:
            return {"error": "'url' and 'summary' required for save", "success": False}
        cmd.extend(["--save", url, summary])
        if tags:
            cmd.extend(["--tags", tags])

    elif action == "list":
        cmd.append("--list")
        if tag_filter:
            cmd.extend(["--filter", tag_filter])

    elif action == "clear":
        cmd.append("--clear")

    else:
        return {"error": f"Invalid action: {action}", "success": False}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace"
        )

        if result.returncode == 0:
            output = result.stdout.strip() if result.stdout else "[OK]"
            return {
                "success": True,
                "action": action,
                "output": output
            }
        else:
            return {
                "success": False,
                "action": action,
                "error": result.stderr.strip() if result.stderr else "Unknown error"
            }
    except Exception as e:
        return {
            "success": False,
            "action": action,
            "error": f"Exception: {str(e)}"
        }


def run_cache_mock(
    action: str,
    mock_entries: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Return mock data for deterministic testing."""
    if action == "lookup":
        return {
            "success": True,
            "action": "lookup",
            "found": False,
            "output": "URL not found in cache"
        }
    elif action == "save":
        return {
            "success": True,
            "action": "save",
            "saved": True,
            "output": "Entry saved to cache"
        }
    elif action == "list":
        entries = mock_entries if mock_entries else []
        return {
            "success": True,
            "action": "list",
            "entries": entries,
            "count": len(entries),
            "output": f"Found {len(entries)} entries"
        }
    elif action == "clear":
        return {
            "success": True,
            "action": "clear",
            "cleared": True,
            "output": "Cache cleared"
        }
    else:
        return {"success": False, "error": f"Invalid action: {action}"}


def main(argv: list) -> int:
    """Main entry point."""
    if len(argv) != 3:
        sys.stderr.write("Usage: run.py <input.json> <output.json>\n")
        return 2

    input_path = Path(argv[1])
    output_path = Path(argv[2])

    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        inp = _load_json(input_path)
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    action = inp.get("action")
    if not action:
        result = {"error": "'action' parameter is required", "success": False}
    else:
        dry_run = inp.get("dry_run", False)

        if dry_run:
            mock_entries = inp.get("_mock_entries")
            result = run_cache_mock(action, mock_entries)
        else:
            url = inp.get("url")
            summary = inp.get("summary")
            tags = inp.get("tags")
            tag_filter = inp.get("filter")

            result = run_cache_command(
                action=action,
                url=url,
                summary=summary,
                tags=tags,
                tag_filter=tag_filter
            )

    # Merge input with result for traceability
    output = {**inp, **result}

    try:
        _write_json(output_path, output)
    except Exception as e:
        print(f"Error writing output: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
