#!/usr/bin/env python3
"""
Skill: canon-governance-check

Contract-style wrapper:
  python run.py <input.json> <output.json>

Deterministic fixture support:
- If `changed_files` is provided in input, evaluate deterministically without
  invoking git or node.
- Otherwise, fall back to the node script wrapper (best-effort).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[4]


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("input must be a JSON object")
    return obj


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_bytes(_canonical_json_bytes(obj))


def _deterministic_check(changed_files: List[str]) -> Dict[str, Any]:
    """Minimal, deterministic governance check over explicit file list."""
    significant_prefixes = (
        "CANON/",
        "TOOLS/",
        "PRIMITIVES/",
        "PIPELINES/",
        "SKILLS/",
        ".github/workflows/",
        "SCHEMAS/",
        "CONTEXT/spectrum/",
        "CONTEXT/decisions/",
    )
    requires_changelog = any(any(p.startswith(pref) for pref in significant_prefixes) for p in changed_files)
    has_canon_changelog = "CANON/CHANGELOG.md" in changed_files
    if requires_changelog and not has_canon_changelog:
        return {
            "ok": False,
            "code": "MISSING_CANON_CHANGELOG",
            "details": {"requires_changelog": True},
        }
    return {"ok": True, "code": "OK", "details": {"requires_changelog": requires_changelog}}


def _node_check(verbose: bool) -> Dict[str, Any]:
    cmd = ["node", str(REPO_ROOT / "TOOLS" / "check-canon-governance.js")]
    if verbose:
        cmd.append("--verbose")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {
        "ok": res.returncode == 0,
        "code": "OK" if res.returncode == 0 else "GOVERNANCE_CHECK_FAILED",
        "details": {"exit_code": res.returncode, "stdout": res.stdout, "stderr": res.stderr},
    }


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write("Usage: run.py <input.json> <output.json>\n")
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    inp = _load_json(input_path)
    verbose = bool(inp.get("verbose", False))
    changed_files = inp.get("changed_files")
    if changed_files is not None:
        if not isinstance(changed_files, list) or not all(isinstance(x, str) for x in changed_files):
            raise ValueError("changed_files must be list[str]")
        result = _deterministic_check(changed_files)
    else:
        result = _node_check(verbose)
    _write_json(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

