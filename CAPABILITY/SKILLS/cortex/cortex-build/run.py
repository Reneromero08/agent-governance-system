#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BUILD_SCRIPT = "NAVIGATION/CORTEX/db/cortex.build.py"
DEFAULT_SECTION_INDEX = "NAVIGATION/CORTEX/_generated/SECTION_INDEX.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git_head_timestamp(project_root: Path) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    value = res.stdout.strip()
    return value or None


def _load_section_index(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    payload = _load_json(input_path)
    expected_paths = payload.get("expected_paths") or []
    timeout_sec = int(payload.get("timeout_sec", 120))
    build_script = Path(payload.get("build_script") or DEFAULT_BUILD_SCRIPT)
    section_index = Path(payload.get("section_index_path") or DEFAULT_SECTION_INDEX)

    errors: List[str] = []
    missing_paths: List[str] = []

    build_path = PROJECT_ROOT / build_script
    if not build_path.exists():
        errors.append(f"build_script_not_found: {build_script.as_posix()}")

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    head_ts = _git_head_timestamp(PROJECT_ROOT)
    if head_ts:
        env["CORTEX_BUILD_TIMESTAMP"] = head_ts

    returncode = 1
    if not errors:
        try:
            result = subprocess.run(
                [sys.executable, str(build_path)],
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            returncode = result.returncode
            if result.returncode != 0:
                stderr = (result.stdout or "") + (result.stderr or "")
                errors.append("build_failed")
                if stderr.strip():
                    errors.append(stderr.strip())
        except subprocess.TimeoutExpired:
            errors.append("build_timeout")
        except OSError as exc:
            errors.append(f"build_os_error: {exc}")

    index_path = PROJECT_ROOT / section_index
    if not errors and not index_path.exists():
        errors.append(f"section_index_missing: {section_index.as_posix()}")

    if not errors:
        try:
            entries = _load_section_index(index_path)
            indexed_paths = {str(entry.get("path", "")) for entry in entries}
            for expected in expected_paths:
                if expected not in indexed_paths:
                    missing_paths.append(expected)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"section_index_read_failed: {exc}")

    ok = not errors and not missing_paths and returncode == 0
    output = {
        "ok": ok,
        "returncode": returncode,
        "section_index_path": section_index.as_posix(),
        "missing_paths": missing_paths,
        "errors": errors,
    }
    _write_json(output_path, output)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
