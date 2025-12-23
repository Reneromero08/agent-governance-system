#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_cortex_query(project_root: Path) -> Tuple[Any, str]:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import CORTEX.query as cortex_query

    try:
        cortex_query.get_metadata("canon_version")
        return cortex_query, ""
    except FileNotFoundError:
        build_script = project_root / "CORTEX" / "cortex.build.py"
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, str(build_script)],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
        )
        if result.returncode != 0:
            return None, (result.stdout + result.stderr).strip()

        try:
            cortex_query.get_metadata("canon_version")
            return cortex_query, ""
        except FileNotFoundError as exc:
            return None, str(exc)


def find_entrypoint(
    cortex_query: Any,
    project_root: Path,
    entrypoint_substring: str,
) -> str:
    results = cortex_query.find_entities_containing_path(entrypoint_substring)
    if results:
        ordered = sorted(results, key=lambda r: r.get("paths", {}).get("source", ""))
        entry = ordered[0]
        source_path = entry.get("paths", {}).get("source", "")
        if source_path:
            return source_path
    # Fallback: use the provided relative path (no filesystem scan).
    return Path(entrypoint_substring).as_posix()


def run_entrypoint(project_root: Path, entrypoint_rel: str, args: List[str]) -> subprocess.CompletedProcess:
    entrypoint_path = project_root / Path(entrypoint_rel)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(entrypoint_path), *args],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
    )


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = load_json(input_path)

    entrypoint_substring = payload.get("entrypoint_substring", "CONTRACTS/_runs/ags_mcp_entrypoint.py")
    args = payload.get("args", ["--test"])

    project_root = Path(__file__).resolve().parents[2]
    cortex_query, error = get_cortex_query(project_root)
    if cortex_query is None:
        write_json(output_path, {
            "ok": False,
            "returncode": 1,
            "entrypoint": Path(entrypoint_substring).as_posix(),
            "args": args,
        })
        print(error)
        return 1

    entrypoint_rel = find_entrypoint(cortex_query, project_root, entrypoint_substring)
    result = run_entrypoint(project_root, entrypoint_rel, args)

    write_json(output_path, {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "entrypoint": Path(entrypoint_rel).as_posix(),
        "args": args,
    })

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
