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
    entrypoint_substring: str,
) -> str:
    results = cortex_query.find_entities_containing_path(entrypoint_substring)
    if results:
        ordered = sorted(results, key=lambda r: r.get("paths", {}).get("source", ""))
        entry = ordered[0]
        source_path = entry.get("paths", {}).get("source", "")
        if source_path:
            return source_path
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

def ensure_entrypoint_wrapper(entrypoint_path: Path) -> None:
    """
    Ensure the recommended MCP entrypoint wrapper exists.

    CI runs from a clean checkout, so anything under CONTRACTS/_runs/ must be
    created at runtime (it is not checked into git).
    """
    if entrypoint_path.exists():
        return

    entrypoint_path.parent.mkdir(parents=True, exist_ok=True)
    entrypoint_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                '"""Generated runtime entrypoint for AGS MCP server."""',
                "",
                "import sys",
                "from pathlib import Path",
                "",
                "PROJECT_ROOT = Path(__file__).resolve().parents[2]",
                "if str(PROJECT_ROOT) not in sys.path:",
                "    sys.path.insert(0, str(PROJECT_ROOT))",
                "",
                "import MCP.server as mcp_server",
                "",
                "# Redirect MCP audit logs to an allowed output root.",
                'mcp_server.LOGS_DIR = PROJECT_ROOT / \"CONTRACTS\" / \"_runs\" / \"mcp_logs\"',
                "",
                "if __name__ == '__main__':",
                "    mcp_server.main()",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def instructions_for(client: str) -> List[str]:
    client_key = (client or "generic").strip().lower()
    if client_key == "vscode":
        return [
            "Command Palette -> Developer: Reload Window.",
            "If tools still missing: Developer: Restart Extension Host.",
            "Open the extension MCP/server logs and confirm initialize/tools/list succeeds.",
            "Ensure the extension config points to the correct command and entrypoint.",
        ]
    if client_key == "claude":
        return [
            "Quit and relaunch the client to reload MCP config.",
            "Verify the MCP server entry shows as connected in the tools panel.",
            "Confirm initialize/tools/list succeeds in the client logs.",
            "If tools missing, re-check the command/args in the MCP config.",
        ]
    return [
        "Restart or reload the MCP client/extension to pick up config changes.",
        "Verify the MCP server command/args point at the expected entrypoint.",
        "Check client logs for successful initialize/tools/list responses.",
        "If tools are missing, restart again and re-check the config.",
    ]


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = load_json(input_path)

    entrypoint_substring = payload.get("entrypoint_substring", "CONTRACTS/ags_mcp_entrypoint.py")
    args = payload.get("args", ["--test"])
    client = payload.get("client", "generic")

    project_root = Path(__file__).resolve().parents[2]
    cortex_query, error = get_cortex_query(project_root)
    if cortex_query is None:
        write_json(output_path, {
            "ok": False,
            "returncode": 1,
            "entrypoint": Path(entrypoint_substring).as_posix(),
            "args": args,
            "client": client,
            "instructions": instructions_for(client),
        })
        print(error)
        return 1

    entrypoint_rel = find_entrypoint(cortex_query, entrypoint_substring)
    entrypoint_path = project_root / Path(entrypoint_rel)
    ensure_entrypoint_wrapper(entrypoint_path)
    result = run_entrypoint(project_root, entrypoint_rel, args)

    write_json(output_path, {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "entrypoint": Path(entrypoint_rel).as_posix(),
        "args": args,
        "client": client,
        "instructions": instructions_for(client),
    })

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
