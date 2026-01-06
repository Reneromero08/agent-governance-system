#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json_guarded(path: Path, payload: Dict[str, Any], writer: Any) -> None:
    writer.mkdir_durable(str(path.parent))
    writer.write_durable(str(path), json.dumps(payload, indent=2))


def get_cortex_query(project_root: Path) -> Tuple[Any, str]:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import NAVIGATION.CORTEX.semantic.query as cortex_query

    try:
        cortex_query.get_metadata("canon_version")
        return cortex_query, ""
    except FileNotFoundError:
        build_script = project_root / "NAVIGATION" / "CORTEX" / "db" / "cortex.build.py"
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
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) if not existing_pythonpath else f"{project_root}{os.pathsep}{existing_pythonpath}"
    return subprocess.run(
        [sys.executable, str(entrypoint_path), *args],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
    )

def ensure_entrypoint_wrapper(entrypoint_path: Path, writer: Any) -> None:
    """
    Ensure the recommended MCP entrypoint wrapper exists.

    CI runs from a clean checkout, so anything under CONTRACTS/_runs/ must be
    created at runtime (it is not checked into git).
    """
    if entrypoint_path.exists():
        return

    writer.mkdir_durable(str(entrypoint_path.parent))
    writer.write_durable(str(entrypoint_path), "\n".join(
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
                "import CAPABILITY.MCP.server as mcp_server",
                "",
                "# Redirect MCP audit logs to an allowed output root.",
                'mcp_server.LOGS_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "mcp_logs"',
                "",
                "if __name__ == '__main__':",
                "    mcp_server.main()",
                "",
            ]
        ) + "\n")


def run_bridge_smoke(project_root: Path, config_path: str, payload: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    bridge_config_path = Path(config_path)
    if not bridge_config_path.is_absolute():
        bridge_config_path = project_root / bridge_config_path

    try:
        config = json.loads(bridge_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"enabled": True, "ok": False, "error": f"CONFIG_READ_FAILED: {exc}"}

    host = str(config.get("connect_host", "127.0.0.1"))
    port = int(config.get("port", 8765))
    token = str(config.get("token", ""))

    url = f"http://{host}:{port}/run"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token and token != "CHANGE_ME":
        headers["X-Bridge-Token"] = token

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return {"enabled": True, "ok": False, "error": f"HTTP_ERROR: {exc.code} {exc.reason}"}
    except urllib.error.URLError as exc:
        return {"enabled": True, "ok": False, "error": f"CONNECTION_ERROR: {exc}"}

    try:
        result = json.loads(raw)
    except Exception:
        return {"enabled": True, "ok": False, "error": "INVALID_JSON", "raw": raw}

    return {
        "enabled": True,
        "ok": bool(result.get("ok", False)),
        "exit_code": result.get("exit_code"),
        "error": result.get("error"),
    }


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = load_json(input_path)

    entrypoint_substring = payload.get("entrypoint_substring", "LAW/CONTRACTS/ags_mcp_entrypoint.py")
    args = payload.get("args", ["--test"])
    bridge_smoke = payload.get("bridge_smoke", {})
    bridge_enabled = bool(bridge_smoke.get("enabled", False))

    project_root = Path(__file__).resolve().parents[4]

    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1

    writer = GuardedWriter(project_root, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
    writer.open_commit_gate()

    cortex_query, error = get_cortex_query(project_root)
    if cortex_query is None:
        write_json_guarded(output_path, {
            "ok": False,
            "returncode": 1,
            "entrypoint": Path(entrypoint_substring).as_posix(),
            "args": args,
            "bridge_smoke": {"enabled": bridge_enabled, "ok": False, "error": "CORTEX_UNAVAILABLE"},
        }, writer=writer)
        print(error)
        return 1

    entrypoint_rel = find_entrypoint(cortex_query, project_root, entrypoint_substring)
    entrypoint_path = project_root / Path(entrypoint_rel)
    ensure_entrypoint_wrapper(entrypoint_path, writer)
    result = run_entrypoint(project_root, entrypoint_rel, args)

    bridge_result = {"enabled": bridge_enabled, "ok": True}
    if bridge_enabled:
        config_path = bridge_smoke.get("config_path", "CAPABILITY/MCP/powershell_bridge_config.json")
        command = bridge_smoke.get("command", "Get-Date")
        cwd = bridge_smoke.get("cwd")
        timeout_seconds = int(bridge_smoke.get("timeout_seconds", 30))
        bridge_payload: Dict[str, Any] = {"command": command}
        if isinstance(cwd, str) and cwd.strip():
            bridge_payload["cwd"] = cwd
        bridge_result = run_bridge_smoke(project_root, config_path, bridge_payload, timeout_seconds)

    write_json_guarded(output_path, {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "entrypoint": Path(entrypoint_rel).as_posix(),
        "args": args,
        "bridge_smoke": bridge_result,
    }, writer=writer)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
