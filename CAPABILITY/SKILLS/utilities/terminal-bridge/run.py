#!/usr/bin/env python3
"""
terminal-bridge skill entrypoint.

Unified terminal bridge supporting two distinct servers:
1. AGS PowerShell Bridge (port 8765) - For Google Antigravity MCP
2. VSCode Antigravity Bridge (port 4000) - For spawning VSCode terminals
"""

import json
import sys
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request, error

# Configuration
AGS_BRIDGE_CONFIG_PATH = "CAPABILITY/MCP/powershell_bridge_config.json"
AGS_BRIDGE_SCRIPT_PATH = "CAPABILITY/MCP/powershell_bridge.ps1"
AGS_BRIDGE_DEFAULT_PORT = 8765

VSCODE_BRIDGE_DEFAULT_PORT = 4000
VSCODE_BRIDGE_URL_TEMPLATE = "http://127.0.0.1:{port}/terminal"

# Setup imports
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
except ImportError:
    GuardedWriter = None
    PROJECT_ROOT = Path.cwd()


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON from file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any], writer: Any = None) -> None:
    """Write JSON to file, using GuardedWriter only for paths in durable roots."""
    content = json.dumps(payload, indent=2, sort_keys=True)

    if writer:
        try:
            rel_path = str(path.resolve().relative_to(writer.project_root))
            # Check if path is in a durable root
            is_durable = any(rel_path.startswith(root) for root in writer.durable_roots)
            if is_durable:
                writer.write_auto(rel_path, content)
                return
        except (ValueError, AttributeError):
            pass

    # Fallback: direct write for non-durable paths
    path.write_text(content, encoding="utf-8")


def get_ags_config() -> Dict[str, Any]:
    """Load AGS PowerShell bridge configuration."""
    config_path = PROJECT_ROOT / AGS_BRIDGE_CONFIG_PATH
    if config_path.exists():
        return load_json(config_path)
    return {
        "connect_host": "127.0.0.1",
        "port": AGS_BRIDGE_DEFAULT_PORT,
        "token": "CHANGE_ME"
    }


def check_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error, OSError):
        return False


def check_ags_bridge_status() -> Dict[str, Any]:
    """Check AGS PowerShell bridge status."""
    config = get_ags_config()
    host = config.get("connect_host", "127.0.0.1")
    port = config.get("port", AGS_BRIDGE_DEFAULT_PORT)

    is_reachable = check_port_open(host, port)
    config_exists = (PROJECT_ROOT / AGS_BRIDGE_CONFIG_PATH).exists()
    script_exists = (PROJECT_ROOT / AGS_BRIDGE_SCRIPT_PATH).exists()

    return {
        "server": "ags",
        "name": "AGS PowerShell Bridge (Google Antigravity)",
        "host": host,
        "port": port,
        "reachable": is_reachable,
        "config_exists": config_exists,
        "script_exists": script_exists,
        "ready": is_reachable and config_exists and script_exists
    }


def check_vscode_bridge_status(port: int = VSCODE_BRIDGE_DEFAULT_PORT) -> Dict[str, Any]:
    """Check VSCode Antigravity bridge status."""
    host = "127.0.0.1"
    is_reachable = check_port_open(host, port)

    return {
        "server": "vscode",
        "name": "VSCode Antigravity Bridge",
        "host": host,
        "port": port,
        "reachable": is_reachable,
        "ready": is_reachable
    }


def execute_ags_command(
    command: str,
    cwd: Optional[str] = None,
    timeout_seconds: int = 30
) -> Dict[str, Any]:
    """Execute command via AGS PowerShell bridge."""
    config = get_ags_config()
    host = config.get("connect_host", "127.0.0.1")
    port = config.get("port", AGS_BRIDGE_DEFAULT_PORT)
    token = config.get("token", "")

    url = f"http://{host}:{port}/run"
    payload = {"command": command}
    if cwd:
        payload["cwd"] = cwd

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token and token != "CHANGE_ME":
        headers["X-Bridge-Token"] = token

    req = request.Request(url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            result = json.loads(raw)
            return {
                "ok": result.get("ok", False),
                "output": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "exit_code": result.get("exitCode"),
                "raw": result
            }
    except error.HTTPError as exc:
        return {
            "ok": False,
            "error": f"HTTP error: {exc.code} {exc.reason}"
        }
    except (error.URLError, socket.timeout) as exc:
        return {
            "ok": False,
            "error": f"Connection error: {exc}"
        }
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "Invalid JSON response from bridge"
        }


def launch_vscode_terminal(
    name: str,
    cwd: Optional[str] = None,
    initial_command: Optional[str] = None,
    port: int = VSCODE_BRIDGE_DEFAULT_PORT
) -> Dict[str, Any]:
    """Launch a named terminal in VSCode via Antigravity Bridge."""
    url = VSCODE_BRIDGE_URL_TEMPLATE.format(port=port)

    payload = {"name": name}
    if cwd:
        payload["cwd"] = cwd
    if initial_command:
        payload["initialCommand"] = initial_command

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = request.Request(url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            result = json.loads(raw)
            return {
                "ok": result.get("status") == "success",
                "message": result.get("message", ""),
                "raw": result
            }
    except error.HTTPError as exc:
        return {
            "ok": False,
            "error": f"HTTP error: {exc.code} {exc.reason}"
        }
    except (error.URLError, socket.timeout) as exc:
        return {
            "ok": False,
            "error": f"Connection error: {exc}. Is VSCode running with antigravity-bridge extension?"
        }
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "Invalid JSON response from VSCode bridge"
        }


def get_setup_info(server: Optional[str] = None) -> Dict[str, Any]:
    """Get setup information for bridge servers."""
    info = {
        "ags": {
            "name": "AGS PowerShell Bridge (Google Antigravity)",
            "port": AGS_BRIDGE_DEFAULT_PORT,
            "config_path": AGS_BRIDGE_CONFIG_PATH,
            "script_path": AGS_BRIDGE_SCRIPT_PATH,
            "instructions": [
                "1. Update token in CAPABILITY/MCP/powershell_bridge_config.json",
                "2. Start bridge: powershell -ExecutionPolicy Bypass -File CAPABILITY\\MCP\\powershell_bridge.ps1",
                "3. Test: POST http://127.0.0.1:8765/run with X-Bridge-Token header",
                "4. Use MCP tool 'terminal_bridge' or this skill with server='ags'"
            ],
            "mcp_tool": "mcp__ags__terminal_bridge"
        },
        "vscode": {
            "name": "VSCode Antigravity Bridge",
            "port": VSCODE_BRIDGE_DEFAULT_PORT,
            "extension_path": "D:/CCC 2.0/AI/AGI/EXTENSIONS/antigravity-bridge/antigravity-bridge-0.1.0.vsix",
            "instructions": [
                "1. Install extension: code --install-extension antigravity-bridge-0.1.0.vsix",
                "2. Open VSCode - extension auto-starts HTTP server on port 4000",
                "3. Test: POST http://127.0.0.1:4000/terminal with {name, cwd, initialCommand}",
                "4. Use this skill with server='vscode' and operation='launch_terminal'"
            ],
            "api_endpoint": "POST http://127.0.0.1:4000/terminal"
        }
    }

    if server and server in info:
        return {"servers": {server: info[server]}}
    return {"servers": info}


def run_skill(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main skill execution logic."""
    operation = input_data.get("operation", "status")
    server = input_data.get("server")

    result = {
        "ok": False,
        "operation": operation,
        "server": server,
        "result": None,
        "error": None
    }

    try:
        if operation == "status":
            if server == "ags":
                result["result"] = check_ags_bridge_status()
                result["ok"] = True
            elif server == "vscode":
                port = input_data.get("port", VSCODE_BRIDGE_DEFAULT_PORT)
                result["result"] = check_vscode_bridge_status(port)
                result["ok"] = True
            else:
                # Check both servers
                result["server"] = "both"
                result["result"] = {
                    "ags": check_ags_bridge_status(),
                    "vscode": check_vscode_bridge_status()
                }
                result["ok"] = True

        elif operation == "execute":
            if server != "ags":
                result["error"] = "execute operation only supports server='ags'"
                return result

            command = input_data.get("command")
            if not command:
                result["error"] = "command is required for execute operation"
                return result

            cwd = input_data.get("cwd")
            timeout = input_data.get("timeout_seconds", 30)
            exec_result = execute_ags_command(command, cwd, timeout)
            result["result"] = exec_result
            result["ok"] = exec_result.get("ok", False)
            if not result["ok"]:
                result["error"] = exec_result.get("error")

        elif operation == "launch_terminal":
            if server != "vscode":
                result["error"] = "launch_terminal operation only supports server='vscode'"
                return result

            terminal_name = input_data.get("terminal_name", "Agent Terminal")
            cwd = input_data.get("cwd")
            initial_command = input_data.get("initial_command")
            port = input_data.get("port", VSCODE_BRIDGE_DEFAULT_PORT)

            launch_result = launch_vscode_terminal(terminal_name, cwd, initial_command, port)
            result["result"] = launch_result
            result["ok"] = launch_result.get("ok", False)
            if not result["ok"]:
                result["error"] = launch_result.get("error")

        elif operation == "setup_info":
            result["result"] = get_setup_info(server)
            result["ok"] = True

        else:
            result["error"] = f"Unknown operation: {operation}. Valid: status, execute, launch_terminal, setup_info"

    except Exception as e:
        result["error"] = str(e)

    return result


def main() -> int:
    """CLI entrypoint."""
    if len(sys.argv) < 3:
        print("Usage: run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    input_data = load_json(input_path)
    result = run_skill(input_data)

    # Use GuardedWriter if available
    writer = None
    if GuardedWriter:
        try:
            writer = GuardedWriter(PROJECT_ROOT, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
        except Exception:
            pass

    write_json(output_path, result, writer)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
