#!/usr/bin/env python3
"""
MCP Toolkit - Unified MCP operations skill.

Consolidates: mcp-builder, mcp-access-validator, mcp-extension-verify,
              mcp-message-board, mcp-precommit-check, mcp-smoke, mcp-adapter

Operations:
  - build: Build MCP servers
  - validate_access: Validate agent MCP tool usage
  - verify_extension: Verify AGS MCP in IDE extensions
  - message_board: Message board operations
  - precommit: Pre-commit MCP health checks
  - smoke: MCP server smoke testing
  - adapt: MCP adapter task wrapper
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None

DURABLE_ROOTS = ["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"]


def get_writer() -> GuardedWriter:
    """Get a configured GuardedWriter instance."""
    if not GuardedWriter:
        raise RuntimeError("GuardedWriter not available")
    writer = GuardedWriter(project_root=PROJECT_ROOT, durable_roots=DURABLE_ROOTS)
    writer.open_commit_gate()
    return writer


def write_output(output_path: Path, data: Dict[str, Any], writer: GuardedWriter) -> None:
    """Write JSON output using GuardedWriter."""
    writer.mkdir_durable(str(output_path.parent))
    writer.write_durable(str(output_path), json.dumps(data, indent=2, sort_keys=True) + "\n")


# ============================================================================
# Operation: validate_access
# ============================================================================

MCP_TOOLS = {
    r"(?i)sqlite3.*connect.*system[13]\.db": "cortex_query",
    r"(?i)SELECT.*FROM.*symbols": "cortex_query",
    r"(?i)SELECT.*FROM.*vectors": "cortex_query",
    r"(?i)SELECT.*FROM.*cassettes": "context_search",
    r"(?i)\.db.*cursor.*execute": "cortex_query",
    r"(?i)open\(.*\.md\)\.read\(\)": "canon_read",
    r"(?i)Path\(.*\)\.read_text\(\)": "canon_read",
    r"(?i)read_file.*LAW/CANON": "canon_read",
    r"(?i)read_file.*LAW/CONTEXT": "context_search",
    r"(?i)read_file.*NAVIGATION/CORTEX": "cortex_query",
    r"(?i)os\.walk.*\.md": "cortex_query",
    r"(?i)glob.*\.md": "cortex_query",
    r"(?i)find.*-name.*\.md": "cortex_query",
    r"(?i)embeddings\.py": "cortex_query",
    r"(?i)vector.*search": "cortex_query",
    r"(?i)semantic.*search": "cortex_query",
    r"(?i)ADR-\d+": "context_search",
    r"(?i)LAW/CONTEXT/decisions": "context_search",
    r"(?i)LAW/CONTEXT/preferences": "context_search",
    r"(?i)session.*id": "session_info",
    r"(?i)uuid.*generate": "session_info",
    r"(?i)ADR-021": "session_info",
}

TOOL_EXAMPLES = {
    "cortex_query": {
        "description": "Search the cortex index for content",
        "example": {"query": "catalytic", "limit": 10},
        "token_savings": 0.95,
        "governance": "ADR-004, ADR-021"
    },
    "context_search": {
        "description": "Search context records (ADRs, preferences, etc.)",
        "example": {"type": "decisions", "query": "catalytic"},
        "token_savings": 0.90,
        "governance": "ADR-004"
    },
    "canon_read": {
        "description": "Read canon governance documents",
        "example": {"file": "CONTRACT"},
        "token_savings": 0.85,
        "governance": "ADR-004"
    },
    "session_info": {
        "description": "Get session information including UUID for ADR-021 compliance",
        "example": {"include_audit_log": True},
        "token_savings": 0.80,
        "governance": "ADR-021"
    },
}


def _detect_mcp_tool_needed(agent_action: str, agent_code: str = "") -> Optional[str]:
    """Detect which MCP tool should be used."""
    text_to_check = f"{agent_action} {agent_code}".lower()

    for pattern, tool in MCP_TOOLS.items():
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return tool

    action_keywords = {
        "database": "cortex_query", "sqlite": "cortex_query",
        "query": "cortex_query", "search": "cortex_query",
        "read": "canon_read", "file": "canon_read", "document": "canon_read",
        "context": "context_search", "adr": "context_search", "decision": "context_search",
        "session": "session_info", "uuid": "session_info", "identity": "session_info",
    }

    for keyword, tool in action_keywords.items():
        if keyword in text_to_check:
            return tool

    return None


def _calculate_token_waste(agent_code: str, recommended_tool: str) -> Dict[str, Any]:
    """Calculate token waste metrics."""
    if not agent_code:
        return {"token_waste_detected": False, "estimated_token_savings": 0.0,
                "code_tokens": 0, "tool_tokens": 0, "raw_savings": 0.0, "baseline_savings": 0.0}

    code_tokens = len(agent_code.split())
    tool_info = TOOL_EXAMPLES.get(recommended_tool, {})
    tool_example = json.dumps(tool_info.get("example", {}))
    tool_tokens = len(tool_example.split())

    savings = max(0.0, (code_tokens - tool_tokens) / code_tokens) if code_tokens else 0.0
    baseline_savings = tool_info.get("token_savings", 0.7)
    waste_detected = savings > 0.1 or code_tokens > 50

    return {
        "token_waste_detected": waste_detected,
        "estimated_token_savings": baseline_savings,
        "code_tokens": code_tokens,
        "tool_tokens": tool_tokens,
        "raw_savings": round(savings, 3),
        "baseline_savings": baseline_savings
    }


def op_validate_access(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Validate agent MCP tool usage."""
    agent_action = payload.get("agent_action", "")
    agent_code = payload.get("agent_code_snippet", "")
    files_accessed = payload.get("files_accessed", [])
    databases_queried = payload.get("databases_queried", [])

    recommended_tool = _detect_mcp_tool_needed(agent_action, agent_code)
    waste_metrics = _calculate_token_waste(agent_code, recommended_tool or "")
    validation_passed = not waste_metrics["token_waste_detected"]

    output = {
        "validation_passed": validation_passed,
        "token_waste_detected": waste_metrics["token_waste_detected"],
        "agent_action": agent_action,
        "code_snippet_length": len(agent_code),
        "files_accessed_count": len(files_accessed),
        "databases_queried_count": len(databases_queried),
        "token_waste_metrics": waste_metrics,
    }

    if recommended_tool and waste_metrics["token_waste_detected"]:
        tool_info = TOOL_EXAMPLES.get(recommended_tool, {})
        output.update({
            "recommended_mcp_tool": recommended_tool,
            "tool_description": tool_info.get("description", ""),
            "tool_usage_example": tool_info.get("example", {}),
            "estimated_token_savings": waste_metrics["estimated_token_savings"],
            "governance_compliance": tool_info.get("governance", ""),
        })
    elif recommended_tool:
        output["recommended_mcp_tool"] = recommended_tool
        output["note"] = "MCP tool available but token waste minimal"
    else:
        output["recommended_mcp_tool"] = None
        output["note"] = "No specific MCP tool match found"

    write_output(output_path, output, writer)
    return 0


# ============================================================================
# Operation: verify_extension / smoke - shared helpers
# ============================================================================

def _get_cortex_query(project_root: Path) -> Tuple[Any, str]:
    """Get cortex query module."""
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from NAVIGATION.CORTEX.semantic import query as cortex_query
        cortex_query.get_metadata("canon_version")
        return cortex_query, ""
    except (ImportError, FileNotFoundError):
        build_script = project_root / "NAVIGATION" / "CORTEX" / "db" / "cortex.build.py"
        if not build_script.exists():
            return None, "Cortex build script not found"
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, str(build_script)],
            capture_output=True, text=True, cwd=str(project_root), env=env,
        )
        if result.returncode != 0:
            return None, (result.stdout + result.stderr).strip()

        try:
            from NAVIGATION.CORTEX.semantic import query as cortex_query
            cortex_query.get_metadata("canon_version")
            return cortex_query, ""
        except (ImportError, FileNotFoundError) as exc:
            return None, str(exc)


def _find_entrypoint(cortex_query: Any, entrypoint_substring: str) -> str:
    """Find entrypoint via cortex query."""
    try:
        results = cortex_query.find_entities_containing_path(entrypoint_substring)
        if results:
            ordered = sorted(results, key=lambda r: r.get("paths", {}).get("source", ""))
            entry = ordered[0]
            source_path = entry.get("paths", {}).get("source", "")
            if source_path:
                return source_path
    except Exception:
        pass
    return Path(entrypoint_substring).as_posix()


def _run_entrypoint(project_root: Path, entrypoint_rel: str, args: List[str]) -> subprocess.CompletedProcess:
    """Run an MCP entrypoint."""
    entrypoint_path = project_root / Path(entrypoint_rel)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) if not pythonpath else f"{project_root}{os.pathsep}{pythonpath}"
    return subprocess.run(
        [sys.executable, str(entrypoint_path), *args],
        capture_output=True, text=True, cwd=str(project_root), env=env, timeout=120,
    )


def _ensure_entrypoint_wrapper(entrypoint_path: Path, writer: GuardedWriter) -> None:
    """Ensure the MCP entrypoint wrapper exists."""
    if entrypoint_path.exists():
        return

    writer.mkdir_durable(str(entrypoint_path.parent))
    content = "\n".join([
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
    ]) + "\n"
    writer.write_durable(str(entrypoint_path), content)


def _instructions_for(client: str) -> List[str]:
    """Get client-specific reload instructions."""
    client_key = (client or "generic").strip().lower()
    if client_key == "vscode":
        return [
            "Command Palette -> Developer: Reload Window.",
            "If tools still missing: Developer: Restart Extension Host.",
            "Open the extension MCP/server logs and confirm initialize/tools/list succeeds.",
        ]
    if client_key == "claude":
        return [
            "Quit and relaunch the client to reload MCP config.",
            "Verify the MCP server entry shows as connected in the tools panel.",
        ]
    return [
        "Restart or reload the MCP client/extension to pick up config changes.",
        "Check client logs for successful initialize/tools/list responses.",
    ]


# ============================================================================
# Operation: verify_extension
# ============================================================================

def op_verify_extension(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Verify AGS MCP server in IDE extension."""
    entrypoint_substring = payload.get("entrypoint_substring", "LAW/CONTRACTS/ags_mcp_entrypoint.py")
    args = payload.get("args", ["--test"])
    client = payload.get("client", "generic")

    cortex_query, error = _get_cortex_query(PROJECT_ROOT)
    if cortex_query is None:
        write_output(output_path, {
            "ok": False, "returncode": 1,
            "entrypoint": Path(entrypoint_substring).as_posix(),
            "args": args, "client": client,
            "instructions": _instructions_for(client),
        }, writer)
        return 1

    entrypoint_rel = _find_entrypoint(cortex_query, entrypoint_substring)
    entrypoint_path = PROJECT_ROOT / Path(entrypoint_rel)
    _ensure_entrypoint_wrapper(entrypoint_path, writer)
    result = _run_entrypoint(PROJECT_ROOT, entrypoint_rel, args)

    write_output(output_path, {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "entrypoint": Path(entrypoint_rel).as_posix(),
        "args": args, "client": client,
        "instructions": _instructions_for(client),
    }, writer)
    return 0 if result.returncode == 0 else 1


# ============================================================================
# Operation: smoke
# ============================================================================

def _run_bridge_smoke(project_root: Path, config_path: str, bridge_payload: Dict, timeout_seconds: int) -> Dict[str, Any]:
    """Run bridge smoke test."""
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
    data = json.dumps(bridge_payload).encode("utf-8")
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


def op_smoke(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """MCP server smoke testing."""
    entrypoint_substring = payload.get("entrypoint_substring", "LAW/CONTRACTS/ags_mcp_entrypoint.py")
    args = payload.get("args", ["--test"])
    bridge_smoke = payload.get("bridge_smoke", {})
    bridge_enabled = bool(bridge_smoke.get("enabled", False))

    cortex_query, error = _get_cortex_query(PROJECT_ROOT)
    if cortex_query is None:
        write_output(output_path, {
            "ok": False, "returncode": 1,
            "entrypoint": Path(entrypoint_substring).as_posix(),
            "args": args,
            "bridge_smoke": {"enabled": bridge_enabled, "ok": False, "error": "CORTEX_UNAVAILABLE"},
        }, writer)
        return 1

    entrypoint_rel = _find_entrypoint(cortex_query, entrypoint_substring)
    entrypoint_path = PROJECT_ROOT / Path(entrypoint_rel)
    _ensure_entrypoint_wrapper(entrypoint_path, writer)
    result = _run_entrypoint(PROJECT_ROOT, entrypoint_rel, args)

    bridge_result = {"enabled": bridge_enabled, "ok": True}
    if bridge_enabled:
        config_path = bridge_smoke.get("config_path", "CAPABILITY/MCP/powershell_bridge_config.json")
        command = bridge_smoke.get("command", "Get-Date")
        cwd = bridge_smoke.get("cwd")
        timeout_seconds = int(bridge_smoke.get("timeout_seconds", 30))
        bridge_payload: Dict[str, Any] = {"command": command}
        if isinstance(cwd, str) and cwd.strip():
            bridge_payload["cwd"] = cwd
        bridge_result = _run_bridge_smoke(PROJECT_ROOT, config_path, bridge_payload, timeout_seconds)

    write_output(output_path, {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "entrypoint": Path(entrypoint_rel).as_posix(),
        "args": args,
        "bridge_smoke": bridge_result,
    }, writer)
    return 0 if result.returncode == 0 else 1


# ============================================================================
# Operation: precommit
# ============================================================================

def _result(ok: bool, **extras: Any) -> Dict[str, Any]:
    """Create a result dict."""
    payload = {"ok": ok}
    payload.update(extras)
    return payload


def _windows_repo_root(project_root: Path) -> str:
    """Convert to Windows path format."""
    posix = project_root.as_posix()
    if posix.startswith("/mnt/") and len(posix) > 6:
        drive = posix[5]
        rest = posix[7:] if posix[6] == "/" else posix[6:]
        rest = rest.replace("/", "\\")
        return f"{drive.upper()}:\\{rest}"
    return str(project_root)


def _powershell_json(command: str) -> Dict[str, Any]:
    """Execute PowerShell command and parse JSON output."""
    candidates = [
        shutil.which("powershell.exe"),
        shutil.which("powershell"),
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    ]
    powershell = next((item for item in candidates if item and os.path.exists(item)), None)
    if not powershell:
        return _result(False, error="POWERSHELL_NOT_FOUND")

    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return _result(False, error="POWERSHELL_ERROR", detail=result.stderr.strip())
    return _result(True, output=result.stdout.strip())


def _bridge_request(project_root: Path, config_path: str, command: str, timeout_seconds: int) -> Dict[str, Any]:
    """Make a bridge request."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project_root / config_file
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return _result(False, error=f"BRIDGE_CONFIG_READ_FAILED: {exc}")

    host = str(config.get("connect_host", "127.0.0.1"))
    port = int(config.get("port", 8765))
    token = str(config.get("token", ""))
    url = f"http://{host}:{port}/run"
    data = json.dumps({"command": command}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token and token != "CHANGE_ME":
        headers["X-Bridge-Token"] = token

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return _result(False, error=f"BRIDGE_HTTP_ERROR: {exc.code} {exc.reason}")
    except urllib.error.URLError as exc:
        return _result(False, error=f"BRIDGE_CONNECTION_ERROR: {exc}")

    try:
        payload = json.loads(raw)
    except Exception:
        return _result(False, error="BRIDGE_INVALID_JSON", raw=raw)

    if not payload.get("ok", False):
        return _result(False, error=payload.get("error", "BRIDGE_COMMAND_FAILED"))
    return _result(True, output=payload.get("output", ""))


def _check_pid_running(project_root: Path, bridge_config: str, timeout_seconds: int) -> Dict[str, Any]:
    """Check if MCP server PID is running."""
    pid_path = project_root / "LAW" / "CONTRACTS" / "_runs" / "mcp_logs" / "server.pid"

    if os.name == "nt":
        if not pid_path.exists():
            return _result(False, error="PID_MISSING", path=str(pid_path))
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return _result(False, error="PID_INVALID")
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        ok = str(pid) in result.stdout
        return _result(ok, pid=pid, status="RUNNING" if ok else "NOT_RUNNING")

    # Non-Windows: use PowerShell or bridge
    windows_root = _windows_repo_root(project_root)
    pid_path_win = f"{windows_root}\\LAW\\CONTRACTS\\_runs\\mcp_logs\\server.pid"
    command = (
        "& { "
        "$pidPath = '{pid_path}'; "
        "if (Test-Path $pidPath) { "
        "$pid = Get-Content $pidPath -ErrorAction Stop; "
        "$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue; "
        "if ($proc) { @{ok=$true;status='RUNNING';pid=[int]$pid} } "
        "else { @{ok=$false;status='NOT_RUNNING';pid=[int]$pid} } "
        "} else { @{ok=$false;error='PID_MISSING'} } "
        "} | ConvertTo-Json -Compress"
    ).replace("{pid_path}", pid_path_win.replace("'", "''"))

    ps = _powershell_json(command)
    if ps.get("ok"):
        try:
            data = json.loads(ps.get("output", "{}"))
            return _result(bool(data.get("ok", False)), pid=data.get("pid"), status=data.get("status") or data.get("error"))
        except Exception:
            pass

    bridge = _bridge_request(project_root, bridge_config, command, timeout_seconds)
    if bridge.get("ok"):
        try:
            data = json.loads(bridge.get("output", "{}"))
            return _result(bool(data.get("ok", False)), pid=data.get("pid"), status=data.get("status") or data.get("error"))
        except Exception:
            pass
    return _result(False, error="BRIDGE_UNAVAILABLE")


def _check_autostart(project_root: Path, bridge_config: str, timeout_seconds: int) -> Dict[str, Any]:
    """Check Windows autostart status."""
    command = (
        "& { "
        "$task = Get-ScheduledTask -TaskName 'AGS_MCP_Server_Autostart' -ErrorAction SilentlyContinue; "
        "$startup = [Environment]::GetFolderPath('Startup'); "
        "$shortcut = Join-Path $startup 'AGS_MCP_Server_Autostart.lnk'; "
        "if ($null -ne $task) { "
        "@{ok=([bool]$task.Enabled -and $task.State -ne 'Disabled'); enabled=[bool]$task.Enabled; state=$task.State} } "
        "elseif (Test-Path $shortcut) { @{ok=$true; enabled=$true; state='StartupFolder'} } "
        "else { @{ok=$false; error='TASK_MISSING'} } "
        "} | ConvertTo-Json -Compress"
    )

    if os.name == "nt":
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout.strip() or "{}")
                enabled = bool(data.get("Enabled", data.get("enabled", False)))
                state = str(data.get("State", data.get("state", ""))).strip()
                ok = enabled and state.lower() != "disabled"
                if ok or state == "StartupFolder":
                    return _result(True, enabled=enabled, state=state)

                startup_dir = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
                shortcut = os.path.join(startup_dir, "AGS_MCP_Server_Autostart.lnk")
                if os.path.exists(shortcut):
                    return _result(True, enabled=True, state="StartupFolder")
                return _result(False, enabled=enabled, state=state)
            except Exception:
                pass
        return _result(False, error="TASK_QUERY_FAILED")

    # Non-Windows
    ps = _powershell_json(command)
    if ps.get("ok"):
        try:
            data = json.loads(ps.get("output", "{}"))
            return _result(bool(data.get("ok", False)), enabled=data.get("enabled"), state=data.get("state"), error=data.get("error"))
        except Exception:
            pass

    bridge = _bridge_request(project_root, bridge_config, command, timeout_seconds)
    if bridge.get("ok"):
        try:
            data = json.loads(bridge.get("output", "{}"))
            return _result(bool(data.get("ok", False)), enabled=data.get("enabled"), state=data.get("state"), error=data.get("error"))
        except Exception:
            pass
    return _result(False, error="BRIDGE_UNAVAILABLE")


def op_precommit(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Pre-commit MCP health checks."""
    entrypoint = payload.get("entrypoint", "LAW/CONTRACTS/ags_mcp_entrypoint.py")
    auto_entrypoint = payload.get("auto_entrypoint", "LAW/CONTRACTS/_runs/ags_mcp_auto.py")
    args = payload.get("args", ["--test"])
    auto_args = payload.get("auto_args", ["--test"])
    require_running = bool(payload.get("require_running", True))
    require_autostart = bool(payload.get("require_autostart", True))
    dry_run = bool(payload.get("dry_run", False))
    bridge_config = payload.get("bridge_config", "CAPABILITY/MCP/powershell_bridge_config.json")
    bridge_timeout = int(payload.get("bridge_timeout_seconds", 30))

    if dry_run:
        checks = {
            "entrypoint": _result(True, skipped=True),
            "auto_entrypoint": _result(True, skipped=True),
            "running": _result(True, skipped=True),
            "autostart": _result(True, skipped=True),
        }
        write_output(output_path, {"ok": True, "checks": checks}, writer)
        return 0

    checks: Dict[str, Any] = {}

    # Check entrypoints
    entry_path = PROJECT_ROOT / Path(entrypoint)
    if entry_path.exists():
        result = _run_entrypoint(PROJECT_ROOT, entrypoint, args)
        checks["entrypoint"] = _result(
            result.returncode == 0,
            returncode=result.returncode,
            output_tail=(result.stdout + result.stderr).strip()[-400:] if result.returncode != 0 else ""
        )
    else:
        checks["entrypoint"] = _result(False, error="ENTRYPOINT_MISSING", path=str(entry_path))

    auto_path = PROJECT_ROOT / Path(auto_entrypoint)
    if auto_path.exists():
        result = _run_entrypoint(PROJECT_ROOT, auto_entrypoint, auto_args)
        checks["auto_entrypoint"] = _result(
            result.returncode == 0,
            returncode=result.returncode,
            output_tail=(result.stdout + result.stderr).strip()[-400:] if result.returncode != 0 else ""
        )
    else:
        checks["auto_entrypoint"] = _result(False, error="ENTRYPOINT_MISSING", path=str(auto_path))

    if require_running:
        checks["running"] = _check_pid_running(PROJECT_ROOT, bridge_config, bridge_timeout)
    else:
        checks["running"] = _result(True, skipped=True)

    if require_autostart:
        checks["autostart"] = _check_autostart(PROJECT_ROOT, bridge_config, bridge_timeout)
    else:
        checks["autostart"] = _result(True, skipped=True)

    ok = all(check.get("ok") for check in checks.values())
    write_output(output_path, {"ok": ok, "checks": checks}, writer)
    return 0 if ok else 1


# ============================================================================
# Operation: build / adapt / message_board (placeholders)
# ============================================================================

def op_build(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Build MCP servers (placeholder)."""
    task = payload.get("task", {})
    task_id = task.get("id", "unknown")
    write_output(output_path, {"status": "success", "task_id": task_id}, writer)
    return 0


def op_adapt(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """MCP adapter task wrapper."""
    task = payload.get("task", {})
    task_id = task.get("id", "unknown")
    write_output(output_path, {"status": "success", "task_id": task_id}, writer)
    return 0


def op_message_board(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Message board operations (placeholder)."""
    write_output(output_path, {
        "ok": False,
        "code": "NOT_IMPLEMENTED",
        "details": {"message": "Use repo implementation + tests; skill runner is a governance placeholder."}
    }, writer)
    return 0


# ============================================================================
# Main dispatcher
# ============================================================================

OPERATIONS = {
    "build": op_build,
    "validate_access": op_validate_access,
    "verify_extension": op_verify_extension,
    "message_board": op_message_board,
    "precommit": op_precommit,
    "smoke": op_smoke,
    "adapt": op_adapt,
}


def main(input_path: Path, output_path: Path) -> int:
    """Main entry point."""
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    operation = payload.get("operation")
    if not operation:
        print("Error: 'operation' field is required")
        return 1

    if operation not in OPERATIONS:
        print(f"Error: Unknown operation '{operation}'. Valid: {', '.join(OPERATIONS.keys())}")
        return 1

    try:
        writer = get_writer()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    return OPERATIONS[operation](payload, output_path, writer)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
