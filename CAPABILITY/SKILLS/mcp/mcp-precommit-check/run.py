#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _result(ok: bool, **extras: Any) -> Dict[str, Any]:
    payload = {"ok": ok}
    payload.update(extras)
    return payload


def _run_entrypoint(project_root: Path, entrypoint: str, args: List[str]) -> Dict[str, Any]:
    entry_path = project_root / Path(entrypoint)
    if not entry_path.exists():
        return _result(False, error="ENTRYPOINT_MISSING", path=str(entry_path))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) if not pythonpath else f"{project_root}{os.pathsep}{pythonpath}"

    result = subprocess.run(
        [sys.executable, str(entry_path), *args],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
        timeout=120,
    )
    ok = result.returncode == 0
    tail = (result.stdout + result.stderr).strip()[-400:] if not ok else ""
    return _result(ok, returncode=result.returncode, output_tail=tail)


def _windows_repo_root(project_root: Path) -> str:
    posix = project_root.as_posix()
    if posix.startswith("/mnt/") and len(posix) > 6:
        drive = posix[5]
        rest = posix[7:] if posix[6] == "/" else posix[6:]
        rest = rest.replace("/", "\\")
        return f"{drive.upper()}:\\{rest}"
    return str(project_root)


def _bridge_request(project_root: Path, config_path: str, command: str, timeout_seconds: int) -> Dict[str, Any]:
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
        return _result(False, error=payload.get("error", "BRIDGE_COMMAND_FAILED"), raw=payload.get("output"))
    return _result(True, output=payload.get("output", ""))


def _powershell_json(command: str) -> Dict[str, Any]:
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
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return _result(False, error="POWERSHELL_ERROR", detail=result.stderr.strip())
    return _result(True, output=result.stdout.strip())


def _check_pid_running(project_root: Path, bridge_config: str, timeout_seconds: int) -> Dict[str, Any]:
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
            capture_output=True,
            text=True,
        )
        ok = str(pid) in result.stdout
        return _result(ok, pid=pid, status="RUNNING" if ok else "NOT_RUNNING")

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
    )
    command = command.replace("{pid_path}", pid_path_win.replace("'", "''"))
    ps = _powershell_json(command)
    if ps.get("ok"):
        try:
            data = json.loads(ps.get("output", "{}"))
        except Exception:
            return _result(False, error="POWERSHELL_OUTPUT_PARSE_FAILED")
    else:
        bridge = _bridge_request(project_root, bridge_config, command, timeout_seconds)
        if not bridge.get("ok"):
            return _result(False, error=bridge.get("error", "BRIDGE_UNAVAILABLE"))
        try:
            data = json.loads(bridge.get("output", "{}"))
        except Exception:
            return _result(False, error="BRIDGE_OUTPUT_PARSE_FAILED")
    return _result(bool(data.get("ok", False)), pid=data.get("pid"), status=data.get("status") or data.get("error"))


def _check_autostart(project_root: Path, bridge_config: str, timeout_seconds: int) -> Dict[str, Any]:
    if os.name != "nt":
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
        ps = _powershell_json(command)
        if ps.get("ok"):
            try:
                data = json.loads(ps.get("output", "{}"))
            except Exception:
                return _result(False, error="POWERSHELL_OUTPUT_PARSE_FAILED")
        else:
            bridge = _bridge_request(project_root, bridge_config, command, timeout_seconds)
            if not bridge.get("ok"):
                return _result(False, error=bridge.get("error", "BRIDGE_UNAVAILABLE"))
            try:
                data = json.loads(bridge.get("output", "{}"))
            except Exception:
                return _result(False, error="BRIDGE_OUTPUT_PARSE_FAILED")
        return _result(bool(data.get("ok", False)), enabled=data.get("enabled"), state=data.get("state"), error=data.get("error"))

    if not shutil.which("powershell"):
        return _result(False, error="POWERSHELL_NOT_FOUND")

    command = (
        "Get-ScheduledTask -TaskName 'AGS_MCP_Server_Autostart' "
        "| Select-Object State,Enabled "
        "| ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return _result(False, error="TASK_QUERY_FAILED", detail=result.stderr.strip())

    try:
        data = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return _result(False, error="TASK_PARSE_FAILED")

    enabled = bool(data.get("Enabled", False))
    state = str(data.get("State", "")).strip()
    ok = enabled and state.lower() != "disabled"
    if ok:
        return _result(True, enabled=enabled, state=state)

    startup_dir = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
    shortcut = os.path.join(startup_dir, "AGS_MCP_Server_Autostart.lnk")
    if os.path.exists(shortcut):
        return _result(True, enabled=True, state="StartupFolder")
    return _result(False, enabled=enabled, state=state)


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = load_json(input_path)

    entrypoint = payload.get("entrypoint", "LAW/CONTRACTS/ags_mcp_entrypoint.py")
    auto_entrypoint = payload.get("auto_entrypoint", "LAW/CONTRACTS/_runs/ags_mcp_auto.py")
    args = payload.get("args", ["--test"])
    auto_args = payload.get("auto_args", ["--test"])
    require_running = bool(payload.get("require_running", True))
    require_autostart = bool(payload.get("require_autostart", True))
    dry_run = bool(payload.get("dry_run", False))
    bridge_config = payload.get("bridge_config", "CAPABILITY/MCP/powershell_bridge_config.json")
    bridge_timeout = int(payload.get("bridge_timeout_seconds", 30))

    project_root = Path(__file__).resolve().parents[4]

    if dry_run:
        checks = {
            "entrypoint": _result(True, skipped=True),
            "auto_entrypoint": _result(True, skipped=True),
            "running": _result(True, skipped=True),
            "autostart": _result(True, skipped=True),
        }
        write_json(output_path, {"ok": True, "checks": checks})
        return 0

    checks: Dict[str, Any] = {}
    checks["entrypoint"] = _run_entrypoint(project_root, entrypoint, args)
    checks["auto_entrypoint"] = _run_entrypoint(project_root, auto_entrypoint, auto_args)

    if require_running:
        checks["running"] = _check_pid_running(project_root, bridge_config, bridge_timeout)
    else:
        checks["running"] = _result(True, skipped=True)

    if require_autostart:
        checks["autostart"] = _check_autostart(project_root, bridge_config, bridge_timeout)
    else:
        checks["autostart"] = _result(True, skipped=True)

    ok = all(check.get("ok") for check in checks.values())
    write_json(output_path, {"ok": ok, "checks": checks})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
