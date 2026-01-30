---
name: terminal-bridge
description: "Unified terminal bridge for AGS. Supports two distinct servers: (1) AGS PowerShell Bridge for Google Antigravity MCP command execution, and (2) VSCode Antigravity Bridge for spawning terminals inside VSCode."
version: "1.0.0"
compatibility: Windows, VSCode, Google Antigravity
---

<!-- CONTENT_HASH: dadca77801cf954b2c7f65d1434667c19ab69b7fa83b1f175bae23f2369f64a7 -->

**required_canon_version:** >=3.0.0

# Skill: terminal-bridge

**Version:** 1.0.0

**Status:** Active

## Overview

This skill provides unified access to two distinct terminal bridge servers:

### Server 1: AGS PowerShell Bridge (Google Antigravity)

- **Purpose:** Execute commands via HTTP from Claude/MCP to a local PowerShell process
- **Port:** 8765 (configurable)
- **Config:** `CAPABILITY/MCP/powershell_bridge_config.json`
- **Script:** `CAPABILITY/MCP/powershell_bridge.ps1`
- **Use Case:** Google Antigravity MCP tool `terminal_bridge`

### Server 2: VSCode Antigravity Bridge

- **Purpose:** Spawn named terminals inside VSCode IDE
- **Port:** 4000 (default)
- **Extension:** `antigravity-bridge` VSCode extension
- **Use Case:** Creating agent terminals that appear in VSCode panel

## Trigger

Use when:
- Setting up terminal bridges for agent command execution
- Launching terminals inside VSCode
- Executing commands via the PowerShell HTTP bridge
- Checking bridge server status

## Input Schema

```json
{
  "operation": "status | execute | launch_terminal | setup_info",
  "server": "ags | vscode",
  "command": "command to execute (for execute operation)",
  "cwd": "working directory (optional)",
  "terminal_name": "name for VSCode terminal (for launch_terminal)",
  "initial_command": "command to run on terminal start (for launch_terminal)"
}
```

### Operations

| Operation | Server | Description |
|-----------|--------|-------------|
| `status` | both | Check if bridge servers are reachable |
| `execute` | ags | Execute command via PowerShell bridge |
| `launch_terminal` | vscode | Spawn a named terminal in VSCode |
| `setup_info` | both | Return setup instructions and paths |

## Output Schema

```json
{
  "ok": true,
  "server": "ags | vscode",
  "operation": "status | execute | launch_terminal | setup_info",
  "result": {},
  "error": null
}
```

## Server Details

### AGS PowerShell Bridge (Port 8765)

Used by **Google Antigravity** and the MCP `terminal_bridge` tool.

**Setup:**
```powershell
powershell -ExecutionPolicy Bypass -File CAPABILITY\MCP\powershell_bridge.ps1
```

**API:**
```
POST http://127.0.0.1:8765/run
Headers: X-Bridge-Token: <token>
Body: {"command": "dir", "cwd": "C:\\path"}
```

### VSCode Antigravity Bridge (Port 4000)

Used by agents to spawn **terminals inside VSCode**.

**Setup:**
1. Install VSCode extension: `antigravity-bridge-0.1.0.vsix`
2. Extension auto-starts HTTP server on port 4000

**API:**
```
POST http://127.0.0.1:4000/terminal
Body: {"name": "Terminal Name", "cwd": "/path", "initialCommand": "echo hello"}
```

## Constraints

- INV-014: External window spawning is PROHIBITED. Use VSCode bridge for agent terminals.
- AGS bridge requires valid token from config file.
- VSCode bridge only works when VSCode is running with extension installed.

## Fixtures

- `fixtures/basic/` - Basic status check test

## Related

- `mcp__ags__terminal_bridge` - MCP tool using AGS bridge
- `CAPABILITY/SKILLS/utilities/powershell-bridge` - Legacy setup-only skill
- `references/VSCODE_BRIDGE.md` - VSCode extension documentation
