# MCP Integration README

This directory contains the AGS MCP (Model Context Protocol) server, enabling external AI clients to interact with the Agent Governance System.

---

## 🚀 **Quick Start - Automatic On-Demand**

### **Option 1: Auto-Start (Recommended - Zero Setup)**

**No manual start needed!** Just connect from any client and the server auto-starts.

Use `LAW/CONTRACTS/_runs/ags_mcp_auto.py` as your entrypoint - server starts automatically on first interaction.

**For Claude Desktop:** The config in `MCP/claude_desktop_config.json` is already configured for auto-start.

### **Option 2: Manual Start**

```powershell
# 1. Start the server (runs in background)
cd "d:\CCC 2.0\AI\agent-governance-system"
.\CAPABILITY\MCP\autostart.ps1 -Start

# 2. Verify it's running
.\CAPABILITY\MCP\autostart.ps1 -Status
```

**Both work!** Multiple agents, extensions, and clients can connect simultaneously.

### **Common Commands**

| Command | What it does |
|---------|-------------|
| `.\CAPABILITY\MCP\autostart.ps1 -Start` | Start server in background |
| `.\CAPABILITY\MCP\autostart.ps1 -Stop` | Stop the server |
| `.\CAPABILITY\MCP\autostart.ps1 -Status` | Check if running |
| `.\CAPABILITY\MCP\autostart.ps1 -Restart` | Restart the server |
| `.\CAPABILITY\MCP\autostart.ps1 -Install` | Install autostart at boot (requires admin) |

### **Alternative: Foreground Mode**

```cmd
CAPABILITY\MCP\start_simple.cmd
```
Runs in current window. Press Ctrl+C to stop.

---

## Quick Start (Claude Desktop)

**Recommended entrypoint:** `LAW/CONTRACTS/_runs/ags_mcp_auto.py` (auto-starts server, redirects audit logs to `LAW/CONTRACTS/_runs/mcp_logs/`).

### Connect Claude Desktop

1. Find your Claude Desktop config file:
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add this entry (or copy from `MCP/claude_desktop_config.json`):

**Windows (Auto-Start - Recommended):**

```json
{
  "mcpServers": {
    "ags": {
      "command": "python",
      "args": ["D:/CCC 2.0/AI/agent-governance-system/LAW/CONTRACTS/_runs/ags_mcp_auto.py"],
      "cwd": "D:/CCC 2.0/AI/agent-governance-system"
    }
  }
}
```

**Alternative (Manual Start Required):**

```json
{
  "mcpServers": {
    "ags": {
      "command": "python",
      "args": ["D:/CCC 2.0/AI/agent-governance-system/LAW/CONTRACTS/ags_mcp_entrypoint.py"],
      "cwd": "D:/CCC 2.0/AI/agent-governance-system"
    }
  }
}
```

**WSL (python3):**

```json
{
  "mcpServers": {
    "ags": {
      "command": "python3",
      "args": ["/mnt/d/CCC 2.0/AI/agent-governance-system/LAW/CONTRACTS/ags_mcp_entrypoint.py"],
      "cwd": "/mnt/d/CCC 2.0/AI/agent-governance-system"
    }
  }
}
```

3. Restart Claude Desktop (or the MCP client).

4. You should now see AGS tools available in Claude.

## Test the Server

Run the built-in test:
```bash
python LAW/CONTRACTS/ags_mcp_entrypoint.py --test
```

Or use the verification skills:
- `mcp-smoke` (CLI smoke test)
- `mcp-extension-verify` (extension-agnostic checklist + smoke test)

## Available Tools

| Tool | Description |
|------|-------------|
| `cortex_query` | Search the cortex index for files and entities |
| `context_search` | Search ADRs, preferences, and other context records |
| `context_review` | Check for overdue or upcoming ADR reviews |
| `canon_read` | Read canon files (CONTRACT, INVARIANTS, etc.) |
| `skill_run` | Execute an AGS skill with JSON input |
| `pack_validate` | Validate a memory pack for completeness |
| `critic_run` | Run governance checks via `TOOLS/critic.py` |
| `adr_create` | Create a new ADR with the standard template |
| `commit_ceremony` | Execute commit ceremony checks |
| `codebook_lookup` | Look up codebook entries by ID |
| `research_cache` | Manage the persistent research cache |
| `terminal_bridge` | Execute a command via the local PowerShell bridge |

## Terminal Bridge Setup

The `terminal_bridge` tool sends commands to a local PowerShell HTTP bridge.

1. Update the token in `CAPABILITY/MCP/powershell_bridge_config.json`.
2. Start the bridge:

```powershell
.\CAPABILITY\MCP\powershell_bridge.ps1
```

3. If you keep the config elsewhere, set `MCP_TERMINAL_BRIDGE_CONFIG` to its path.

WSL note: WSL2 cannot reach Windows services bound to 127.0.0.1. Set
`listen_host` to `0.0.0.0` and `connect_host` to the Windows host IP
(from WSL: `grep nameserver /etc/resolv.conf`) in
`CAPABILITY/MCP/powershell_bridge_config.json`.

## Available Resources

| URI | Description |
|-----|-------------|
| `ags://canon/contract` | The CONTRACT.md (supreme authority) |
| `ags://canon/invariants` | The INVARIANTS.md (locked decisions) |
| `ags://canon/genesis` | The Genesis Prompt for session bootstrap |
| `ags://canon/versioning` | The VERSIONING.md |
| `ags://context/decisions` | Live list of all ADRs |
| `ags://context/preferences` | Live list of STYLE records |
| `ags://cortex/index` | Full cortex index in JSON |
| `ags://maps/entrypoints` | The ENTRYPOINTS.md |
| `ags://agents` | The AGENTS.md |

## Available Prompts

| Name | Description |
|------|-------------|
| `genesis` | Bootstrap prompt for AGS session initialization |
| `commit_ceremony` | Checklist for the commit ceremony |
| `adr_template` | Template for creating new ADRs |

## Example Usage (in Claude)

"Read the CONTRACT rules"
→ Claude uses `canon_read` tool with `{"file": "CONTRACT"}`

"What files mention 'packer'?"
→ Claude uses `cortex_query` tool with `{"query": "packer"}`

"Are any ADR reviews overdue?"
→ Claude uses `context_review` tool

"Run the pack-validate skill on the latest pack"
→ Claude uses `skill_run` tool with `{"skill": "pack-validate", "input": {...}}`
