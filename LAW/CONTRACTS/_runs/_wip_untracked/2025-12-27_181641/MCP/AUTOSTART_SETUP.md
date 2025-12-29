# AGS MCP Server Autostart Setup

Your MCP server autostart system has been created. Here's how to use it:

## Quick Start (Start Server Now)

```powershell
cd "d:\CCC 2.0\AI\agent-governance-system"
.\MCP\autostart.ps1 -Start
```

## Check Status

```powershell
.\MCP\autostart.ps1 -Status
```

## Stop Server

```powershell
.\MCP\autostart.ps1 -Stop
```

## Restart Server

```powershell
.\MCP\autostart.ps1 -Restart
```

---

## Option 1: Windows Task Scheduler (Recommended - Runs at Boot)

**Requires Administrator privileges**

```powershell
# Run PowerShell as Administrator, then:
cd "d:\CCC 2.0\AI\agent-governance-system"
.\MCP\autostart.ps1 -Install
```

This creates a Windows Task that:
- Starts automatically on system boot
- Runs hidden in the background
- Restarts on failure (up to 3 times)
- Works even if you're not logged in

**To uninstall:**
```powershell
.\MCP\autostart.ps1 -Uninstall
```

---

## Option 2: Windows Startup Folder (Simpler - No Admin Needed)

1. Press `Win+R` and type `shell:startup`
2. Create a shortcut to: `d:\CCC 2.0\AI\agent-governance-system\MCP\autostart.ps1`
3. Right-click shortcut → Properties → Shortcut
4. In "Target" field, change to:
   ```
   powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File "d:\CCC 2.0\AI\agent-governance-system\MCP\autostart.ps1" -Start
   ```
5. Click OK

This starts the server when you log in (not at boot).

---

## Option 3: Manual Start Each Time

Simply run:
```powershell
.\MCP\autostart.ps1 -Start
```

The server will stay running until you reboot or manually stop it.

---

## Configuration

Edit `MCP/autostart_config.json` to customize:

```json
{
  "components": {
    "mcp_server": {
      "enabled": true,
      "restart_on_failure": true,
      "max_restarts": 3
    },
    "cortex_rebuild": {
      "enabled": true,
      "on_startup": true
    }
  }
}
```

---

## Logs

All logs are saved to: `CONTRACTS/_runs/mcp_logs/`

- `autostart.log` - Startup script logs
- `server_stdout.log` - MCP server output
- `server_stderr.log` - MCP server errors
- `audit.jsonl` - MCP tool execution audit trail

---

## Connecting Multiple Clients

Once the server is running, multiple clients can connect:

### Claude Desktop
Edit `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ags": {
      "command": "python",
      "args": ["d:\\CCC 2.0\\AI\\agent-governance-system\\MCP\\server.py"]
    }
  }
}
```

### VSCode Extensions
Extensions can connect via the same stdio interface.

### Your Swarm Agents
Governor and Ant Workers use the ledger files in `CONTRACTS/_runs/mcp_logs/`:
- `directives.jsonl` - Tasks from President
- `task_queue.jsonl` - Tasks to workers
- `task_results.jsonl` - Results from workers

---

## Troubleshooting

**Server won't start:**
```powershell
# Check logs
Get-Content CONTRACTS\_runs\mcp_logs\server_stderr.log -Tail 20

# Rebuild Cortex manually
python CORTEX\cortex.build.py
```

**Server keeps restarting:**
```powershell
# Stop autostart
.\MCP\autostart.ps1 -Stop

# Check what's failing
.\MCP\autostart.ps1 -Status
```

**Can't connect from clients:**
- Verify server is running: `.\MCP\autostart.ps1 -Status`
- Check server PID file exists: `CONTRACTS\_runs\mcp_logs\server.pid`
- Ensure clients use correct Python path and script path
