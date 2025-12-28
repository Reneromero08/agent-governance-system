# Auto-Start MCP Server

**Zero-setup multi-agent coordination: Server starts automatically on first interaction.**

---

## How It Works

### **The Magic: ags_mcp_auto.py**

When any client connects (Claude Desktop, VSCode, custom scripts), the auto-start entrypoint:

1. **Checks** if MCP server is running
2. **Starts** it if needed (via `autostart.ps1`)
3. **Connects** you to the running server
4. **All automatic** - no manual intervention

```
Client connects → Check PID file → Server not running? → Start it → Connect
                                 ↓
                          Server running? → Connect directly
```

---

## Setup (One-Time)

### For Claude Desktop

**Just use this config:**

```json
{
  "mcpServers": {
    "ags": {
      "command": "python",
      "args": ["D:/CCC 2.0/AI/agent-governance-system/CONTRACTS/_runs/ags_mcp_auto.py"],
      "cwd": "D:/CCC 2.0/AI/agent-governance-system"
    }
  }
}
```

Copy to: `%APPDATA%\Claude\claude_desktop_config.json`

**That's it!** No need to run `autostart.ps1` manually.

### For VSCode Extensions

Point your extension config to:
```
D:/CCC 2.0/AI/agent-governance-system/CONTRACTS/_runs/ags_mcp_auto.py
```

### For Custom Scripts

```python
import subprocess

result = subprocess.run([
    "python",
    "D:/CCC 2.0/AI/agent-governance-system/CONTRACTS/_runs/ags_mcp_auto.py"
], cwd="D:/CCC 2.0/AI/agent-governance-system")
```

---

## What Gets Auto-Started

When the first client connects:

1. **MCP Server** starts in background
2. **Cortex rebuild** (if configured - see `MCP/autostart_config.json`)
3. **PID tracking** (saved to `CONTRACTS/_runs/mcp_logs/server.pid`)
4. **Logging** starts (audit trail to `audit.jsonl`)

**Subsequent clients** just connect to the already-running server.

---

## Benefits

✅ **Zero manual setup** - Just connect and work
✅ **Shared server** - All clients use same instance
✅ **Governance enforced** - All agents subject to same rules
✅ **Audit trail** - All actions logged
✅ **Persistent** - Server keeps running after first start

---

## Advanced: How Auto-Start Works

### File: `CONTRACTS/_runs/ags_mcp_auto.py`

```python
def ensure_server_running():
    """Start server if not already running."""

    # Check PID file
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text())
        if process_is_alive(pid):
            return True  # Server running, we're good

    # Server not running - start it
    subprocess.run(["powershell", "-File", "MCP/autostart.ps1", "-Start"])
    time.sleep(2)  # Give it a moment
    return True

# Then run the actual MCP server
server_main()
```

### Graceful Fallback

If auto-start fails (e.g., PowerShell not available):
- Falls back to running the server directly
- This client becomes the server instance
- Still works, just not persistent

---

## Configuration

Edit `MCP/autostart_config.json`:

```json
{
  "components": {
    "mcp_server": {
      "enabled": true,
      "restart_on_failure": true
    },
    "cortex_rebuild": {
      "enabled": true,
      "on_startup": true  ← Rebuild Cortex on auto-start
    }
  }
}
```

---

## Troubleshooting

### "Server won't auto-start"

**Check PowerShell is available:**
```powershell
powershell -ExecutionPolicy Bypass -Command "echo 'OK'"
```

**Check autostart.ps1 exists:**
```powershell
Test-Path MCP\autostart.ps1
```

**Manually start once:**
```powershell
.\MCP\autostart.ps1 -Start
```

### "Multiple servers starting"

Only one server can run at a time (protected by PID file).

If you see errors:
```powershell
# Stop all instances
.\MCP\autostart.ps1 -Stop

# Clean PID file
Remove-Item CONTRACTS\_runs\mcp_logs\server.pid -Force

# Try auto-start again
```

### "Client can't connect"

**Verify server is running:**
```powershell
.\MCP\autostart.ps1 -Status
```

**Check logs:**
```powershell
Get-Content CONTRACTS\_runs\mcp_logs\autostart.log -Tail 20
Get-Content CONTRACTS\_runs\mcp_logs\server_stderr.log -Tail 20
```

---

## Comparison: Auto vs Manual

| Feature | Auto-Start | Manual Start |
|---------|-----------|--------------|
| **Setup** | Zero | Run `autostart.ps1 -Start` |
| **First connection** | 2-3 second delay | Instant |
| **Subsequent** | Instant | Instant |
| **Persistence** | Survives terminal close | Survives terminal close |
| **Boot autostart** | Optional (`-Install`) | Optional (`-Install`) |
| **Best for** | Development, quick use | Production, debugging |

**Recommendation:** Use auto-start for daily work. Use manual start if debugging server issues.

---

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Fast setup guide
- [README.md](README.md) - Full MCP documentation
- [autostart.ps1](autostart.ps1) - PowerShell autostart manager
- [CATALYTIC-DPT/LAB/RESEARCH/MULTI_AGENT_MCP_COORDINATION.md](../CATALYTIC-DPT/LAB/RESEARCH/MULTI_AGENT_MCP_COORDINATION.md) - Technical deep-dive

---

**Status:** ✅ Fully Implemented
**Files:**
- `CONTRACTS/_runs/ags_mcp_auto.py` - Auto-start entrypoint
- `MCP/autostart.ps1` - Background server manager
- `MCP/autostart_config.json` - Configuration
