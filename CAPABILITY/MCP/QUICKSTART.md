# MCP Server Quick Start

⚡ **Zero-setup auto-start: Just connect and it works!**

---

## Auto-Start (Recommended)

**No manual start needed!** The server auto-starts on first interaction.

### For Claude Desktop

Use the config in `MCP/claude_desktop_config.json`:

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

**That's it!** Server starts automatically when Claude Desktop connects.

---

## Manual Start (Optional)

If you prefer to start manually:

```powershell
cd "d:\CCC 2.0\AI\agent-governance-system"

# Start server
.\CAPABILITY\MCP\autostart.ps1 -Start

# Verify it's running
.\CAPABILITY\MCP\autostart.ps1 -Status
```

---

## Common Commands

```powershell
.\CAPABILITY\MCP\autostart.ps1 -Start     # Start server
.\CAPABILITY\MCP\autostart.ps1 -Stop      # Stop server
.\CAPABILITY\MCP\autostart.ps1 -Status    # Check status
.\CAPABILITY\MCP\autostart.ps1 -Restart   # Restart server
```

---

## Install Autostart (Optional)

**Option 1: Windows Startup Folder (No Admin)**
1. Press `Win+R`, type `shell:startup`
2. Create shortcut to: `CAPABILITY\MCP\start_simple.cmd`

**Option 2: Task Scheduler (Admin Required)**
```powershell
# Run PowerShell as Administrator
.\CAPABILITY\MCP\autostart.ps1 -Install
```
Server now starts automatically on boot.

---

## Who Can Connect?

Once running, these can all connect simultaneously:

✅ **Claude Desktop** - via stdio config
✅ **Governor Agent** - via MCP ledger
✅ **Ant Workers** - via task queue
✅ **VSCode Extensions** - via stdio
✅ **Custom Scripts** - via stdio

All share:
- Same Cortex index
- Same governance rules
- Same audit trail
- Same skill library

---

## Troubleshooting

**Server won't start?**
```powershell
python MCP\server.py --test
```

**Check logs:**
```powershell
Get-Content LAW\CONTRACTS\_runs\mcp_logs\autostart.log -Tail 20
```

---

## Next Steps

- **Full Setup Guide:** [README.md](README.md)
- **Connection Status:** Check with `.\CAPABILITY\MCP\autostart.ps1 -Status`
- **Research Report:** [CATALYTIC-DPT/LAB/RESEARCH/MULTI_AGENT_MCP_COORDINATION.md](../CATALYTIC-DPT/LAB/RESEARCH/MULTI_AGENT_MCP_COORDINATION.md)

---

**Status:** ✅ Ready
**Test Result:** ALL TESTS PASSED
**Startup Methods:** 4 available
**Multi-Client:** Enabled
