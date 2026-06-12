<!-- CONTENT_HASH: a59474179532161f11d70450ec5d2894068abcab4e41a6a691141a001c169b5c -->

# MCP Server Quick Start

The AGS MCP server runs over stdio: your MCP client launches it on demand.
There is no daemon to start or stop.

---

## Connect a Client

### Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS),
adjusting the repository path for your machine:

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

Restart the client. The AGS tools appear automatically.

A reference config lives at `CAPABILITY/MCP/claude_desktop_config.json`.

---

## Manual / Foreground Run

```cmd
CAPABILITY\MCP\start_simple.cmd
```

Runs the server over stdio in the current window. Ctrl+C to stop.

For a persistent shared instance (PID file managed), use:

```powershell
python CAPABILITY\MCP\server_wrapper.py
```

---

## Verify It Works

```powershell
# Built-in self test (initialize, tools, resources, prompts)
python LAW\CONTRACTS\ags_mcp_entrypoint.py --test

# Governance smoke test (exits non-zero on failure)
python CAPABILITY\MCP\verify_governance.py
```

**Check logs:**

```powershell
Get-Content LAW\CONTRACTS\_runs\mcp_logs\audit.jsonl -Tail 20
```

---

## Who Can Connect?

Any MCP-capable client can spawn its own server instance over stdio:

- Claude Desktop
- VS Code extensions
- Custom scripts and harnesses (e.g. hermes-harness)

All instances share:

- Same cassette network and cortex index
- Same governance rules (preflight, admission, critic)
- Same audit trail (`LAW/CONTRACTS/_runs/mcp_logs/`)
- Same skill library

---

## Next Steps

- **Full setup guide:** [README.md](README.md)
- **Protocol details:** [MCP_SPEC.md](MCP_SPEC.md)
