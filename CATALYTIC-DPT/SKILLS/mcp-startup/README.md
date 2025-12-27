# MCP Startup Skill

Complete startup automation for the CATALYTIC-DPT Model Context Protocol network.

## Quick Start

### Python (Recommended)
```bash
# One command to start everything
python scripts/startup.py --all

# Or choose what to start
python scripts/startup.py --ollama-only
python scripts/startup.py --mcp-only
python scripts/startup.py --interactive
```

### PowerShell (Windows)
```powershell
# Interactive mode
.\scripts\startup.ps1

# Or choose directly
.\scripts\startup.ps1 -All
.\scripts\startup.ps1 -OllamaOnly
.\scripts\startup.ps1 -MCPOnly
```

## What This Skill Does

1. **Starts Ollama Server** - Launches your local LFM2 model inference engine on port 11434
2. **Validates MCP Ledger** - Ensures the task queue and results directories exist
3. **Prompts to Start MCP Server** - Helps you launch the JSON-RPC protocol handler
4. **Starts Ant Workers** - Launches multiple local task executor agents
5. **Health Checks** - Verifies all components are running and accessible

## Components Started

| Component | Port | Purpose |
|-----------|------|---------|
| **Ollama** | 11434 | Local LFM2 model inference |
| **MCP Server** | stdio | JSON-RPC protocol / task coordination |
| **Ant Workers** | varies | Local task executors (polling the MCP) |

## Prerequisites

- ✅ Ollama installed on your machine
- ✅ LFM2 model available in Ollama
- ✅ Python 3.8+ with `requests` library
- ✅ MCP server script configured

## Full Network Flow

```
Claude (You)
    ↓
President Role (Claude Agent)
    ↓
Governor Role (Gemini / CLI)
    ↓
Ant Workers (LFM2 via Ollama)
    ↓
MCP Ledger (immutable task log)
```

All communication flows through the MCP ledger - no direct agent-to-agent connections.

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Ensure Ollama is running
ollama serve

# Load the model
ollama run lfm2.gguf
```

### "MCP server won't start"
```bash
# Check if the script exists
ls CATALYTIC-DPT/LAB/MCP/stdio_server.py

# Run it manually in a separate terminal
python CATALYTIC-DPT/LAB/MCP/stdio_server.py
```

### "Ant workers not connecting"
```bash
# Check the MCP ledger exists
ls CONTRACTS/_runs/mcp_ledger/

# Check Ollama is healthy
curl http://localhost:11434/api/tags

# Run a test prompt
python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "test"
```

## Environment Variables

Optional configuration via environment:
```bash
# Set Ollama port (default 11434)
export OLLAMA_PORT=11434

# Set MCP ledger path (default CONTRACTS/_runs/mcp_ledger)
export MCP_LEDGER_PATH=CONTRACTS/_runs/mcp_ledger

# Set logging level (default INFO)
export LOG_LEVEL=DEBUG
```

## Files in This Skill

- `SKILL.md` - Official documentation (frontmatter + details)
- `scripts/startup.py` - Main Python startup script
- `scripts/startup.ps1` - PowerShell launcher for Windows
- `README.md` - This file

## Next Steps After Startup

1. **Connect Claude Desktop** to your MCP server
   - Edit `~/.config/Claude/claude_desktop_config.json`
   - Add your MCP server configuration

2. **Send Tasks from Claude**
   - Use Claude to dispatch work to your Ant workers
   - Tasks flow: Claude → Governor → Ant Workers → Ollama

3. **Monitor Execution**
   - Watch the MCP ledger: `CONTRACTS/_runs/mcp_ledger/`
   - Check Ant worker output in their terminal windows
   - Review results in `task_results.jsonl`

## Development

To modify this skill:
- `scripts/startup.py` - Python startup logic
- `scripts/startup.ps1` - PowerShell startup logic
- `SKILL.md` - Update documentation
- Test with: `python scripts/startup.py --interactive`

---

**Author:** CATALYTIC-DPT MCP Team
**Version:** 1.0.0
**Status:** Production Ready
