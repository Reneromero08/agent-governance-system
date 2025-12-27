# MCP Startup Skill - Installation & Setup

## Installation

The `mcp-startup` skill is already installed at:
```
CATALYTIC-DPT/SKILLS/mcp-startup/
```

No additional installation required.

## Using the Skill

### Method 1: Python (All Platforms)

```bash
# Navigate to repo root
cd "d:\CCC 2.0\AI\agent-governance-system"

# Start everything
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --all

# Or interactive mode
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --interactive

# Or specific components
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --ollama-only
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --mcp-only
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --ants 2
```

### Method 2: PowerShell (Windows)

```powershell
# Navigate to repo root
cd "d:\CCC 2.0\AI\agent-governance-system"

# Change execution policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run the script
.\CATALYTIC-DPT\SKILLS\mcp-startup\scripts\startup.ps1

# Or with options
.\CATALYTIC-DPT\SKILLS\mcp-startup\scripts\startup.ps1 -All
.\CATALYTIC-DPT\SKILLS\mcp-startup\scripts\startup.ps1 -OllamaOnly
```

## Verification

After running startup, verify all components:

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check MCP ledger exists
ls CONTRACTS/_runs/mcp_ledger/

# Test local model
python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "What is 2+2?"
```

Expected output:
```
Helper: Sending prompt to Ollama (lfm2.gguf)...
2 + 2 equals 4.
```

## Common Issues & Fixes

### Issue: "Ollama not found"
**Solution:**
```bash
# Install from https://ollama.ai/
# Then ensure it's in PATH
ollama --version
```

### Issue: "Model not loaded"
**Solution:**
```bash
# Ensure model is available
ollama run lfm2.gguf

# Verify
curl http://localhost:11434/api/tags
```

### Issue: "MCP server won't start"
**Solution:**
```bash
# Check script exists
ls CATALYTIC-DPT/LAB/MCP/stdio_server.py

# Check ledger directory
mkdir -p CONTRACTS/_runs/mcp_ledger

# Run manually
python CATALYTIC-DPT/LAB/MCP/stdio_server.py
```

### Issue: "Ant workers not connecting"
**Solution:**
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check MCP ledger
ls -la CONTRACTS/_runs/mcp_ledger/

# Run Ant worker manually with debug
python CATALYTIC-DPT/SKILLS/ant-worker/scripts/ant_agent.py --poll_interval 5
```

## Startup Modes Explained

### `--all` (Full Network)
- Starts Ollama server (if not running)
- Sets up MCP ledger
- Prompts to start MCP server
- Launches N Ant workers

**Best for:** Full swarm operation with all components

### `--ollama-only`
- Starts Ollama server
- Skips MCP and Ant workers

**Best for:** Just testing the local model

### `--mcp-only`
- Assumes Ollama already running
- Ensures MCP ledger exists
- Prompts to start MCP server
- Skips Ant workers

**Best for:** Testing MCP server setup

### `--interactive`
- Prompts you to choose what to start
- Best for learning or manual control

**Best for:** First-time setup

## Architecture Verification

After startup, your system should look like this:

```
┌─────────────────────────────────────┐
│ Claude Desktop (optional)            │
│ (connects to MCP server)             │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ MCP Server (stdio)                   │  ← Started: `python stdio_server.py`
│ Port: stdio (no HTTP port)           │
└────────────┬────────────────────────┘
             │
      ┌──────┼──────┬──────┐
      ↓      ↓      ↓      ↓
┌─────────┬──────┬──────┬──────────┐
│ Ollama  │ Ant-1│ Ant-2│ Ant-N    │
│ :11434  │      │      │          │
└─────────┴──────┴──────┴──────────┘
     ▲
     │
  lfm2_runner.py connects here
```

## Performance Notes

### Ollama Server
- Uses ~2GB RAM (LFM2-2.6B model)
- Processes one request at a time
- First request ~5-10 seconds (model loading)
- Subsequent requests ~2-5 seconds

### Ant Workers
- Lightweight Python processes
- Each polls MCP every 5 seconds
- ~100MB RAM each
- Can run 10+ workers on modern hardware

### MCP Server
- Manages all coordination
- Maintains JSONL ledger files
- ~50MB RAM
- Bottleneck: File I/O on ledger operations

## Security Notes

1. **Ollama Server**
   - Only listens on localhost:11434
   - Not exposed to network by default
   - Safe for local use

2. **MCP Ledger**
   - All task data stored in JSONL files
   - Immutable append-only design
   - Kept in `CONTRACTS/_runs/` (durable root)

3. **Ant Workers**
   - Execute tasks from MCP only
   - Can't modify critical paths (CMP-01 governance)
   - All operations logged with hash verification

## Cleanup

To stop all components:

```bash
# Kill Ollama
killall ollama

# Kill MCP server (Ctrl+C in its terminal)

# Kill Ant workers (Ctrl+C in their terminals)

# Clear old task logs (optional)
rm -rf CONTRACTS/_runs/mcp_ledger/*.jsonl
```

## Next Steps

1. Start all components: `python startup.py --all`
2. Open Claude Desktop and connect to your MCP server
3. Try a simple task: "Copy this file"
4. Watch the MCP ledger for task flow
5. Check Ant worker output in their terminals

---

For detailed architecture, see [SKILL.md](SKILL.md)
For troubleshooting, see [README.md](README.md)
