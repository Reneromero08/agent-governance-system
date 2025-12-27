# MCP Startup Skill - Usage Guide

## What This Skill Does

The `mcp-startup` skill brings your entire CATALYTIC-DPT MCP network online with a single command. It:

1. ✅ Starts Ollama server (your local LFM2 model)
2. ✅ Prepares MCP ledger directories
3. ✅ Guides you to start the MCP coordination server
4. ✅ Launches Ant worker processes
5. ✅ Validates all components are working

## Installation (Already Done!)

The skill is already installed at:
```
CATALYTIC-DPT/SKILLS/mcp-startup/
```

No additional installation needed.

## Usage

### Method 1: Python (Recommended)

Navigate to your repo root and run:

```bash
# Start everything
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --all

# Or choose what to start
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --ollama-only
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --mcp-only
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --interactive
```

### Method 2: PowerShell (Windows)

```powershell
# Interactive
.\CATALYTIC-DPT\SKILLS\mcp-startup\scripts\startup.ps1

# Or direct options
.\CATALYTIC-DPT\SKILLS\mcp-startup\scripts\startup.ps1 -All
.\CATALYTIC-DPT\SKILLS\mcp-startup\scripts\startup.ps1 -OllamaOnly
```

## Startup Options Explained

### `--all` (Full Network)
Starts everything needed for a complete MCP swarm:
- Ollama server (localhost:11434)
- MCP ledger validation
- Prompts for MCP server launch
- 2 Ant workers (by default)

**Use when:** You want the complete system running

### `--ollama-only`
Just starts the Ollama server with the LFM2 model.

**Use when:** 
- Testing the model
- Developing
- Just want inference without task queue

### `--mcp-only`
Assumes Ollama is already running. Just validates MCP ledger.

**Use when:**
- Ollama is already started
- You only want to test the coordination layer

### `--interactive`
Asks you interactively what you want to start.

**Use when:**
- First time setup
- Learning the components
- Manual control

## Step-by-Step: Getting Started

### Step 1: Start the Network
```bash
cd "d:\CCC 2.0\AI\agent-governance-system"
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --all
```

Output:
```
MCP NETWORK STARTUP
==================================================
Starting Ollama server...
Ollama is already running
...
[SUCCESS] MCP Network is online!
```

### Step 2: Start the MCP Server
In a **new terminal**, run:
```bash
python CATALYTIC-DPT/LAB/MCP/stdio_server.py
```

This runs in the foreground. Keep this terminal open.

### Step 3: Verify Health
In yet another terminal, check:
```bash
# Test the local model
python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "2+2"

# Should output:
# Helper: Sending prompt to Ollama (lfm2.gguf)...
# 2 + 2 equals 4.
```

### Step 4: Send Tasks
Now you can:
- Connect Claude Desktop to your MCP server
- Send tasks from Claude
- Watch Ant workers execute them

## Output Examples

### Successful Startup
```
[INFO] MCP NETWORK STARTUP
[INFO] Starting Ollama server...
[OK] Ollama is already running

=== Health Check ===
[OK] Ollama server is running
[OK] LFM2 model is loaded
[OK] MCP ledger exists at CONTRACTS/_runs/mcp_ledger

[SUCCESS] MCP Network is online!
```

### Test Response from Model
```
Helper: Sending prompt to Ollama (lfm2.gguf)...
The sum of 2 and 2 is 4. This is a basic arithmetic operation...
```

## Troubleshooting

### Problem: "Cannot connect to Ollama"
```bash
# Solution: Start Ollama manually
ollama serve

# In another terminal, load the model
ollama run lfm2.gguf
```

### Problem: "MCP ledger does NOT exist"
```bash
# Solution: Create the directory
mkdir -p CONTRACTS/_runs/mcp_ledger
chmod 755 CONTRACTS/_runs
```

### Problem: "Ant workers not executing"
```bash
# Check 1: Is Ollama running?
curl http://localhost:11434/api/tags

# Check 2: Is MCP server running?
ps aux | grep stdio_server

# Check 3: Are ledger files being created?
ls CONTRACTS/_runs/mcp_ledger/
```

### Problem: "Port already in use"
```bash
# Ollama might be running. Check:
lsof -i :11434  # on macOS/Linux
netstat -ano | findstr :11434  # on Windows

# Kill the existing process if needed
killall ollama
```

## File Structure

```
mcp-startup/
├── SKILL.md              ← Official skill documentation
├── README.md             ← User guide & troubleshooting
├── INSTALLATION.md       ← Detailed setup instructions
├── USAGE.md              ← This file
├── QUICKREF.txt          ← One-page quick reference
└── scripts/
    ├── startup.py        ← Main Python launcher
    └── startup.ps1       ← PowerShell launcher
```

## Architecture After Startup

```
Your Computer
├── Ollama (port 11434)
│   └── LFM2-2.6B Model (inference)
├── MCP Server (stdio)
│   └── Task Queue & Ledger
├── Ant-1 (poll MCP → send to Ollama)
├── Ant-2 (poll MCP → send to Ollama)
└── Ant-N (poll MCP → send to Ollama)
```

All coordination through immutable JSONL ledger files.

## Performance

First startup: ~10-15 seconds (loading Ollama)
Subsequent startups: ~2-3 seconds
Per inference request: ~5-10 seconds

## Advanced Usage

### Start with Custom Number of Ants
```bash
python scripts/startup.py --all --ants 4
```

### Use Environment Variables
```bash
export OLLAMA_PORT=11434
export MCP_LEDGER_PATH=CONTRACTS/_runs/mcp_ledger
export LOG_LEVEL=DEBUG

python scripts/startup.py --all
```

### Run in Background (Advanced)
```bash
# Python: Use nohup or screen
nohup python scripts/startup.py --all &

# PowerShell: Run minimized
.\scripts\startup.ps1 -All
```

## Next Steps

1. ✅ Run the startup skill
2. ✅ Start the MCP server
3. ✅ Test the model with lfm2_runner.py
4. ✅ Connect Claude Desktop
5. ✅ Send your first task!

---

**Need help?** See [README.md](README.md) for troubleshooting or [QUICKREF.txt](QUICKREF.txt) for quick reference.
