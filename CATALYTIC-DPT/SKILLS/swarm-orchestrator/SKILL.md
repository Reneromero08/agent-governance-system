---
name: swarm-orchestrator
version: "0.1.0"
description: Launches and coordinates the CATALYTIC-DPT agent swarm. Use to start Governor and Ant Workers for distributed task execution.
compatibility: Python 3.8+, MCP server
---

# Swarm Orchestrator

Launches and coordinates Governor + Ant Workers.

## Usage

```bash
# Launch Governor
python scripts/poll_and_execute.py --role Governor

# Launch Ant Workers
python scripts/poll_and_execute.py --role Ant-1
python scripts/poll_and_execute.py --role Ant-2
```

## PowerShell Launcher

```powershell
.\scripts\launch_swarm.ps1              # Python mode
.\scripts\launch_swarm.ps1 -Mode cli    # Direct CLI mode
```

## Architecture

```
Claude → send_directive → Governor
Governor → dispatch_task → Ant Workers
Ant Workers → report_result → Governor
Governor → (aggregates) → Claude
```
