---
name: mcp-startup
version: "1.0.0"
description: Starts the complete CATALYTIC-DPT MCP network. Launches Ollama server, MCP server, and optional Ant Workers. One command to connect your entire agent swarm.
compatibility: Python 3.8+, Ollama installed
---

# MCP Startup

Complete startup tool for the CATALYTIC-DPT Model Context Protocol network.

## Usage

```bash
# Start all components (Ollama + MCP server + Ant workers)
python scripts/startup.py --all

# Start only Ollama server
python scripts/startup.py --ollama-only

# Start only MCP server (assumes Ollama already running)
python scripts/startup.py --mcp-only

# Start MCP server + N Ant workers
python scripts/startup.py --mcp --ants 2

# Interactive mode (choose what to start)
python scripts/startup.py --interactive
```

## What Gets Started

| Component | Default Port | Purpose |
|-----------|--------------|---------|
| **Ollama Server** | 11434 | Local LFM2 model inference |
| **MCP Server** | stdio | JSON-RPC protocol for agent communication |
| **Ant Workers** | N/A | Local task executors polling MCP ledger |

## Prerequisites

- ✅ Ollama installed (`ollama --version`)
- ✅ LFM2 model loaded in Ollama
- ✅ MCP server configured
- ✅ Python 3.8+ with requests library

## Quick Start

```bash
# One command to bring everything online
cd d:\CCC\ 2.0\AI\agent-governance-system
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --all
```

## Component Details

### Ollama Server
- Runs LFM2-2.6B model inference
- Available at `http://localhost:11434/api/chat`
- Required for Ant workers to execute tasks

### MCP Server
- Manages task queue, results ledger, directives
- Implements JSON-RPC 2.0 over stdio
- Can be integrated with Claude Desktop

### Ant Workers
- Poll MCP ledger for pending tasks
- Send prompts to Ollama
- Report results back to MCP

## Health Checks

The startup script validates:
- ✅ Ollama running and accessible
- ✅ LFM2 model loaded
- ✅ MCP ledger directories exist
- ✅ Ant workers can connect to MCP

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Ensure Ollama is running
ollama serve &
# Load LFM2 model
ollama run lfm2.gguf
```

### "MCP server failed to start"
```bash
# Check ledger directory exists
mkdir -p CONTRACTS/_runs/mcp_ledger
# Verify permissions
ls -la CONTRACTS/
```

### "Ant workers not executing tasks"
```bash
# Check Ollama health
curl http://localhost:11434/api/tags
# Check MCP ledger for stuck tasks
ls -la CONTRACTS/_runs/mcp_ledger/
```

## Architecture

```
┌─────────────────────────────┐
│   Ollama Server (Port 11434) │  ← LFM2 model inference
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│   MCP Server (Stdio)         │  ← Task coordination
└────────────┬────────────────┘
             ↓
     ┌───────┼───────┐
     ↓       ↓       ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Ant-1   │ │ Ant-2   │ │ Ant-N   │  ← Task executors
└─────────┘ └─────────┘ └─────────┘
```

## Environment Variables

Optional configuration:
```bash
# Override default Ollama port
export OLLAMA_PORT=11434

# Override MCP ledger path
export MCP_LEDGER_PATH=CONTRACTS/_runs/mcp_ledger

# Set logging level
export LOG_LEVEL=DEBUG
```
