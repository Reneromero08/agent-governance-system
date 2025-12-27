# MCP Startup Skill - Setup Checklist

Use this checklist to ensure your MCP network is properly configured and ready to use.

## Pre-Startup Checklist

Before running the startup skill, verify:

- [ ] Ollama is installed
  ```bash
  ollama --version
  ```

- [ ] LFM2 model is available in Ollama
  ```bash
  ollama run lfm2.gguf
  # You may need to pull it first:
  # ollama pull lfm2
  ```

- [ ] Python 3.8+ is installed
  ```bash
  python --version
  ```

- [ ] Required Python libraries are available
  ```bash
  python -c "import requests; print('OK')"
  ```

- [ ] You have write access to the repo directory
  ```bash
  touch CONTRACTS/_runs/.test && rm CONTRACTS/_runs/.test
  ```

- [ ] CONTRACTS directory structure exists
  ```bash
  ls CONTRACTS/
  mkdir -p CONTRACTS/_runs/mcp_ledger
  ```

## Startup Process Checklist

### Step 1: Start the Network
- [ ] Open a terminal
- [ ] Navigate to repo root: `cd "d:\CCC 2.0\AI\agent-governance-system"`
- [ ] Run startup script: `python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --all`
- [ ] Wait for "MCP Network is online!" message
- [ ] Keep this terminal open

### Step 2: Start the MCP Server
- [ ] Open a **new** terminal
- [ ] Run: `python CATALYTIC-DPT/LAB/MCP/stdio_server.py`
- [ ] Look for initialization messages
- [ ] Keep this terminal open

### Step 3: Verify Components

#### Check Ollama
- [ ] In a third terminal, run:
  ```bash
  curl http://localhost:11434/api/tags
  ```
- [ ] Should see JSON response with models listed

#### Check MCP Ledger
- [ ] Run:
  ```bash
  ls CONTRACTS/_runs/mcp_ledger/
  ```
- [ ] Directory should exist and be writable

#### Check Model Connection
- [ ] Run:
  ```bash
  python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "Hello"
  ```
- [ ] Should see model response
- [ ] NOT an error message

## Full System Verification Checklist

Once startup is complete:

### Ollama Server
- [ ] Port 11434 is accessible
  ```bash
  curl -s http://localhost:11434/api/tags | head -20
  ```

- [ ] LFM2 model is loaded
  ```bash
  curl -s http://localhost:11434/api/tags | grep -i "lfm"
  ```

- [ ] Can process requests
  ```bash
  python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "test"
  ```

### MCP Server
- [ ] Running without errors
  - Check for error messages in its terminal
  - Look for initialization messages

- [ ] Ledger files being created
  ```bash
  ls -la CONTRACTS/_runs/mcp_ledger/
  ```

- [ ] Can receive commands
  - Ready for task dispatch
  - Ready to integrate with Claude Desktop

### Ant Workers
- [ ] Processes started
  ```bash
  ps aux | grep ant_agent
  ```

- [ ] Can connect to MCP
  - Check their terminal output
  - Should show polling messages

- [ ] Can execute tasks
  - Send a test task through MCP
  - Check execution output

## Integration Checklist

Ready to connect your full system?

- [ ] All startup scripts have completed
- [ ] All three terminal windows (startup, MCP, test) show success
- [ ] Model responds to test queries
- [ ] MCP ledger directory populated

### Claude Desktop Integration (Optional)
- [ ] Claude Desktop installed
- [ ] MCP server configuration file created/updated
- [ ] MCP server path correct in config
- [ ] Claude Desktop restarted
- [ ] Can see MCP tools in Claude

## Common Issues Checklist

### If Ollama Won't Start
- [ ] Ollama is installed: `ollama --version`
- [ ] Not already running: `ps aux | grep ollama`
- [ ] Model available: `ollama list`
- [ ] Port 11434 not in use: `lsof -i :11434` (macOS/Linux)

### If MCP Server Won't Start
- [ ] Script exists: `ls CATALYTIC-DPT/LAB/MCP/stdio_server.py`
- [ ] Python can run it: `python --version`
- [ ] Ledger directory created: `mkdir -p CONTRACTS/_runs/mcp_ledger`
- [ ] No permission errors: `ls -la CATALYTIC-DPT/LAB/MCP/`

### If Ant Workers Won't Connect
- [ ] MCP server is running
- [ ] Ollama is running
- [ ] Ledger directory is writable
- [ ] Check error messages in Ant terminal

## Cleanup Checklist

When stopping the system:

- [ ] Terminate Ant worker processes (Ctrl+C in their terminals)
- [ ] Terminate MCP server (Ctrl+C in its terminal)
- [ ] Terminate Ollama server (Ctrl+C or `killall ollama`)

Optional cleanup:
- [ ] Archive old logs: `mv CONTRACTS/_runs/mcp_ledger CONTRACTS/_runs/mcp_ledger.bak`
- [ ] Clear ledger: `rm CONTRACTS/_runs/mcp_ledger/*.jsonl`
- [ ] Reset tasks: `rm -rf CONTRACTS/_runs/mcp_ledger`

## Maintenance Checklist

For ongoing operation:

### Weekly
- [ ] Check ledger size: `du -sh CONTRACTS/_runs/mcp_ledger/`
- [ ] Review error logs: `grep ERROR CONTRACTS/_runs/mcp_ledger/operations.jsonl`
- [ ] Restart Ollama if memory issues

### Monthly
- [ ] Archive old logs: `mv CONTRACTS/_runs CONTRACTS/_runs_$(date +%Y%m%d)`
- [ ] Create new ledger: `mkdir -p CONTRACTS/_runs/mcp_ledger`
- [ ] Update skill documentation

### As Needed
- [ ] Update Ollama: `ollama update`
- [ ] Update LFM2 model: `ollama pull lfm2`
- [ ] Check for skill updates: `git status CATALYTIC-DPT/SKILLS/mcp-startup`

## Performance Checklist

Monitor performance:

- [ ] Ollama response time: `time python lfm2_runner.py "test"`
  - First run: 10-15 seconds (normal)
  - Subsequent: 5-10 seconds (normal)

- [ ] MCP ledger size: `du -sh CONTRACTS/_runs/mcp_ledger/`
  - Should grow slowly
  - Archive if > 1GB

- [ ] System resources: `top` or `Task Manager`
  - Ollama: ~2GB RAM
  - MCP: ~50MB RAM
  - Per Ant: ~100MB RAM

- [ ] Task completion time
  - Simple tasks: <10s
  - Complex tasks: varies

## Success Criteria

Your MCP network is ready when ALL of these are true:

✅ Ollama running on localhost:11434
✅ LFM2 model loaded and responding
✅ MCP server running without errors
✅ Ant workers polling for tasks
✅ Health checks all pass
✅ Model responds to test queries
✅ MCP ledger files being created
✅ No error messages in any terminal

**Once all checks pass: Your MCP network is online and ready to use!**

---

## Quick Reference

### Start Everything
```bash
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --all
```

### Test Model
```bash
python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "test"
```

### Check Ollama
```bash
curl http://localhost:11434/api/tags
```

### View Ledger
```bash
ls CONTRACTS/_runs/mcp_ledger/
tail CONTRACTS/_runs/mcp_ledger/operations.jsonl
```

### Restart Everything
```bash
killall ollama
# Ctrl+C in MCP and Ant terminals
# Re-run startup scripts
```

---

**Checklist Version:** 1.0.0
**Last Updated:** 2025-12-26
