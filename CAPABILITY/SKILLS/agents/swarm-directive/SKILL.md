# Skill: swarm-directive

**Version:** 0.1.0

**Status:** Draft

**required_canon_version:** ">=2.8.0 <3.0.0"

# Swarm Directive Skill

Send tasks to your CATALYTIC-DPT swarm from Claude Code, Kilo CLI, or Cline CLI.

## Quick Start

### Option 1: Direct CLI Command (Simplest)

```bash
cd "d:\CCC 2.0\AI\agent-governance-system"

# Create input
cat > /tmp/swarm_task.json << 'EOF'
{
  "directive": "Analyze the MCP architecture",
  "task_type": "research"
}
EOF

# Send to swarm
python SKILLS/swarm-directive/run.py /tmp/swarm_task.json /tmp/result.json

# View result
cat /tmp/result.json
```

### Option 2: With Result Waiting

```bash
cat > /tmp/swarm_task.json << 'EOF'
{
  "directive": "Analyze code structure",
  "task_type": "research",
  "wait_for_result": true,
  "timeout": 30
}
EOF

python SKILLS/swarm-directive/run.py /tmp/swarm_task.json /tmp/result.json
```

### Option 3: Kilo/Cline Integration

In your Kilo or Cline CLI, run:

```bash
python SKILLS/swarm-directive/run.py input.json output.json
```

## Input Format

```json
{
  "directive": "Your task description here",
  "task_type": "research|file_operation|code_adapt|validate",
  "wait_for_result": false,
  "timeout": 30
}
```

**Fields:**
- `directive` (required): Task description to send to the swarm
- `task_type` (optional): Type of task. Default: "research"
- `wait_for_result` (optional): Block until task completes. Default: false
- `timeout` (optional): Max seconds to wait for result. Default: 30

## Output Format

```json
{
  "status": "success",
  "task_id": "swarm-1234567890",
  "task_status": "dispatched|pending|completed|failed",
  "message": "Human-readable status",
  "task_spec": {...},
  "result": {...}
}
```

## Examples

### Example 1: Fire and Forget

Send a task without waiting for completion:

```bash
cat > task.json << 'EOF'
{
  "directive": "Copy README.md to backups/README-backup.md",
  "task_type": "file_operation"
}
EOF

python SKILLS/swarm-directive/run.py task.json result.json

# Task dispatched! Check CONTRACTS/_runs/mcp_ledger/task_results.jsonl for completion
```

### Example 2: Wait for Completion

Block until task is done:

```bash
cat > task.json << 'EOF'
{
  "directive": "Analyze the swarm architecture",
  "task_type": "research",
  "wait_for_result": true,
  "timeout": 60
}
EOF

python SKILLS/swarm-directive/run.py task.json result.json

# Returns immediately when task completes or timeout
```

### Example 3: From Claude Code

```python
import subprocess
import json

task = {
    "directive": "Read and summarize ROADMAP.md",
    "task_type": "research",
    "wait_for_result": True,
    "timeout": 45
}

with open("/tmp/input.json", "w") as f:
    json.dump(task, f)

result = subprocess.run(
    ["python", "SKILLS/swarm-directive/run.py", "/tmp/input.json", "/tmp/output.json"],
    cwd="d:\\CCC 2.0\\AI\\agent-governance-system"
)

with open("/tmp/output.json") as f:
    output = json.load(f)
    print(f"Task: {output['task_id']}")
    print(f"Status: {output['task_status']}")
```

## Prerequisites

Before using this skill, start the swarm:

```bash
python CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/launch_swarm.py start
```

Check swarm status:

```bash
python CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/launch_swarm.py status
```

Stop swarm:

```bash
python CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/launch_swarm.py stop
```

## Monitoring

Watch tasks in real-time:

```bash
# Watch directives sent to Governor
tail -f CONTRACTS/_runs/mcp_ledger/directives.jsonl

# Watch task queue
tail -f CONTRACTS/_runs/mcp_ledger/task_queue.jsonl

# Watch results
tail -f CONTRACTS/_runs/mcp_ledger/task_results.jsonl
```

## Error Handling

If you get an error:

1. **"Swarm not running"** - Start with `launch_swarm.py start`
2. **"Task timed out"** - Increase `timeout` value
3. **"Governor not responding"** - Check `launch_swarm.py status`

## Architecture

```
Kilo/Cline CLI
    ↓
swarm-directive/run.py
    ↓ (sends via MCP)
MCP Server
    ↓ (writes to ledger)
CONTRACTS/_runs/mcp_ledger/directives.jsonl
    ↓
Governor (reads directive)
    ↓ (dispatches task)
Ant Worker (executes task)
    ↓ (writes result)
CONTRACTS/_runs/mcp_ledger/task_results.jsonl
    ↓ (read by skill)
Kilo/Cline gets result
```
