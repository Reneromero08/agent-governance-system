<!-- CONTENT_HASH: 06c3fec9b31e86c056666078c91ad0f2b851e4a3c1f1e99d619f4a4c443d5580 -->

# TURBO SWARM Agent Coordinator

**⚠️ COORDINATION SYSTEM: Agent code lives here, tasks live in INBOX.**

## Architecture
- **Agent Code**: `THOUGHT/LAB/TURBO_SWARM/` - orchestrators, dispatcher, coordinator
- **Task Queue**: `INBOX/agents/Local Models/` - pending, active, completed, failed tasks

## Purpose
This is the central coordination hub for the TURBO_SWARM. The Failure Dispatcher scans for test failures,
tracks them in the ledger, and dispatches tasks to the INBOX for local models to pick up.

## Agent Directory (TURBO_SWARM)
```
TURBO_SWARM/
├── COORDINATOR.md              # This file - coordination rules
├── AGENT_WORKFLOW_STATUS.md    # Integration status documentation
├── failure_dispatcher.py       # Failure scanning and task dispatch
└── swarm_orchestrator_*.py     # Orchestrator implementations
```

## Task Directory (INBOX)
```
INBOX/agents/Local Models/
├── DISPATCH_LEDGER.json        # Master ledger of all dispatched tasks
├── PENDING_TASKS/              # Queued but not yet started
├── ACTIVE_TASKS/               # Currently being worked on
├── COMPLETED_TASKS/            # Successfully completed
└── FAILED_TASKS/               # Failed after all retries
```

## Task Format
Each task is a JSON file with this structure:
```json
{
  "task_id": "TASK-2025-12-30-001",
  "created_at": "2025-12-30T13:45:00Z",
  "source": "SYSTEM_FAILURE_PROTOCOL_CONSOLIDATED.md",
  "type": "test_fix",
  "priority": "HIGH|MEDIUM|LOW",
  "target_file": "path/to/file.py",
  "failure_details": {
    "error_type": "ModuleNotFoundError",
    "error_message": "...",
    "line_number": 42,
    "fix_suggestion": "..."
  },
  "status": "PENDING|ACTIVE|COMPLETED|FAILED",
  "assigned_to": null,
  "attempts": 0,
  "max_attempts": 3,
  "result": null
}
```

## Coordination Rules

### 1. Claiming a Task
Before starting work, an agent MUST:
1. Move the task file from `PENDING_TASKS/` to `ACTIVE_TASKS/`
2. Update `status` to "ACTIVE"
3. Set `assigned_to` to agent identifier
4. Update `DISPATCH_LEDGER.json`

### 2. Completing a Task
When work is done, the agent MUST:
1. Update `status` to "COMPLETED"
2. Set `result` with verification details
3. Move task file to `COMPLETED_TASKS/`
4. Update `DISPATCH_LEDGER.json`

### 3. Failing a Task
If work fails, the agent MUST:
1. Increment `attempts`
2. If `attempts < max_attempts`: return to `PENDING_TASKS/`
3. If `attempts >= max_attempts`: move to `FAILED_TASKS/`, set `status` to "FAILED"
4. Update `DISPATCH_LEDGER.json`

### 4. Conflict Avoidance
- Only ONE agent may claim a task
- Check-and-move must be atomic (first to move wins)
- If a task is in `ACTIVE_TASKS/`, do not touch it

## Current Status
- **Pending**: 0 tasks
- **Active**: 0 tasks
- **Completed**: 0 tasks
- **Failed**: 0 tasks

Last updated: 2025-12-30T13:45:00Z
