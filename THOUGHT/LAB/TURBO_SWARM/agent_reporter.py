#!/usr/bin/env python3
"""
Agent Reporter - Integration layer for Antigravity to report to dispatcher

Allows the main IDE agent (Antigravity/Claude) to:
- Claim tasks from the INBOX
- Report work in progress
- Mark tasks as completed
- Communicate with the dispatcher
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
INBOX_ROOT = REPO_ROOT / "INBOX" / "agents" / "Local Models"
LEDGER_PATH = INBOX_ROOT / "DISPATCH_LEDGER.json"

PENDING_DIR = INBOX_ROOT / "PENDING_TASKS"
ACTIVE_DIR = INBOX_ROOT / "ACTIVE_TASKS"
COMPLETED_DIR = INBOX_ROOT / "COMPLETED_TASKS"
FAILED_DIR = INBOX_ROOT / "FAILED_TASKS"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def claim_task(task_id: str, agent_name: str = "Antigravity") -> bool:
    """Claim a task from PENDING and move to ACTIVE."""
    task_file = PENDING_DIR / f"{task_id}.json"
    
    if not task_file.exists():
        print(f"❌ Task {task_id} not found in PENDING_TASKS/")
        return False
    
    # Load task
    task = json.loads(task_file.read_text(encoding="utf-8"))
    
    # Update task
    task["status"] = "ACTIVE"
    task["assigned_to"] = agent_name
    task["claimed_at"] = now_iso()
    
    # Move to ACTIVE
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    active_file = ACTIVE_DIR / f"{task_id}.json"
    active_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Remove from PENDING
    task_file.unlink()
    
    print(f"✅ Claimed {task_id} for {agent_name}")
    print(f"   Target: {task['target_file']}")
    return True


def update_task(task_id: str, progress: str, details: Optional[dict] = None) -> bool:
    """Update task progress while it's ACTIVE."""
    task_file = ACTIVE_DIR / f"{task_id}.json"
    
    if not task_file.exists():
        print(f"❌ Task {task_id} not found in ACTIVE_TASKS/")
        return False
    
    task = json.loads(task_file.read_text(encoding="utf-8"))
    
    if "progress_log" not in task:
        task["progress_log"] = []
    
    task["progress_log"].append({
        "timestamp": now_iso(),
        "message": progress,
        "details": details or {}
    })
    
    task_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"📝 Updated {task_id}: {progress}")
    return True


def complete_task(task_id: str, result: dict) -> bool:
    """Mark task as COMPLETED and move to COMPLETED_TASKS."""
    task_file = ACTIVE_DIR / f"{task_id}.json"
    
    if not task_file.exists():
        print(f"❌ Task {task_id} not found in ACTIVE_TASKS/")
        return False
    
    task = json.loads(task_file.read_text(encoding="utf-8"))
    
    task["status"] = "COMPLETED"
    task["completed_at"] = now_iso()
    task["result"] = result
    
    # Move to COMPLETED
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    completed_file = COMPLETED_DIR / f"{task_id}.json"
    completed_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Remove from ACTIVE
    task_file.unlink()
    
    print(f"✅ Completed {task_id}")
    print(f"   Result: {result.get('summary', 'No summary')}")
    return True


def fail_task(task_id: str, error: str, retry: bool = True) -> bool:
    """Mark task as FAILED or return to PENDING for retry."""
    task_file = ACTIVE_DIR / f"{task_id}.json"
    
    if not task_file.exists():
        print(f"❌ Task {task_id} not found in ACTIVE_TASKS/")
        return False
    
    task = json.loads(task_file.read_text(encoding="utf-8"))
    
    task["attempts"] += 1
    task["last_error"] = error
    task["last_attempt_at"] = now_iso()
    
    if retry and task["attempts"] < task["max_attempts"]:
        # Return to PENDING for retry
        task["status"] = "PENDING"
        task["assigned_to"] = None
        
        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        pending_file = PENDING_DIR / f"{task_id}.json"
        pending_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
        
        print(f"⚠️ Task {task_id} failed (attempt {task['attempts']}/{task['max_attempts']})")
        print(f"   Returned to PENDING for retry")
    else:
        # Move to FAILED
        task["status"] = "FAILED"
        
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failed_file = FAILED_DIR / f"{task_id}.json"
        failed_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
        
        print(f"❌ Task {task_id} FAILED after {task['attempts']} attempts")
    
    # Remove from ACTIVE
    task_file.unlink()
    return True


def list_pending() -> list:
    """List all pending tasks."""
    if not PENDING_DIR.exists():
        return []
    
    tasks = []
    for task_file in PENDING_DIR.glob("*.json"):
        task = json.loads(task_file.read_text(encoding="utf-8"))
        tasks.append(task)
    
    return tasks


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands:")
        print("  list                           - List pending tasks")
        print("  claim <task_id> [agent_name]   - Claim a task")
        print("  update <task_id> <message>     - Update task progress")
        print("  complete <task_id> <summary>   - Mark task complete")
        print("  fail <task_id> <error>         - Mark task failed")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "list":
        tasks = list_pending()
        if not tasks:
            print("No pending tasks")
        else:
            print(f"Found {len(tasks)} pending tasks:")
            for task in tasks:
                print(f"  {task['task_id']}: {task['target_file']}")
    
    elif command == "claim":
        if len(sys.argv) < 3:
            print("Usage: claim <task_id> [agent_name]")
            sys.exit(1)
        task_id = sys.argv[2]
        agent_name = sys.argv[3] if len(sys.argv) > 3 else "Antigravity"
        claim_task(task_id, agent_name)
    
    elif command == "update":
        if len(sys.argv) < 4:
            print("Usage: update <task_id> <message>")
            sys.exit(1)
        task_id = sys.argv[2]
        message = " ".join(sys.argv[3:])
        update_task(task_id, message)
    
    elif command == "complete":
        if len(sys.argv) < 4:
            print("Usage: complete <task_id> <summary>")
            sys.exit(1)
        task_id = sys.argv[2]
        summary = " ".join(sys.argv[3:])
        complete_task(task_id, {"summary": summary, "completed_by": "Antigravity"})
    
    elif command == "fail":
        if len(sys.argv) < 4:
            print("Usage: fail <task_id> <error>")
            sys.exit(1)
        task_id = sys.argv[2]
        error = " ".join(sys.argv[3:])
        fail_task(task_id, error)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
