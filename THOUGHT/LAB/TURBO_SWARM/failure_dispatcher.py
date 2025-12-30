#!/usr/bin/env python3
"""
Failure Dispatcher Agent (Powered by mistral:7b)

Monitors the test suite, tracks failures, and dispatches tasks to the 
Local Models inbox for automated fixing. Uses Ollama for intelligent analysis.

Usage:
    python failure_dispatcher.py scan          # Scan for failures and update ledger
    python failure_dispatcher.py dispatch      # Dispatch pending tasks to inbox
    python failure_dispatcher.py status        # Show current task status
    python failure_dispatcher.py sync          # Sync completed tasks back to protocol
    python failure_dispatcher.py observe       # Observe swarm activity in real-time
    python failure_dispatcher.py test          # Run verification tests on the pipeline
"""

import json
import subprocess
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
SWARM_ROOT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM"
INBOX_ROOT = REPO_ROOT / "INBOX" / "agents" / "Local Models"
LEDGER_PATH = INBOX_ROOT / "DISPATCH_LEDGER.json"
PROTOCOL_PATH = REPO_ROOT / "CAPABILITY" / "TESTBENCH" / "SYSTEM_FAILURE_PROTOCOL_CONSOLIDATED.md"

# Tasks live in INBOX for agents to pick up
PENDING_DIR = INBOX_ROOT / "PENDING_TASKS"
ACTIVE_DIR = INBOX_ROOT / "ACTIVE_TASKS"
COMPLETED_DIR = INBOX_ROOT / "COMPLETED_TASKS"
FAILED_DIR = INBOX_ROOT / "FAILED_TASKS"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
DISPATCHER_MODEL = "ministral-3:8b"  # Coordinator model (already installed)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ollama_generate(prompt: str, model: str = DISPATCHER_MODEL) -> Optional[str]:
    """Generate a response from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
    return None


def ollama_available() -> bool:
    """Check if Ollama is available."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def load_ledger() -> Dict[str, Any]:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return {
        "ledger_version": "1.0.0",
        "created_at": now_iso(),
        "last_updated": now_iso(),
        "summary": {"total_dispatched": 0, "pending": 0, "active": 0, "completed": 0, "failed": 0},
        "tasks": []
    }


def save_ledger(ledger: Dict[str, Any]) -> None:
    ledger["last_updated"] = now_iso()
    # Recalculate summary
    tasks = ledger["tasks"]
    ledger["summary"] = {
        "total_dispatched": len(tasks),
        "pending": sum(1 for t in tasks if t["status"] == "PENDING"),
        "active": sum(1 for t in tasks if t["status"] == "ACTIVE"),
        "completed": sum(1 for t in tasks if t["status"] == "COMPLETED"),
        "failed": sum(1 for t in tasks if t["status"] == "FAILED")
    }
    INBOX_ROOT.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pytest_collect() -> List[str]:
    """Run pytest and collect failing test paths."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "CAPABILITY/TESTBENCH", "--tb=no", "-q"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    failures = []
    for line in result.stdout.splitlines():
        if line.startswith("FAILED "):
            test_path = line.replace("FAILED ", "").split("::")[0].strip()
            failures.append(test_path)
    
    return list(set(failures))  # Deduplicate


def generate_task_id() -> str:
    """Generate a unique task ID."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    ledger = load_ledger()
    existing_ids = [t["task_id"] for t in ledger["tasks"]]
    
    for i in range(1, 1000):
        task_id = f"TASK-{date_str}-{i:03d}"
        if task_id not in existing_ids:
            return task_id
    
    raise RuntimeError("Too many tasks for today")


def create_task(target_file: str, failure_details: Dict[str, Any], priority: str = "MEDIUM") -> Dict[str, Any]:
    """Create a new task object."""
    return {
        "task_id": generate_task_id(),
        "created_at": now_iso(),
        "source": "failure_dispatcher_scan",
        "type": "test_fix",
        "priority": priority,
        "target_file": target_file,
        "failure_details": failure_details,
        "status": "PENDING",
        "assigned_to": None,
        "attempts": 0,
        "max_attempts": 3,
        "result": None
    }


def cmd_scan() -> None:
    """Scan for test failures and add new tasks to ledger."""
    print("🔍 Scanning for test failures...")
    failures = run_pytest_collect()
    
    if not failures:
        print("✅ No failures found!")
        return
    
    print(f"Found {len(failures)} failing test files:")
    for f in failures:
        print(f"  - {f}")
    
    ledger = load_ledger()
    existing_files = {t["target_file"] for t in ledger["tasks"] if t["status"] in ["PENDING", "ACTIVE"]}
    
    new_tasks = []
    for failure in failures:
        if failure not in existing_files:
            task = create_task(
                target_file=failure,
                failure_details={"error_type": "test_failure", "scanned_at": now_iso()},
                priority="MEDIUM"
            )
            new_tasks.append(task)
            ledger["tasks"].append(task)
    
    if new_tasks:
        save_ledger(ledger)
        print(f"\n✅ Added {len(new_tasks)} new tasks to ledger")
    else:
        print("\n⚠️ All failures already tracked in ledger")


def cmd_dispatch() -> None:
    """Dispatch pending tasks to the inbox."""
    ledger = load_ledger()
    pending = [t for t in ledger["tasks"] if t["status"] == "PENDING"]
    
    if not pending:
        print("No pending tasks to dispatch")
        return
    
    print(f"📤 Dispatching {len(pending)} tasks to inbox...")
    
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    
    for task in pending:
        task_file = PENDING_DIR / f"{task['task_id']}.json"
        task_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  → {task['task_id']}: {task['target_file']}")
    
    save_ledger(ledger)
    print(f"\n✅ Dispatched {len(pending)} tasks to PENDING_TASKS/")


def cmd_status() -> None:
    """Show current task status."""
    ledger = load_ledger()
    summary = ledger["summary"]
    
    print("=" * 60)
    print("FAILURE DISPATCHER STATUS")
    print("=" * 60)
    print(f"Dispatcher Model: {DISPATCHER_MODEL}")
    print(f"Ollama Available: {'✅ Yes' if ollama_available() else '❌ No'}")
    print(f"Last Updated: {ledger['last_updated']}")
    print()
    print(f"📊 Summary:")
    print(f"   Total Tasks:  {summary['total_dispatched']}")
    print(f"   🟡 Pending:   {summary['pending']}")
    print(f"   🔵 Active:    {summary['active']}")
    print(f"   ✅ Completed: {summary['completed']}")
    print(f"   ❌ Failed:    {summary['failed']}")
    print()
    
    # Check filesystem state
    pending_files = list(PENDING_DIR.glob("*.json")) if PENDING_DIR.exists() else []
    active_files = list(ACTIVE_DIR.glob("*.json")) if ACTIVE_DIR.exists() else []
    completed_files = list(COMPLETED_DIR.glob("*.json")) if COMPLETED_DIR.exists() else []
    failed_files = list(FAILED_DIR.glob("*.json")) if FAILED_DIR.exists() else []
    
    print(f"📁 Inbox State:")
    print(f"   PENDING_TASKS/:   {len(pending_files)} files")
    print(f"   ACTIVE_TASKS/:    {len(active_files)} files")
    print(f"   COMPLETED_TASKS/: {len(completed_files)} files")
    print(f"   FAILED_TASKS/:    {len(failed_files)} files")
    
    # Track active agents
    if active_files:
        print("\n🔵 Currently Active:")
        agents = {}
        for f in active_files:
            task = json.loads(f.read_text(encoding="utf-8"))
            agent = task.get('assigned_to', 'unknown')
            if agent not in agents:
                agents[agent] = []
            agents[agent].append(task)
        
        for agent, tasks in agents.items():
            print(f"\n   👤 {agent} ({len(tasks)} task{'s' if len(tasks) > 1 else ''}):")
            for task in tasks:
                claimed_at = task.get('claimed_at', 'unknown')
                elapsed = "unknown"
                if claimed_at != 'unknown':
                    try:
                        from datetime import datetime
                        claimed = datetime.fromisoformat(claimed_at.replace('Z', '+00:00'))
                        now = datetime.now(claimed.tzinfo)
                        elapsed_sec = (now - claimed).total_seconds()
                        if elapsed_sec < 60:
                            elapsed = f"{int(elapsed_sec)}s"
                        elif elapsed_sec < 3600:
                            elapsed = f"{int(elapsed_sec/60)}m"
                        else:
                            elapsed = f"{int(elapsed_sec/3600)}h {int((elapsed_sec%3600)/60)}m"
                    except:
                        pass
                
                print(f"      • {task['task_id']}: {task['target_file']}")
                print(f"        Working for: {elapsed}")
                
                # Show progress log if available
                if 'progress_log' in task and task['progress_log']:
                    latest = task['progress_log'][-1]
                    print(f"        Latest: {latest['message'][:60]}...")
    
    # Track completed agents
    if completed_files:
        print("\n✅ Recently Completed:")
        recent_completions = []
        for f in completed_files:
            task = json.loads(f.read_text(encoding="utf-8"))
            recent_completions.append(task)
        
        # Sort by completion time, most recent first
        recent_completions.sort(key=lambda t: t.get('completed_at', ''), reverse=True)
        
        for task in recent_completions[:5]:  # Show last 5
            agent = task.get('assigned_to', 'unknown')
            completed_at = task.get('completed_at', 'unknown')
            result_summary = task.get('result', {}).get('summary', 'No summary')[:60]
            print(f"   • {task['task_id']} by {agent}")
            print(f"     {result_summary}...")



def cmd_sync() -> None:
    """Sync completed tasks back to ledger and update protocol."""
    print("🔄 Syncing completed tasks...")
    
    ledger = load_ledger()
    task_map = {t["task_id"]: t for t in ledger["tasks"]}
    
    synced = 0
    
    # Sync completed tasks
    for task_file in COMPLETED_DIR.glob("*.json") if COMPLETED_DIR.exists() else []:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task["task_id"] in task_map:
            task_map[task["task_id"]].update(task)
            synced += 1
    
    # Sync failed tasks
    for task_file in FAILED_DIR.glob("*.json") if FAILED_DIR.exists() else []:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task["task_id"] in task_map:
            task_map[task["task_id"]].update(task)
            synced += 1
    
    # Sync active tasks
    for task_file in ACTIVE_DIR.glob("*.json") if ACTIVE_DIR.exists() else []:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task["task_id"] in task_map:
            task_map[task["task_id"]].update(task)
            synced += 1
    
    ledger["tasks"] = list(task_map.values())
    save_ledger(ledger)
    
    print(f"✅ Synced {synced} tasks")
    cmd_status()


def cmd_observe() -> None:
    """Observe swarm activity in real-time."""
    print("👁️ Observing swarm activity (Ctrl+C to stop)...")
    print()
    
    try:
        while True:
            # Check inbox state
            pending = list(PENDING_DIR.glob("*.json")) if PENDING_DIR.exists() else []
            active = list(ACTIVE_DIR.glob("*.json")) if ACTIVE_DIR.exists() else []
            completed = list(COMPLETED_DIR.glob("*.json")) if COMPLETED_DIR.exists() else []
            failed = list(FAILED_DIR.glob("*.json")) if FAILED_DIR.exists() else []
            
            # Clear line and print status
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"🟡 {len(pending)} pending | "
                  f"🔵 {len(active)} active | "
                  f"✅ {len(completed)} done | "
                  f"❌ {len(failed)} failed", end="", flush=True)
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nObservation stopped.")


def cmd_test() -> None:
    """Run verification tests on the pipeline."""
    print("🧪 Testing pipeline integration...")
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Ollama availability
    print("1. Ollama service...", end=" ")
    if ollama_available():
        print("✅ PASS")
        tests_passed += 1
    else:
        print("❌ FAIL (Ollama not running)")
        tests_failed += 1
    
    # Test 2: Ledger read/write
    print("2. Ledger read/write...", end=" ")
    try:
        ledger = load_ledger()
        save_ledger(ledger)
        print("✅ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL ({e})")
        tests_failed += 1
    
    # Test 3: Inbox directories
    print("3. Inbox directories...", end=" ")
    try:
        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
        COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        print("✅ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL ({e})")
        tests_failed += 1
    
    # Test 4: Pytest collection
    print("4. Pytest collection...", end=" ")
    try:
        failures = run_pytest_collect()
        print(f"✅ PASS ({len(failures)} failures detected)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL ({e})")
        tests_failed += 1
    
    # Test 5: Orchestrator import
    print("5. Caddy Deluxe import...", end=" ")
    try:
        sys.path.insert(0, str(SWARM_ROOT))
        from swarm_orchestrator_caddy_deluxe import main as caddy_main
        print("✅ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL ({e})")
        tests_failed += 1
    
    # Test 6: MCP Server import
    print("6. MCP Server import...", end=" ")
    try:
        mcp_path = REPO_ROOT / "THOUGHT" / "LAB" / "MCP"
        sys.path.insert(0, str(mcp_path))
        from server import MCPTerminalServer
        print("✅ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL ({e})")
        tests_failed += 1
    
    print()
    print("=" * 40)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 40)
    
    if tests_failed > 0:
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "scan":
        cmd_scan()
    elif command == "dispatch":
        cmd_dispatch()
    elif command == "status":
        cmd_status()
    elif command == "sync":
        cmd_sync()
    elif command == "observe":
        cmd_observe()
    elif command == "test":
        cmd_test()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

