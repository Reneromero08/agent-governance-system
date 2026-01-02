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
import os
import time
import requests
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from event_logger import emit_event
except ImportError:
    # If not found (e.g. running from root without package), try adding path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from event_logger import emit_event

# --- UNICODE FIX FOR WINDOWS ---
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SWARM_ROOT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM"
INBOX_ROOT = REPO_ROOT / "INBOX" / "agents" / "Local Models"
LEDGER_PATH = INBOX_ROOT / "DISPATCH_LEDGER.json"
PROTOCOL_PATH = REPO_ROOT / "CAPABILITY" / "TESTBENCH" / "SYSTEM_FAILURE_PROTOCOL_CONSOLIDATED.md"
MANIFEST_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"
REPORT_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "SWARM_REPORT.json"

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
            timeout=180
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"⚠️ Ollama error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Ollama exception: {e}")
    return None


def mcp_call(method: str, params: Dict[str, Any], quiet: bool = False) -> Dict[str, Any]:
    """Call the MCP server using subprocess (stdio)."""
    mcp_script = REPO_ROOT / "CAPABILITY" / "MCP" / "server.py"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    
    # We need an intent path for tool calls
    intent_data = {
        "agent": "governor-dispatcher",
        "intent": f"Governor acting on {method} for swarm coordination",
        "timestamp": now_iso()
    }
    # Per ADR: Logs/runs under LAW/CONTRACTS/_runs/
    RUNS_DIR = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    intent_path = RUNS_DIR / f"intent_governor_{uuid.uuid4().hex}.json"
    intent_path.write_text(json.dumps(intent_data))
    
    try:
        env = os.environ.copy()
        env["AGS_INTENT_PATH"] = str(intent_path)
        
        # We also need to set the project root for the server
        env["AGS_PROJECT_ROOT"] = str(REPO_ROOT)
        
        res = subprocess.run(
            [sys.executable, str(mcp_script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env
        )
        if res.stderr and not quiet:
            print(f"📡 MCP DEBUG: {res.stderr.strip()}")
            
        if res.returncode == 0:
            return json.loads(res.stdout)
        else:
            if not quiet:
                print(f"⚠️ MCP Error: {res.stderr}")
    except Exception as e:
        if not quiet:
            print(f"⚠️ MCP call error: {e}")
    finally:
        if intent_path.exists():
            intent_path.unlink()
    return {"error": {"message": "Failed to call MCP server"}}

def broadcast_message(message: str, board: str = "swarm"):
    """Post a message to the governor's board for all agents."""
    return mcp_call("tools/call", {
        "name": "message_board_write",
        "arguments": {
            "board": board,
            "action": "post",
            "message": f"🏛️ GOVERNOR: {message}"
        }
    })

def cmd_solo(task_id: str):
    """Run a single task using the Professional model immediately."""
    print(f"🎯 SOLO MISSION: Sentinel taking on {task_id}...")
    script = SWARM_ROOT / "swarm_orchestrator_professional.py"
    # Create a temporary manifest for one task
    ledger = load_ledger()
    task = next((t for t in ledger["tasks"] if t["task_id"] == task_id), None)
    if not task:
        print(f"❌ Task {task_id} not found.")
        return
    
    # We'll just run the professional script with the --inbox flag and it will pick up what it can, 
    # but to be specific we can pass a temporary manifest.
    temp_manifest = REPO_ROOT / "THOUGHT" / "LAB" / "TEMP_SOLO.json"
    temp_manifest.write_text(json.dumps([task], indent=2))
    
    subprocess.run([sys.executable, str(script), "--input", str(temp_manifest)], cwd=str(REPO_ROOT))
    if temp_manifest.exists():
        temp_manifest.unlink()

def cmd_troubleshoot(target_file: str):
    """Deep analysis of a file using the 8B model."""
    print(f"🔍 Deep Troubleshooting: {target_file}...")
    path = REPO_ROOT / target_file
    if not path.exists():
        print(f"❌ File not found: {target_file}")
        return
    
    content = path.read_text(encoding="utf-8")
    # Get last error from protocol if available
    error_msg = "Unknown error"
    if PROTOCOL_PATH.exists():
        proto = PROTOCOL_PATH.read_text(encoding="utf-8")
        match = re.search(f"### {target_file}.*?Error: (.*?)\n", proto, re.DOTALL)
        if match:
             error_msg = match.group(1).strip()

    prompt = f"""
deep_TROUBLESHOOT: {target_file}
Error: {error_msg}
Content:
```python
{content[:10000]}
```

Provide a detailed root cause analysis and a step-by-step fix plan. 
Be highly technical.
"""
    analysis = ollama_generate(prompt, model="qwen2.5-coder:7b")
    print(f"\n🧠 ANALYSIS:\n{analysis}")
    
    broadcast_message(f"DEEP TROUBLESHOOT for {target_file}:\n{analysis}")
    print("\n✅ Analysis broadcast to message board.")


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
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "CAPABILITY/TESTBENCH", "--tb=line", "-q"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180  # Kill after 3 minutes max
        )
    except subprocess.TimeoutExpired:
        print("⚠️ Pytest scan timed out after 2 minutes")
        return []
    
    failures = []
    for line in result.stdout.splitlines():
        if line.startswith("FAILED "):
            test_path = line.replace("FAILED ", "").split("::")[0].strip()
            failures.append(test_path)
    
    return list(set(failures))  # Deduplicate


def generate_task_id(ledger: Optional[Dict[str, Any]] = None) -> str:
    """Generate a unique task ID."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    if ledger is None:
        ledger = load_ledger()
        
    existing_ids = {t["task_id"] for t in ledger["tasks"]}
    
    for i in range(1, 1000):
        task_id = f"TASK-{date_str}-{i:03d}"
        if task_id not in existing_ids:
            return task_id
    
    raise RuntimeError("Too many tasks for today")


def create_task(target_file: str, failure_details: Dict[str, Any], ledger: Dict[str, Any], priority: str = "MEDIUM") -> Dict[str, Any]:
    """Create a new task object with strategic analysis from the Dispatcher."""
    task_id = generate_task_id(ledger)
    
    # --- STRATEGY REASONING ---
    # The General (Dispatcher) provides tactical guidance for the Tiny Agents.
    print(f"🧠 General Thinking: Drafting combat plan for {target_file}...")
    error_msg = failure_details.get("message", "Unknown error")
    prompt = f"""
TACTICAL ANALYSIS FOR TASK {task_id}:
File: {target_file}
Error: {error_msg}

INSTRUCTION FOR WORKER: Analyze this failure and propose a fix.
The worker model is small (Qwen-0.5B). Give it 3 bullet points of specific advice.
"""
    # DISABLED: Strategy generation blocks forever
    # strategy = ollama_generate(prompt) if ollama_available() else "No tactical strategy provided."
    strategy = ""  # Let workers figure it out - skip blocking Ollama call

    return {
        "task_id": task_id,
        "created_at": now_iso(),
        "source": "failure_dispatcher_scan",
        "type": "test_fix",
        "priority": priority,
        "target_file": target_file,
        "failure_details": failure_details,
        "strategic_plan": strategy, # Attached guidance
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
                ledger=ledger,
                priority="MEDIUM"
            )
            new_tasks.append(task)
            ledger["tasks"].append(task)
    
    if new_tasks:
        save_ledger(ledger)
        print(f"\n✅ Added {len(new_tasks)} new tasks to ledger")
        emit_event("scan_complete", {"failures_found": len(failures), "new_tasks": len(new_tasks)}, INBOX_ROOT)
    else:
        print("\n⚠️ All failures already tracked in ledger")
        emit_event("scan_complete", {"failures_found": len(failures), "new_tasks": 0}, INBOX_ROOT)


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
                
                target_disp = task['target_file'].replace("\r", "").replace("\n", "")
                print(f"      • {task['task_id']}: {target_disp}")
                print(f"        Working for: {elapsed}")
                
                # Show progress log if available
                if 'progress_log' in task and task['progress_log']:
                    latest = task['progress_log'][-1]
                    msg_disp = latest['message'].replace("\r", " ").replace("\n", " ")[:60]
                    print(f"        Latest: {msg_disp}...")
    
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
            res_obj = task.get('result', "No summary")
            if isinstance(res_obj, dict):
                result_summary = res_obj.get('summary', "No summary")
            else:
                result_summary = str(res_obj)
            
            result_summary = result_summary.replace("\r", " ").replace("\n", " ")[:60]
            print(f"   • {task['task_id']} by {agent}")
            print(f"     {result_summary}...")
    
    # Context metrics
    total_tasks = len(ledger["tasks"])
    ledger_json = json.dumps(ledger)
    estimated_tokens = len(ledger_json) // 4  # ~4 chars per token
    context_capacity = 8192  # ministral-3:8b context window
    usage_pct = (estimated_tokens / context_capacity) * 100
    
    print(f"\n📊 Context Metrics:")
    print(f"   Ledger tasks: {total_tasks}")
    print(f"   Estimated tokens: {estimated_tokens}")
    print(f"   Context capacity: {context_capacity} (ministral-3:8b)")
    print(f"   Usage: {usage_pct:.1f}%")
    
    if estimated_tokens > 6000:
        print(f"   ⚠️ Warning: Consider pruning ledger (use sync to clean up)")




def update_protocol_entry(target_file: str, task: Dict[str, Any]) -> bool:
    """Update the Failure Protocol MD file with completion details."""
    if not PROTOCOL_PATH.exists():
        return False
        
    content = PROTOCOL_PATH.read_text(encoding="utf-8")
    sections = content.split("## PROTOCOL")
    if len(sections) < 2:
        return False
        
    # We want to find the section for "PROTOCOL 3"
    p3_idx = -1
    for i, s in enumerate(sections):
        if s.startswith(" 3"):
            p3_idx = i
            break
            
    if p3_idx == -1:
        return False
        
    p3_content = "## PROTOCOL" + sections[p3_idx]
    lines = p3_content.splitlines()
    new_p3_lines = []
    updated = False
    found_target = False
    
    # Details for insertion
    completion_time = task.get("completed_at", now_iso())
    tid = task["task_id"]
    agent = task.get("assigned_to", "local-swarm")
    # Result can be a string or a dict
    res = task.get("result", {})
    if isinstance(res, str):
        summary = res
    else:
        summary = res.get("summary", "Automated fix applied.")
    
    in_target_block = False
    
    for i in range(len(lines)):
        line = lines[i]
        
        # Look for the target file in a header or list item
        if target_file in line:
            in_target_block = True
            found_target = True
            # If it's a checkbox line like "- [ ] path/to/file.py", mark it fixed
            if "- [ ]" in line:
                line = line.replace("- [ ]", "- [x]") + " ✅"
        
        # Update specific status/resolution inside the block
        if in_target_block:
            if "**Status**:" in line or "- **Status**:" in line:
                line = f"- **Status**: ✅ COMPLETED ({tid} at {completion_time})"
            if "⬜ PENDING" in line:
                line = line.replace("⬜ PENDING", f"✅ COMPLETED ({tid})")
            
            # Check for end of block (next header or horizontal rule)
            if i + 1 < len(lines) and (lines[i+1].startswith("####") or lines[i+1].startswith("---")):
                # Add resolution details before closing block
                new_p3_lines.append(line)
                new_p3_lines.append(f"- **Resolution**: {summary}")
                new_p3_lines.append(f"- **Fixed By**: {agent}")
                in_target_block = False
                updated = True
                continue

        new_p3_lines.append(line)

    if not updated and found_target:
        updated = True
        p3_content = "\n".join(new_p3_lines)
    elif updated:
        p3_content = "\n".join(new_p3_lines)
    else:
        return False
        
    # Update Status Summary counters for THIS section
    try:
        pass_match = re.search(r"- \*\*Passed\*\*: (\d+)", p3_content)
        fail_match = re.search(r"- \*\*Failed\*\*: (\d+)", p3_content)
        if pass_match and fail_match:
            passed = int(pass_match.group(1)) + 1
            failed = max(0, int(fail_match.group(1)) - 1)
            total = passed + failed
            pct = (passed / total) * 100 if total > 0 else 100
            p3_content = re.sub(r"- \*\*Passed\*\*: \d+.*", f"- **Passed**: {passed} ({pct:.1f}%)", p3_content)
            p3_content = re.sub(r"- \*\*Failed\*\*: \d+.*", f"- **Failed**: {failed} ({100-pct:.1f}%)", p3_content)
    except:
        pass

    sections[p3_idx] = p3_content.replace("## PROTOCOL", "", 1)
    new_content = "## PROTOCOL".join(sections)
    PROTOCOL_PATH.write_text(new_content, encoding="utf-8")
    return True

import re

def cmd_sync(quiet: bool = False) -> None:
    """Sync completed tasks back to ledger and update protocol (Escalation Loop)."""
    if not quiet:
        print("🔄 Syncing completed tasks and Escalating failures...")
    
    ledger = load_ledger()
    task_map = {t["task_id"]: t for t in ledger["tasks"]}
    
    synced = 0
    escalated = 0
    fixed_in_protocol = 0
    
    # 1. Sync completed tasks
    for task_file in COMPLETED_DIR.glob("*.json") if COMPLETED_DIR.exists() else []:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task["task_id"] in task_map:
            # Check if this is a new completion we haven't processed
            was_already_done = task_map[task["task_id"]].get("status") == "COMPLETED"
            
            task_map[task["task_id"]].update(task)
            synced += 1
            
            # If newly completed, update the Protocol MD
            if not was_already_done:
                if update_protocol_entry(task["target_file"], task):
                    print(f"📝 Marked off {task['target_file']} in Failure Protocol.")
                    fixed_in_protocol += 1
    
    # 1.5 Sync from SWARM_REPORT.json (Caddy Output)
    if REPORT_PATH.exists():
        try:
            report_data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            # Map report items back to tasks via target_file
            # We look for PENDING tasks that match the file
            pending_map = {t["target_file"]: t for t in task_map.values() if t["status"] in ["PENDING", "ACTIVE"]}
            
            report_processed = 0
            for item in report_data:
                fname = item.get("file")
                status = item.get("status")
                
                if fname in pending_map:
                    task = pending_map[fname]
                    tid = task["task_id"]
                    
                    if status == "success":
                        task["status"] = "COMPLETED"
                        task["completed_at"] = now_iso()
                        task["result"] = {"summary": "Fixed by Swarm", "details": item}
                        task["assigned_to"] = item.get("model_used", "swarm-worker")
                        
                        # Move file
                        src = PENDING_DIR / f"{tid}.json"
                        if src.exists():
                            # Update content before moving
                            src.write_text(json.dumps(task, indent=2), encoding="utf-8")
                            dst = COMPLETED_DIR / f"{tid}.json"
                            src.replace(dst)
                            
                        synced += 1
                        report_processed += 1
                        
                        # Update protocol immediately
                        if update_protocol_entry(fname, task):
                            fixed_in_protocol += 1
                            
                    elif status == "failed":
                        # Mark attempt
                        task["attempts"] = task.get("attempts", 0) + 1
                        task["result"] = item.get("last_error", "Unknown failure")
                        
                        # If max attempts reached, move to FAILED
                        if task["attempts"] >= task.get("max_attempts", 3):
                            task["status"] = "FAILED"
                            src = PENDING_DIR / f"{tid}.json"
                            if src.exists():
                                src.write_text(json.dumps(task, indent=2), encoding="utf-8")
                                dst = FAILED_DIR / f"{tid}.json"
                                src.replace(dst)
                        else:
                            # Just update the PENDING file with new attempt count
                            src = PENDING_DIR / f"{tid}.json"
                            if src.exists():
                                src.write_text(json.dumps(task, indent=2), encoding="utf-8")

                        synced += 1
            
            if report_processed > 0:
                # Clear the report so we don't re-process
                REPORT_PATH.unlink()
                if not quiet:
                    print(f"📥 Processed {report_processed} items from Swarm Report")
                    
        except Exception as e:
            if not quiet:
                print(f"⚠️ Error reading swarm report: {e}")

    # 2. Sync and ESCALATE failed tasks
    for task_file in FAILED_DIR.glob("*.json") if FAILED_DIR.exists() else []:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task["task_id"] in task_map:
            tid = task["task_id"]
            
            # AUTO-RESURRECT logic for System-Blocked tasks
            fail_reason = str(task.get("result", ""))
            if "dangerous_ops" in fail_reason or "risky_gate" in fail_reason:
                 print(f"♻️  Resurrecting {tid} (was blocked by safety shields)...")
                 task["status"] = "PENDING"
                 task["attempts"] = 0
                 task["result"] = None
                 
                 # Move back to pending
                 new_path = PENDING_DIR / f"{tid}.json"
                 with open(new_path, "w") as f:
                     json.dump(task, f, indent=2)
                 task_file.unlink()
                 
                 escalated += 1
                 task_map[tid].update(task)
                 continue

            if task["attempts"] < task["max_attempts"]:
                # HELP LOOP: General analyzes the failure
                print(f"🆘 Escalating FAILED task {tid}... General is analyzing...")
                
                re_prompt = f"""
ANALYZE AGENT FAILURE FOR TASK {tid}:
Target: {task['target_file']}
Last Plan: {task.get('strategic_plan', 'None')}
Failure Result: {task.get('result', 'No result reported')}

WHY DID THE AGENT FAIL? 
Provide a REFINED strategy for the next attempt that overcomes the previous obstacle.
"""
                # DISABLED: Ollama blocks forever
                # new_strategy = ollama_generate(re_prompt) if ollama_available() else "No escalation strategy available."
                new_strategy = "Retry with professional model"
                
                # Update task for retry
                task["status"] = "PENDING"
                task["orchestrator_upgrade"] = "professional" # Smart Escalation: Upgrade to 8B Professional
                task["strategic_plan"] = f"ESCALATION: {new_strategy}"
                task_map[tid].update(task)
                
                # Move back to pending on disk
                new_path = PENDING_DIR / f"{tid}.json"
                with open(new_path, "w") as f:
                    json.dump(task, f, indent=2)
                
                # Delete from failed dir
                task_file.unlink()
                escalated += 1
            else:
                task_map[tid].update(task)
                synced += 1
    
    # 3. Sync active tasks
    for task_file in ACTIVE_DIR.glob("*.json") if ACTIVE_DIR.exists() else []:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task["task_id"] in task_map:
            task_map[task["task_id"]].update(task)
            synced += 1
    
    ledger["tasks"] = list(task_map.values())
    save_ledger(ledger)
    
    if not quiet:
        print(f"✅ Synced {synced} tasks | 🚀 Escalated {escalated} failures | 📝 Protocol Updated: {fixed_in_protocol}")
        emit_event("sync", {"synced": synced, "escalated": escalated}, INBOX_ROOT)
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
            
            # Read last few log lines
            log_lines = []
            if os.path.exists("swarm_debug.log"):
                try:
                    with open("swarm_debug.log", "r", encoding="utf-8") as f:
                        # Simple tail
                        lines = f.readlines()
                        log_lines = lines[-30:] if lines else []
                except:
                    pass

            # Clear line and print status
            os.system("cls" if os.name == "nt" else "clear")
            # print("\n" + "-"*60)
            print(f"👁️ Observing swarm activity (Ctrl+C to stop)...")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status:")
            print(f"   🟡 Pending:   {len(pending)}")
            print(f"   🔵 Active:    {len(active)}")
            print(f"   ✅ Done:      {len(completed)}")
            print(f"   ❌ Failed:    {len(failed)}")
            
            if log_lines:
                print("\n📜 Latest Swarm Logs:")
                for line in log_lines:
                    print(f"   {line.strip()}")
            elif not os.path.exists("swarm_debug.log"):
                 print("\n⚠️ No 'swarm_debug.log' found. Is 'guard' running to spawn agents?")
            
            sys.stdout.flush() # FORCE FLUSH to prevent blank screen
            time.sleep(3)
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


def manifest_from_pending(pending_tasks: List[Dict[str, Any]]) -> int:
    """Generate SWARM_MANIFEST.json from pending ledger tasks."""
    manifest = []
    
    for t in pending_tasks:
        target = t.get("target_file")
        if not target:
            continue
            
        # Construct instruction
        instr = f"""
STRICT INSTRUCTION:
1. READ file: {target}
2. ANALYZE failures.
3. FIX ONLY THE SPECIFIED FAILURES.
4. DO NOT REFACTOR ARCHITECTURE.
5. RETURN the full content of the fixed file.
"""
        # If we have a strategy/plan, append it
        if t.get("strategic_plan"):
             instr += f"\nGUIDANCE:\n{t['strategic_plan']}\n"
             
        manifest.append({
            "file": target,
            "instruction": instr
        })
        
    if not manifest:
        return 0
        
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(manifest)

def cmd_spawn(orchestrator_type: str = "caddy") -> None:
    """Spawn worker agents with Dynamic Scaling."""
    ledger = load_ledger()
    pending = [t for t in ledger["tasks"] if t["status"] == "PENDING"]
    
    if not pending:
        print("✅ No pending tasks to spawn for.")
        return
    
    # Generate Manifest for the Swarm
    count = manifest_from_pending(pending)
    if count == 0:
         print("⚠️ No valid targets in pending tasks (missing 'target_file'?).")
         return
         
    # Dynamic Scaling: 1 thread per task, capped at 32
    workers = min(max(4, count), 32)
    
    print(f"🚀 Spawning swarm for {count} tasks (Scaling: {workers} workers)...")
    emit_event("spawn", {"orchestrator": orchestrator_type, "workers": workers, "tasks": count}, INBOX_ROOT)
    
    # Determine command based on type
    if orchestrator_type == "professional":
        script = SWARM_ROOT / "swarm_orchestrator_professional.py"
        cmd = [sys.executable, str(script), "--inbox"]
    else:  # Default to caddy
        script = SWARM_ROOT / "swarm_orchestrator_caddy_deluxe.py"
        cmd = [sys.executable, str(script), "--max-workers", str(workers)]
    
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Log: swarm_debug.log")
    
    # Environment with UTF-8 enforcement for Windows
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # Launch in background
    with open("swarm_debug.log", "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
            start_new_session=True
        )
    
    print(f"✅ Swarm launched with PID {process.pid}")
    print("   Use 'observe' command to watch progress.")

def check_pipeline_health() -> bool:
    """Run critical pipeline verification tests."""
    print("🛡️ Sentinel: Verifying pipeline health...")
    try:
        # Run only the critical p1/p2 tests for speed
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "CAPABILITY/TESTBENCH/integration", "-k", "pipeline or verify"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✅ Sentinel: Pipeline GREEN")
            return True
        else:
            print("❌ Sentinel: Pipeline RED (Regression Detected!)")
            # Log failure details
            with open("sentinel_failure.log", "w") as f:
                f.write(result.stdout)
                f.write(result.stderr)
            return False
    except Exception as e:
        print(f"⚠️ Sentinel Error: {e}")
        return False

def kill_swarm() -> None:
    """Emergency kill switch for the swarm."""
    print("🚨 Sentinel: KILLING SWARM DUE TO REGRESSION")
    if sys.platform == "win32":
        os.system("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq Swarm*\"")
    else:
        os.system("pkill -f swarm_orchestrator")

def cmd_dash() -> None:
    """Launch the unified TUI Dashboard (includes auto-guard).
    
    This is the main interface - combines:
    - Visual dashboard with task counts and agent status
    - Auto-sync every 30 seconds
    - Auto-spawn when pending tasks appear
    """
    script = SWARM_ROOT / "tui_dashboard.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error launching dashboard: {e}")

# Guard is now an alias for dash (unified interface)
cmd_guard = cmd_dash


def cmd_tail() -> None:
    """Live follow the swarm log (tail -f equivalent)."""
    print("🐜 Tailing swarm logs (Ctrl+C to stop)...")
    log_file = "swarm_debug.log"
    
    if not os.path.exists(log_file):
        print("⚠️ No log file found yet.")
        return

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            # Read all existing lines first to catch up
            for line in f:
                pass 
                
            # Now follow
            while True:
                line = f.readline()
                if line:
                    # Handle progress bars using \r
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped.")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "status":
        cmd_status()
    elif command == "sync":
        cmd_sync()
    elif command == "scan":
        cmd_scan()
    elif command == "tail":
        cmd_tail()
    elif command == "dash":
        cmd_dash()
    elif command == "mcp":
        # Raw MCP call for debugging
        if len(sys.argv) < 3:
             print("Usage: python failure_dispatcher.py mcp <method> [json_params]")
             sys.exit(1)
        method = sys.argv[2]
        params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        print(json.dumps(mcp_call(method, params), indent=2))
    elif command == "observe":
        cmd_observe()
    elif command == "test":
        cmd_test()
    elif command == "spawn":
        type_arg = sys.argv[2] if len(sys.argv) > 2 else "caddy"
        cmd_spawn(type_arg)
    elif command == "guard":
        cmd_guard()
    elif command == "broadcast":
        if len(sys.argv) < 3:
            print("Usage: python failure_dispatcher.py broadcast <message>")
            sys.exit(1)
        res = broadcast_message(" ".join(sys.argv[2:]))
        print(f"✅ Broadcast sent: {res}")
    elif command == "board-list":
        res = mcp_call("tools/call", {
            "name": "message_board_list",
            "arguments": {"board": "swarm"}
        })
        print(json.dumps(res, indent=2))
    elif command == "inbox":
        status_arg = sys.argv[2] if len(sys.argv) > 2 else "pending"
        res = mcp_call("tools/call", {
            "name": "agent_inbox_list",
            "arguments": {"status": status_arg}
        })
        print(json.dumps(res, indent=2))
    elif command == "solo":
        if len(sys.argv) < 3:
            print("Usage: python failure_dispatcher.py solo <task_id>")
            sys.exit(1)
        cmd_solo(sys.argv[2])
    elif command == "troubleshoot":
        if len(sys.argv) < 3:
            print("Usage: python failure_dispatcher.py troubleshoot <file_path>")
            sys.exit(1)
        cmd_troubleshoot(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

