#!/usr/bin/env python3
"""
TURBO SWARM DASHBOARD - Unified TUI + Guard
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static, Log
from textual.containers import Horizontal, Vertical
from textual.binding import Binding
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
import os
import time

# Fix Windows Unicode issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
INBOX_ROOT = REPO_ROOT / "INBOX" / "agents" / "Local Models"
LEDGER_PATH = INBOX_ROOT / "DISPATCH_LEDGER.json"
PENDING_DIR = INBOX_ROOT / "PENDING_TASKS"
ACTIVE_DIR = INBOX_ROOT / "ACTIVE_TASKS"
COMPLETED_DIR = INBOX_ROOT / "COMPLETED_TASKS"
FAILED_DIR = INBOX_ROOT / "FAILED_TASKS"
SWARM_LOG = REPO_ROOT / "swarm_debug.log"
DISPATCHER_SCRIPT = Path(__file__).parent / "failure_dispatcher.py"


def count_files(d: Path) -> int:
    try:
        return len(list(d.glob("*.json"))) if d.exists() else 0
    except Exception as e:
        print(f"Error counting files in {d}: {e}")
        return 0


class StatusWidget(Static):
    """Shows task counts."""
    
    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh_data)
    
    def refresh_data(self) -> None:
        p = count_files(PENDING_DIR)
        a = count_files(ACTIVE_DIR)
        c = count_files(COMPLETED_DIR)
        f = count_files(FAILED_DIR)
        now = datetime.now().strftime("%H:%M:%S")
        
        if f > 5:
            state = "!! CRITICAL"
        elif a > 0:
            state = ">> ACTIVE"
        elif p > 0:
            state = ".. PENDING"
        else:
            state = "-- IDLE"
        
        self.update(f"""+-----------------+
|  SWARM COMMAND  |
+-----------------+

TIME:  {now}
STATE: {state}

--- TASKS ---
PENDING:   {p:>3}
ACTIVE:    {a:>3}
DONE:      {c:>3}
FAILED:    {f:>3}

--- KEYS ---
[S] Scan  [C] Spawn
[G] Guard [K] Kill
[Q] Quit""")


class AgentsWidget(Static):
    """Shows agents and active tasks."""
    
    def compose(self) -> ComposeResult:
        yield Static("[AGENTS]", id="agents_header")
        yield DataTable(id="agents_table")
    
    def on_mount(self) -> None:
        table = self.query_one("#agents_table", DataTable)
        table.add_columns("Model", "Status", "Task", "Progress")
        table.zebra_stripes = True
        self.set_interval(2.0, self.refresh_data)
    
    def refresh_data(self) -> None:
        table = self.query_one("#agents_table", DataTable)
        table.clear()
        
        # Get active tasks
        active_tasks = {}
        if ACTIVE_DIR.exists():
            for f in ACTIVE_DIR.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    agent = data.get("assigned_to", "")
                    logs = data.get("progress_log", [])
                    progress = logs[-1]["message"][:25] if logs else "Starting..."
                    tid = data.get("task_id", "?")[-8:]
                    active_tasks[agent] = {"tid": tid, "progress": progress}
                except:
                    pass
        
        # Get Ollama models
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                models_data = json.loads(resp.read().decode())
                models = [m["name"] for m in models_data.get("models", [])]
        except:
            models = ["(Ollama offline)"]
        
        # Display each model
        if not models and not active_tasks:
            table.add_row("No agents", "--", "--", "--")
            return
        
        for model in models[:6]:  # Show max 6 models
            short_name = model.split(":")[0][-12:]
            # Check if this model is currently working
            is_busy = any(model in str(agent) or short_name in str(agent) 
                         for agent in active_tasks.keys())
            
            if is_busy:
                status = "BUSY"
                # Find the matching task
                for agent, info in active_tasks.items():
                    if model in agent or short_name in agent:
                        table.add_row(short_name, status, info["tid"], info["progress"])
                        break
            else:
                status = "READY"
                table.add_row(short_name, status, "--", "--")
        
        # Show any active tasks with unknown agents
        for agent, info in active_tasks.items():
            if not any(model in agent for model in models):
                short_agent = str(agent)[:12] if agent else "Unknown"
                table.add_row(short_agent, "BUSY", info["tid"], info["progress"])


class LogWidget(Log):
    """Shows swarm log."""
    
    def on_mount(self) -> None:
        self.last_pos = 0
        self.set_interval(0.5, self.tail_log)
        self.write_line("Waiting for swarm activity...")
    
    def tail_log(self) -> None:
        if not SWARM_LOG.exists():
            return
        try:
            with open(SWARM_LOG, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.last_pos)
                data = f.read(2048)
                if data:
                    self.last_pos = f.tell()
                    for line in data.split("\n"):
                        line = line.strip()
                        if line:
                            # Sanitize to ASCII-safe for Windows
                            safe_line = line.encode('ascii', 'replace').decode('ascii')
                            self.write_line(safe_line)
        except:
            pass


class SwarmDashboard(App):
    """Main dashboard app."""
    
    CSS = """
    Screen {
        background: #000000;
    }
    Header {
        background: #000000;
        color: #00ff00;
    }
    Footer {
        background: #000000;
        color: #00ff00;
    }
    StatusWidget {
        width: 22;
        background: #000000;
        border: solid #00aa00;
        color: #00ff00;
        padding: 1;
    }
    AgentsWidget {
        background: #000000;
        border: solid #00aa00;
        color: #00ff00;
        padding: 1;
    }
    LogWidget {
        height: 12;
        background: #000000;
        border: solid #00aa00;
        color: #00ff00;
    }
    DataTable {
        background: #000000;
        color: #00ff00;
    }
    DataTable > .datatable--header {
        background: #001100;
        color: #00ff00;
    }
    DataTable > .datatable--cursor {
        background: #003300;
        color: #00ff00;
    }
    #top {
        height: 1fr;
    }
    """
    
    TITLE = "TURBO SWARM"
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "scan", "Scan"),
        Binding("c", "spawn", "Spawn"),
        Binding("g", "toggle_guard", "Guard"),
        Binding("k", "kill_all", "Kill"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal(id="top"):
                yield StatusWidget()
                yield AgentsWidget()
            yield LogWidget()
        yield Footer()
    
    def on_mount(self) -> None:
        self.spawn_cooldown = 0
        self.sync_cooldown = 0
        self.prune_cooldown = 0
        self.guard_enabled = True  # Auto-guard on by default
        self.set_interval(5.0, self.guard_tick)
    
    def guard_tick(self) -> None:
        """Auto-guard: sync, prune, detect stuck tasks, and spawn."""
        if not self.guard_enabled:
            return  # Guard is paused
        
        now = time.time()
        
        # Auto-sync
        if now > self.sync_cooldown:
            self.run_cmd("sync", silent=True)
            self.sync_cooldown = now + 30
        
        # Auto-prune ledger (every 5 minutes)
        if now > self.prune_cooldown:
            self.prune_ledger()
            self.prune_cooldown = now + 300
        
        # Detect and reset stuck tasks (active > 30 minutes)
        self.reset_stuck_tasks()
        
        # Auto-spawn - use direct orchestrator launch
        p = count_files(PENDING_DIR)
        a = count_files(ACTIVE_DIR)
        if p > 0 and a < 2 and now > self.spawn_cooldown:
            self.spawn_swarm_direct()
            self.spawn_cooldown = now + 60
    
    def spawn_swarm_direct(self) -> None:
        """Launch orchestrator directly without going through dispatcher."""
        try:
            orchestrator = Path(__file__).parent / "swarm_orchestrator_caddy_deluxe.py"
            log_file = REPO_ROOT / "swarm_debug.log"
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            with open(log_file, "a", encoding="utf-8") as log:
                subprocess.Popen(
                    [sys.executable, str(orchestrator), "--max-workers", "4"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                    env=env,
                )
            
            try:
                self.query_one(LogWidget).write_line(">>> SWARM LAUNCHED")
            except:
                pass
        except Exception as e:
            try:
                self.query_one(LogWidget).write_line(f"[ERR] Spawn failed: {e}")
            except:
                pass
    
    def run_cmd(self, cmd: str, silent: bool = False) -> None:
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            subprocess.Popen(
                [sys.executable, str(DISPATCHER_SCRIPT), cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(REPO_ROOT),
                env=env,
            )
            if not silent:
                log = self.query_one(LogWidget)
                log.write_line(f">>> {cmd.upper()}")
        except:
            pass
    
    def reset_stuck_tasks(self) -> None:
        """Reset tasks that have been active for more than 30 minutes."""
        if not ACTIVE_DIR.exists():
            return
        
        now = datetime.now()
        stuck_threshold_minutes = 30
        
        for f in ACTIVE_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                claimed_at = data.get("claimed_at", "")
                if not claimed_at:
                    continue
                
                claimed = datetime.fromisoformat(claimed_at.replace('Z', '+00:00'))
                elapsed_minutes = (now.astimezone() - claimed).total_seconds() / 60
                
                if elapsed_minutes > stuck_threshold_minutes:
                    # Reset to PENDING
                    data["status"] = "PENDING"
                    data["assigned_to"] = None
                    data["attempts"] = data.get("attempts", 0)  # Keep attempt count
                    
                    # Move to pending
                    new_path = PENDING_DIR / f.name
                    new_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    f.unlink()
                    
                    # Log it
                    try:
                        log = self.query_one(LogWidget)
                        tid = data.get("task_id", "?")
                        log.write_line(f"[RESET] {tid} was stuck for {int(elapsed_minutes)}m")
                    except:
                        pass
            except:
                pass
    
    def prune_ledger(self) -> None:
        """Auto-prune ledger: archive completed/failed, trim plans."""
        if not LEDGER_PATH.exists():
            return
        
        try:
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
            ledger_size = LEDGER_PATH.stat().st_size
            
            # Only prune if ledger > 10KB
            if ledger_size < 10000:
                return
            
            archived = []
            active = []
            
            for task in ledger.get("tasks", []):
                status = task.get("status", "")
                if status in ["COMPLETED", "FAILED"]:
                    archived.append(task)
                else:
                    # Trim strategic plan to 200 chars
                    plan = task.get("strategic_plan", "")
                    if len(plan) > 200:
                        task["strategic_plan"] = plan[:200] + "..."
                    active.append(task)
            
            if not archived:
                return  # Nothing to prune
            
            # Update ledger
            ledger["tasks"] = active
            ledger["summary"] = {
                "total_dispatched": len(active),
                "pending": sum(1 for t in active if t.get("status") == "PENDING"),
                "active": sum(1 for t in active if t.get("status") == "ACTIVE"),
                "completed": 0,
                "failed": 0
            }
            LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            
            # Archive
            archive_path = INBOX_ROOT / "LEDGER_ARCHIVE.json"
            existing = []
            if archive_path.exists():
                try:
                    existing = json.loads(archive_path.read_text(encoding="utf-8"))
                except:
                    pass
            existing.extend(archived)
            archive_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            
            # Log
            try:
                log = self.query_one(LogWidget)
                log.write_line(f"[PRUNE] Archived {len(archived)} tasks, {len(active)} remain")
            except:
                pass
        except:
            pass
    
    def action_quit(self) -> None:
        self.exit()
    
    def action_scan(self) -> None:
        self.run_cmd("scan")
    
    def action_spawn(self) -> None:
        self.spawn_swarm_direct()
    
    def action_toggle_guard(self) -> None:
        self.guard_enabled = not self.guard_enabled
        status = "ON" if self.guard_enabled else "OFF"
        try:
            log = self.query_one(LogWidget)
            log.write_line(f"[GUARD] Auto-guard is now {status}")
        except:
            pass
    
    def action_kill_all(self) -> None:
        """Kill all running swarm orchestrator processes."""
        self.guard_enabled = False  # Stop auto-spawn
        
        try:
            if sys.platform == "win32":
                # Windows: use taskkill
                subprocess.run(
                    ["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq *swarm_orchestrator*"],
                    capture_output=True
                )
                # Also kill by command line pattern
                result = subprocess.run(
                    ["wmic", "process", "where", "commandline like '%swarm_orchestrator%'", "delete"],
                    capture_output=True
                )
            else:
                # Linux/Mac: use pkill
                subprocess.run(["pkill", "-f", "swarm_orchestrator"], capture_output=True)
            
            # Also reset any active tasks back to pending
            if ACTIVE_DIR.exists():
                for f in ACTIVE_DIR.glob("*.json"):
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        data["status"] = "PENDING"
                        data["assigned_to"] = None
                        new_path = PENDING_DIR / f.name
                        new_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                        f.unlink()
                    except:
                        pass
            
            try:
                log = self.query_one(LogWidget)
                log.write_line("[KILL] All swarms stopped, guard OFF")
            except:
                pass
        except Exception as e:
            try:
                self.query_one(LogWidget).write_line(f"[ERR] Kill failed: {e}")
            except:
                pass


if __name__ == "__main__":
    SwarmDashboard().run()
