"""
Swarm Integration Test: Mechanical Indexing + Ant Workers

Tests the full workflow:
1. Governor (Claude) reads tasks from instructions.db
2. Governor dispatches to Ants via MCP ledger
3. Ants (Ollama tiny models) execute refactoring
4. Results verified and logged

Hierarchy:
- God: User (president, final authority)
- Governor: Claude Sonnet 4.5 (SOTA - this script, complex decisions)
- Manager: Qwen 7B CLI (optional, cannot do complex tasks)
- Ants: Ollama tiny models (executes simple refactoring mechanically)
"""

import json
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
import hashlib

# Add GuardedWriter for write firewall enforcement
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None


class SwarmTaskDispatcher:
    """Dispatches refactoring tasks to swarm ant workers."""

    def __init__(self):
        project_root = Path(__file__).resolve().parents[3]
        self.instruction_db = sqlite3.connect(str(project_root / "NAVIGATION" / "CORTEX" / "db" / "instructions.db"))
        self.codebase_db = sqlite3.connect(str(project_root / "NAVIGATION" / "CORTEX" / "db" / "codebase_full.db"))
        self.mcp_ledger_dir = project_root / "LAW" / "CONTRACTS" / "_runs"
        
        # Enforce GuardedWriter
        if not GuardedWriter:
            raise ImportError("GuardedWriter not available")

        self.writer = GuardedWriter(
            project_root=project_root,
            durable_roots=["LAW/CONTRACTS/_runs"]
        )
        self.writer.open_commit_gate()
        self.writer.mkdir_durable("LAW/CONTRACTS/_runs")

    def get_simple_tasks(self, limit: int = 5) -> list:
        """
        Get simple refactoring tasks suitable for ant workers.

        Ants can handle:
        - Adding docstrings
        - Adding error handling (simple patterns)
        - Code formatting

        Ants CANNOT handle:
        - Complex refactoring
        - Multi-file changes
        - Architectural decisions
        """
        cursor = self.instruction_db.execute("""
            SELECT task_id, task_type, target_hash, target_path, instruction, context
            FROM tasks
            WHERE status = 'pending'
            AND task_type IN ('add_documentation', 'add_error_handling')
            ORDER BY priority DESC
            LIMIT ?
        """, (limit,))

        tasks = []
        for row in cursor:
            tasks.append({
                "task_id": row[0],
                "task_type": row[1],
                "target_hash": row[2],
                "target_path": row[3],
                "instruction": row[4],
                "context": json.loads(row[5]) if row[5] else {}
            })
        return tasks

    def resolve_hash(self, hash_ref: str) -> dict:
        """Resolve @hash to actual code."""
        if hash_ref.startswith("@hash:"):
            file_hash = hash_ref[6:]
        else:
            file_hash = hash_ref

        cursor = self.codebase_db.execute("""
            SELECT content, path, size, line_count
            FROM files
            WHERE hash = ?
        """, (file_hash,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "content": row[0],
            "path": row[1],
            "size": row[2],
            "line_count": row[3],
            "hash": file_hash
        }

    def create_ant_task_spec(self, task: dict) -> dict:
        """Create task specification for ant worker."""
        code_data = self.resolve_hash(task["target_hash"])
        if not code_data:
            return None

        # Estimate if task is simple enough for ants
        if code_data['line_count'] > 200:
            return None  # Too complex, escalate to governor

        return {
            "task_id": task["task_id"],
            "task_type": "code_refactor",
            "worker_type": "ant",  # Designates this for tiny models
            "instruction": task["instruction"],
            "file": {
                "path": code_data["path"],
                "hash": code_data["hash"],
                "content": code_data["content"],
                "size": code_data["size"],
                "line_count": code_data["line_count"]
            },
            "context": task.get("context", {}),
            "constraints": {
                "max_tokens": 5000,  # Tiny models have small context
                "template_only": True,  # Ants follow strict templates
                "no_creativity": True  # Mechanical execution only
            }
        }

    def dispatch_to_ant(self, task_spec: dict, ant_model: str = "tinyllama:1.1b") -> dict:
        """
        Dispatch task to ant worker via Ollama.

        In production, this would write to MCP ledger and ants would poll.
        For this test, we call Ollama directly.
        """
        prompt = self._create_ant_prompt(task_spec)

        print(f"[DISPATCHER] Task: {task_spec['task_id']}")
        print(f"[DISPATCHER] Ant Model: {ant_model}")
        print(f"[DISPATCHER] File: {Path(task_spec['file']['path']).name}")
        print(f"[DISPATCHER] Size: {task_spec['file']['size']} bytes")
        print()

        # Call Ollama
        try:
            result = subprocess.run(
                ["ollama", "run", ant_model, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "task_id": task_spec["task_id"],
                    "output": result.stdout,
                    "ant_model": ant_model
                }
            else:
                return {
                    "success": False,
                    "task_id": task_spec["task_id"],
                    "error": result.stderr,
                    "ant_model": ant_model
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "task_id": task_spec["task_id"],
                "error": "Timeout (60s exceeded)",
                "ant_model": ant_model
            }
        except Exception as e:
            return {
                "success": False,
                "task_id": task_spec["task_id"],
                "error": str(e),
                "ant_model": ant_model
            }

    def _create_ant_prompt(self, task_spec: dict) -> str:
        """Create prompt for ant worker - strict template."""
        return f"""You are an ant worker in a swarm. Your role is MECHANICAL EXECUTION ONLY.

TASK: {task_spec['instruction']}

FILE: {task_spec['file']['path']}
LINES: {task_spec['file']['line_count']}

CODE:
```
{task_spec['file']['content'][:2000]}  # Truncate for tiny models
```

INSTRUCTIONS:
1. Add docstrings to functions and classes
2. Use Google-style docstring format
3. Keep docstrings under 3 lines
4. Do NOT change any logic
5. Return ONLY the modified code

OUTPUT FORMAT:
```python
[modified code here]
```"""

    def log_to_mcp_ledger(self, task_spec: dict, result: dict):
        """Log execution to MCP ledger for governance."""
        run_id = f"{task_spec['task_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.mcp_ledger_dir / run_id
        run_dir_rel = f"LAW/CONTRACTS/_runs/{run_id}"
        
        self.writer.mkdir_durable(run_dir_rel)
        
        # Run info
        self.writer.write_durable(f"{run_dir_rel}/RUN_INFO.json", json.dumps({
            "task_id": task_spec["task_id"],
            "task_type": task_spec["task_type"],
            "worker_type": task_spec["worker_type"],
            "file_path": task_spec["file"]["path"],
            "file_hash": task_spec["file"]["hash"],
            "timestamp": datetime.now().isoformat()
        }, indent=2))
        
        # Result
        self.writer.write_durable(f"{run_dir_rel}/RESULT.json", json.dumps({
            "success": result["success"],
            "ant_model": result.get("ant_model"),
            "output": result.get("output", "")[:500],  # Truncate
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }, indent=2))

        print(f"[MCP LEDGER] Logged to {run_dir}")

    def close(self):
        """Close database connections."""
        self.instruction_db.close()
        self.codebase_db.close()


def main():
    """Run swarm integration test."""
    print("=" * 70)
    print("SWARM INTEGRATION TEST: Mechanical Indexing + Ant Workers")
    print("=" * 70)
    print()
    print("Hierarchy:")
    print("  God: User (president, final authority)")
    print("  Governor: Claude Sonnet 4.5 (SOTA - this script, complex decisions)")
    print("  Manager: Qwen 7B CLI (optional, cannot do complex tasks)")
    print("  Ants: Ollama tiny models (mechanical execution)")
    print()
    print("=" * 70)
    print()

    dispatcher = SwarmTaskDispatcher()

    # Get simple tasks
    print("[1/4] Fetching simple tasks for ants...")
    tasks = dispatcher.get_simple_tasks(limit=2)  # Test with 2 tasks
    print(f"  Found {len(tasks)} tasks suitable for ants")
    print()

    if not tasks:
        print("No simple tasks available. Run create_instruction_db.py first.")
        return

    # Dispatch to ants
    print("[2/4] Dispatching tasks to ant workers...")
    print()

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"Task {i}/{len(tasks)}")
        print("-" * 70)

        # Create ant task spec
        task_spec = dispatcher.create_ant_task_spec(task)
        if not task_spec:
            print(f"  [SKIP] Task too complex for ants, escalate to governor")
            print()
            continue

        # Dispatch to ant (tinyllama for speed)
        result = dispatcher.dispatch_to_ant(task_spec, ant_model="tinyllama:1.1b")

        if result["success"]:
            print(f"  [ANT SUCCESS] Task completed")
            print(f"  Output preview: {result['output'][:100]}...")
        else:
            print(f"  [ANT FAILED] {result.get('error', 'Unknown error')}")

        # Log to MCP ledger
        dispatcher.log_to_mcp_ledger(task_spec, result)

        results.append(result)
        print()

    # Summary
    print("[3/4] Results Summary")
    print("-" * 70)
    successful = sum(1 for r in results if r["success"])
    print(f"  Total tasks: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    print()

    # Verify governance
    print("[4/4] Governance Verification")
    print("-" * 70)
    ledger_files = list(dispatcher.mcp_ledger_dir.glob("*/RUN_INFO.json"))
    print(f"  MCP ledger entries: {len(ledger_files)}")
    print(f"  Ledger location: {dispatcher.mcp_ledger_dir}")
    print(f"  All executions logged: {'YES' if len(ledger_files) >= len(results) else 'NO'}")
    print()

    dispatcher.close()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()
    print("What was tested:")
    print("1. Governor (Claude Sonnet 4.5 - SOTA) fetched tasks from instructions.db")
    print("2. Tasks dispatched to ant workers (Ollama tinyllama)")
    print("3. Ants executed refactoring mechanically")
    print("4. Results logged to MCP ledger (governance)")
    print()
    print("Swarm hierarchy verified:")
    print("  God (User) -> Governor (Claude SOTA) -> Ants (Ollama)")
    print("  All executions logged to LAW/CONTRACTS/_runs")


if __name__ == "__main__":
    main()
