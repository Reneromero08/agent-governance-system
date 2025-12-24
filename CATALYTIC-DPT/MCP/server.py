#!/usr/bin/env python3
"""
CATALYTIC-DPT MCP Server

Core infrastructure for multi-agent orchestration:
1. Terminal sharing (you see Claude's, Claude sees yours)
2. Skill execution (single source of truth, no drift)
3. File synchronization (hash-verified)
4. Immutable ledger (every action logged)

Governance: All changes via MCP, zero drift, bidirectional monitoring.
"""

import json
import sys
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# This would be replaced with actual MCP SDK
# For now, we structure it as a mock MCP server

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONTRACTS_DIR = PROJECT_ROOT / "CONTRACTS" / "_runs"
SKILLS_DIR = PROJECT_ROOT / "CATALYTIC-DPT" / "SKILLS"

class MCPTerminalServer:
    """MCP Server for terminal sharing and monitoring."""

    def __init__(self):
        self.terminals = {}  # terminal_id → TerminalSession
        self.ledger_path = CONTRACTS_DIR / "mcp_ledger"
        self.ledger_path.mkdir(parents=True, exist_ok=True)
        self.agents = {"Claude", "Gemini", "Grok"}

    def register_terminal(self, terminal_id: str, owner: str, cwd: str) -> Dict:
        """Register a terminal for sharing."""
        session = {
            "terminal_id": terminal_id,
            "owner": owner,
            "cwd": cwd,
            "created": datetime.now().isoformat(),
            "commands": [],
            "visible_to": list(self.agents),
            "status": "active"
        }
        self.terminals[terminal_id] = session

        self._log_operation({
            "operation": "terminal_register",
            "terminal_id": terminal_id,
            "owner": owner,
            "visible_to": list(self.agents)
        })

        return session

    def log_terminal_command(
        self,
        terminal_id: str,
        command: str,
        executor: str,
        output: Optional[str] = None,
        exit_code: Optional[int] = None
    ) -> Dict:
        """Log a command executed in a terminal."""

        if terminal_id not in self.terminals:
            return {"status": "error", "message": f"Terminal {terminal_id} not registered"}

        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "executor": executor,
            "output": output,
            "exit_code": exit_code,
            "visible_to": self.terminals[terminal_id]["visible_to"]
        }

        self.terminals[terminal_id]["commands"].append(entry)

        self._log_operation({
            "operation": "terminal_command",
            "terminal_id": terminal_id,
            "command": command,
            "executor": executor,
            "exit_code": exit_code,
            "timestamp": entry["timestamp"]
        })

        return {
            "status": "success",
            "terminal_id": terminal_id,
            "command_logged": command,
            "visible_to": entry["visible_to"]
        }

    def get_terminal_output(self, terminal_id: str) -> Dict:
        """Retrieve all commands and output from a terminal."""
        if terminal_id not in self.terminals:
            return {"status": "error", "message": f"Terminal {terminal_id} not found"}

        return {
            "status": "success",
            "terminal_id": terminal_id,
            "commands": self.terminals[terminal_id]["commands"],
            "visible_to": self.terminals[terminal_id]["visible_to"]
        }

    def execute_skill(
        self,
        skill_name: str,
        task_spec: Dict,
        executor: str,
        run_id: Optional[str] = None
    ) -> Dict:
        """Execute a skill via MCP (canonical source of truth)."""

        if run_id is None:
            run_id = f"{skill_name}-{uuid.uuid4().hex[:8]}"

        run_dir = CONTRACTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load skill definition
        skill_path = SKILLS_DIR / skill_name / "SKILL.md"
        if not skill_path.exists():
            return {
                "status": "error",
                "message": f"Skill {skill_name} not found at {skill_path}"
            }

        # 2. Load schema if exists
        schema_path = SKILLS_DIR / skill_name / "schema.json"
        skill_schema = None
        if schema_path.exists():
            with open(schema_path) as f:
                skill_schema = json.load(f)

        # 3. Validate task against schema
        if skill_schema:
            validation = self._validate_against_schema(task_spec, skill_schema)
            if not validation["valid"]:
                return {
                    "status": "error",
                    "message": "Task spec validation failed",
                    "errors": validation["errors"]
                }

        # 4. Prepare execution context
        execution_context = {
            "run_id": run_id,
            "skill": skill_name,
            "executor": executor,
            "task_spec": task_spec,
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "status": "running",
            "outputs": {},
            "ledger_dir": str(run_dir)
        }

        # 5. Log execution start
        self._log_operation({
            "operation": "skill_execute_start",
            "run_id": run_id,
            "skill": skill_name,
            "executor": executor,
            "task_spec": task_spec
        })

        # 6. Save task spec to ledger (immutable)
        with open(run_dir / "TASK_SPEC.json", "w") as f:
            json.dump(task_spec, f, indent=2)

        return {
            "status": "pending",
            "run_id": run_id,
            "skill": skill_name,
            "executor": executor,
            "ledger_dir": str(run_dir),
            "next_step": f"Call CATALYTIC-DPT/SKILLS/{skill_name}/run.py with inputs"
        }

    def file_sync(
        self,
        source: str,
        destination: str,
        executor: str,
        verify_hash: bool = True
    ) -> Dict:
        """Synchronize file via MCP (hash-verified)."""

        source_path = Path(source)
        dest_path = Path(destination)

        # 1. Verify source exists
        if not source_path.exists():
            return {
                "status": "error",
                "message": f"Source file not found: {source}"
            }

        # 2. Compute source hash
        source_hash = self._compute_hash(source_path)

        # 3. Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Copy file
        try:
            dest_path.write_bytes(source_path.read_bytes())
        except Exception as e:
            return {
                "status": "error",
                "message": f"Copy failed: {str(e)}"
            }

        # 5. Verify destination hash
        dest_hash = self._compute_hash(dest_path)

        hash_match = source_hash == dest_hash
        if verify_hash and not hash_match:
            # Remove corrupted file
            dest_path.unlink()
            return {
                "status": "error",
                "message": "Hash mismatch after copy (file removed)",
                "source_hash": source_hash,
                "dest_hash": dest_hash
            }

        # 6. Log operation
        self._log_operation({
            "operation": "file_sync",
            "source": source,
            "destination": destination,
            "executor": executor,
            "source_hash": source_hash,
            "dest_hash": dest_hash,
            "hash_verified": hash_match
        })

        return {
            "status": "success",
            "source": source,
            "destination": destination,
            "executor": executor,
            "source_hash": source_hash,
            "dest_hash": dest_hash,
            "hash_match": hash_match,
            "size_bytes": source_path.stat().st_size
        }

    def skill_complete(
        self,
        run_id: str,
        status: str,
        outputs: Dict,
        errors: Optional[List[str]] = None
    ) -> Dict:
        """Mark skill execution as complete."""

        run_dir = CONTRACTS_DIR / run_id
        if not run_dir.exists():
            return {
                "status": "error",
                "message": f"Run directory not found: {run_dir}"
            }

        # Save outputs
        with open(run_dir / "OUTPUTS.json", "w") as f:
            json.dump(outputs, f, indent=2)

        # Save errors if any
        if errors:
            with open(run_dir / "ERRORS.json", "w") as f:
                json.dump({"errors": errors}, f, indent=2)

        # Log completion
        self._log_operation({
            "operation": "skill_complete",
            "run_id": run_id,
            "status": status,
            "outputs": list(outputs.keys()),
            "error_count": len(errors) if errors else 0
        })

        return {
            "status": "success",
            "run_id": run_id,
            "skill_status": status,
            "ledger_dir": str(run_dir),
            "outputs_saved": len(outputs),
            "errors_logged": len(errors) if errors else 0
        }

    def get_ledger(self, run_id: Optional[str] = None) -> Dict:
        """Retrieve ledger entries."""
        ledger_file = self.ledger_path / "operations.jsonl"

        if not ledger_file.exists():
            return {
                "status": "success",
                "entries": [],
                "run_id_filter": run_id
            }

        entries = []
        with open(ledger_file) as f:
            for line in f:
                entry = json.loads(line)
                if run_id is None or entry.get("run_id") == run_id:
                    entries.append(entry)

        return {
            "status": "success",
            "entries": entries,
            "run_id_filter": run_id,
            "total_entries": len(entries)
        }

    # Private helper methods

    def _log_operation(self, operation: Dict) -> None:
        """Log operation to immutable ledger."""
        ledger_file = self.ledger_path / "operations.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            **operation
        }

        with open(ledger_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(65536)  # 64kb chunks
                if not data:
                    break
                sha.update(data)
        return sha.hexdigest()

    @staticmethod
    def _validate_against_schema(instance: Dict, schema: Dict) -> Dict:
        """Validate JSON instance against schema."""
        # This would use jsonschema library in production
        # For now, return simple validation
        return {
            "valid": True,
            "errors": []
        }


# Exported for MCP integration
mcp_server = MCPTerminalServer()


def register_mcp_tools():
    """Register all MCP tools."""
    return {
        "terminal_register": {
            "description": "Register a terminal for monitoring",
            "parameters": {
                "terminal_id": "Unique terminal ID",
                "owner": "Agent that owns the terminal",
                "cwd": "Current working directory"
            }
        },
        "terminal_log_command": {
            "description": "Log a command executed in a terminal",
            "parameters": {
                "terminal_id": "ID of the terminal",
                "command": "Command string",
                "executor": "Agent that executed it",
                "output": "Command output",
                "exit_code": "Exit code"
            }
        },
        "terminal_get_output": {
            "description": "Retrieve all output from a terminal",
            "parameters": {
                "terminal_id": "ID of the terminal"
            }
        },
        "skill_execute": {
            "description": "Execute a skill via MCP",
            "parameters": {
                "skill_name": "Name of the skill",
                "task_spec": "Task specification JSON",
                "executor": "Agent executing the skill",
                "run_id": "Optional run ID"
            }
        },
        "skill_complete": {
            "description": "Mark skill execution as complete",
            "parameters": {
                "run_id": "Run ID from skill_execute",
                "status": "success|failed",
                "outputs": "Output dictionary",
                "errors": "List of errors if any"
            }
        },
        "file_sync": {
            "description": "Synchronize file with hash verification",
            "parameters": {
                "source": "Source file path",
                "destination": "Destination file path",
                "executor": "Agent performing sync",
                "verify_hash": "Verify integrity"
            }
        },
        "get_ledger": {
            "description": "Retrieve immutable operations ledger",
            "parameters": {
                "run_id": "Optional filter by run ID"
            }
        }
    }


if __name__ == "__main__":
    # Test the MCP server
    import json

    server = MCPTerminalServer()

    # Test 1: Register terminal
    print("Test 1: Register terminal")
    result = server.register_terminal("user_vscode", "You", "/d/CCC 2.0/AI/agent-governance-system")
    print(json.dumps(result, indent=2))

    # Test 2: Log command
    print("\nTest 2: Log terminal command")
    result = server.log_terminal_command(
        "user_vscode",
        "python CATALYTIC-DPT/SKILLS/gemini-file-analyzer/run.py input.json output.json",
        "Gemini",
        "Analysis complete",
        0
    )
    print(json.dumps(result, indent=2))

    # Test 3: Get ledger
    print("\nTest 3: Get operations ledger")
    result = server.get_ledger()
    print(json.dumps(result, indent=2, default=str))
