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
import subprocess
import tempfile
import os
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Iterator

# Windows-compatible file locking

# Windows-compatible file locking
if sys.platform == 'win32':
    import msvcrt
    def _lock_file(f, exclusive: bool = True):
        """Lock file on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)
    def _unlock_file(f):
        """Unlock file on Windows."""
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
else:
    import fcntl
    def _lock_file(f, exclusive: bool = True):
        """Lock file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    def _unlock_file(f):
        """Unlock file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# This would be replaced with actual MCP SDK
# For now, we structure it as a mock MCP server

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# SPECTRUM-02 Validator version (for OUTPUT_HASHES.json binding)
VALIDATOR_SEMVER = "1.0.0"
SUPPORTED_VALIDATOR_SEMVERS = {"1.0.0", "1.0.1", "1.1.0"}

# Cache for build ID (computed once per process)
_VALIDATOR_BUILD_ID_CACHE: Optional[str] = None


def get_validator_build_id() -> str:
    """Get deterministic validator build fingerprint.

    Preferred: git commit SHA (short) if repo is a git checkout.
    Fallback: SHA-256 of MCP/server.py file bytes.

    Returns a non-empty string. Result is cached for process lifetime.
    """
    global _VALIDATOR_BUILD_ID_CACHE

    if _VALIDATOR_BUILD_ID_CACHE is not None:
        return _VALIDATOR_BUILD_ID_CACHE

    # Try git commit SHA first
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            _VALIDATOR_BUILD_ID_CACHE = f"git:{result.stdout.strip()}"
            return _VALIDATOR_BUILD_ID_CACHE
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: hash of server.py
    server_path = Path(__file__)
    if server_path.exists():
        sha = hashlib.sha256()
        with open(server_path, "rb") as f:
            sha.update(f.read())
        _VALIDATOR_BUILD_ID_CACHE = f"file:{sha.hexdigest()[:12]}"
        return _VALIDATOR_BUILD_ID_CACHE

    # Ultimate fallback (should never happen)
    _VALIDATOR_BUILD_ID_CACHE = "unknown"
    return _VALIDATOR_BUILD_ID_CACHE


CONTRACTS_DIR = PROJECT_ROOT / "CONTRACTS" / "_runs"
SKILLS_DIR = PROJECT_ROOT / "CATALYTIC-DPT" / "SKILLS"

# =============================================================================
# CMP-01 ROOT RULES - Strict path governance
# =============================================================================

# Durable output roots (only places files may persist after run)
DURABLE_ROOTS = [
    "CONTRACTS/_runs/",
    "CORTEX/_generated/",
    "MEMORY/LLM_PACKER/_packs/",
]

# Catalytic domains (temporary, must be restored byte-identical)
CATALYTIC_ROOTS = [
    "CONTRACTS/_runs/_tmp/",
    "CORTEX/_generated/_tmp/",
    "MEMORY/LLM_PACKER/_packs/_tmp/",
    "TOOLS/_tmp/",
    "MCP/_tmp/",
]

# Forbidden roots (must never be written to or overlapped)
FORBIDDEN_ROOTS = [
    "CANON/",
    "AGENTS.md",
]

# Task state machine transitions (valid state changes)
TASK_STATES = {
    "pending": ["acknowledged", "cancelled"],
    "acknowledged": ["processing", "cancelled"],
    "processing": ["completed", "failed", "timeout", "cancelled"],
    "completed": [],  # Terminal state
    "failed": [],     # Terminal state
    "timeout": [],    # Terminal state
    "cancelled": [],  # Terminal state
}

# Configuration constants
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit for file reads
MAX_RESULTS_PER_PAGE = 100
DEFAULT_POLL_INTERVAL = 5
MAX_POLL_INTERVAL = 60
BACKOFF_MULTIPLIER = 1.5


def _atomic_write_jsonl(file_path: Path, line: str) -> bool:
    """Atomically append a single line to a JSONL file.

    Uses write-to-temp-then-append pattern with file locking to prevent:
    1. Partial writes from crashes
    2. Interleaved writes from concurrent processes

    Returns True on success, False on failure.
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first (atomic on most filesystems)
        fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='jsonl_',
            dir=file_path.parent
        )
        try:
            os.write(fd, (line.rstrip('\n') + '\n').encode('utf-8'))
            os.fsync(fd)  # Force flush to disk
        finally:
            os.close(fd)

        # Now append temp content to target with locking
        with open(file_path, 'a', encoding='utf-8') as f:
            _lock_file(f, exclusive=True)
            try:
                with open(temp_path, 'r', encoding='utf-8') as tmp:
                    f.write(tmp.read())
                f.flush()
                os.fsync(f.fileno())
            finally:
                _unlock_file(f)

        # Clean up temp file
        os.unlink(temp_path)
        return True

    except Exception as e:
        # Clean up temp file on error
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except Exception:
            pass
        return False


def _atomic_rewrite_jsonl(
    file_path: Path,
    transform: Callable[[List[Dict]], List[Dict]]
) -> bool:
    """Atomically rewrite a JSONL file with a transformation function.

    Uses read-transform-write pattern:
    1. Read all lines (with lock on read)
    2. Apply transformation
    3. Write to temp file
    4. Atomic rename over original

    Returns True on success, False on failure.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create/touch file if it doesn't exist
        if not file_path.exists():
            file_path.touch()

        # Read phase - use lock to read
        entries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            _lock_file(f, exclusive=False)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
            finally:
                _unlock_file(f)

        # Apply transformation
        transformed = transform(entries)

        # Write to temp file
        fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='jsonl_',
            dir=file_path.parent
        )
        try:
            for entry in transformed:
                os.write(fd, (json.dumps(entry) + '\n').encode('utf-8'))
            os.fsync(fd)
        finally:
            os.close(fd)

        # Atomic rename (works on Unix, best-effort on Windows)
        temp_path_obj = Path(temp_path)
        if sys.platform == 'win32':
            # Windows needs file to be closed before rename
            # Use replace for atomic overwrite (Python 3.3+)
            import shutil
            backup_path = file_path.with_suffix('.bak')
            try:
                # Create backup
                if file_path.exists():
                    shutil.copy2(file_path, backup_path)
                # Replace original with temp
                shutil.move(str(temp_path_obj), str(file_path))
                # Remove backup
                if backup_path.exists():
                    backup_path.unlink()
            except Exception:
                # Restore from backup on failure
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    backup_path.unlink()
                raise
        else:
            os.replace(temp_path, file_path)

        return True

    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception:
            pass
        return False


def _read_jsonl_streaming(
    file_path: Path,
    filter_fn: Optional[Callable[[Dict], bool]] = None,
    limit: int = MAX_RESULTS_PER_PAGE,
    offset: int = 0
) -> Iterator[Dict]:
    """Stream JSONL file with optional filtering and pagination.

    Yields entries one at a time without loading entire file into memory.
    """
    if not file_path.exists():
        return

    count = 0
    skipped = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        _lock_file(f, exclusive=False)  # Shared lock for reading
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

                # Apply filter
                if filter_fn and not filter_fn(entry):
                    continue

                # Apply offset
                if skipped < offset:
                    skipped += 1
                    continue

                # Apply limit
                if count >= limit:
                    break

                yield entry
                count += 1
        finally:
            _unlock_file(f)


def _validate_task_state_transition(current: str, target: str) -> bool:
    """Validate that a task state transition is allowed."""
    if current not in TASK_STATES:
        return False
    return target in TASK_STATES.get(current, [])


def _validate_task_spec(task_spec: Dict) -> Dict:
    """Validate task_spec has required fields and valid structure.

    Returns: {"valid": bool, "errors": [...]}
    """
    errors = []

    # Required fields
    required_fields = ["task_id", "task_type"]
    for field in required_fields:
        if field not in task_spec:
            errors.append(f"Missing required field: {field}")

    # Validate task_type
    valid_task_types = ["file_operation", "code_adapt", "validate", "research"]
    if "task_type" in task_spec and task_spec["task_type"] not in valid_task_types:
        errors.append(f"Invalid task_type: {task_spec['task_type']}. Must be one of: {valid_task_types}")

    # Validate task_id format (alphanumeric with hyphens/underscores)
    if "task_id" in task_spec:
        task_id = task_spec["task_id"]
        if not isinstance(task_id, str) or not re.match(r'^[\w\-]+$', task_id):
            errors.append(f"Invalid task_id format: must be alphanumeric with hyphens/underscores")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


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

        # 3.5 CMP-01 Path validation (pre-execution)
        path_validation = self._validate_jobspec_paths(task_spec)
        if not path_validation["valid"]:
            return {
                "status": "error",
                "message": "CMP-01 path validation failed",
                "errors": path_validation["errors"]
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
        task_spec_bytes = json.dumps(task_spec, indent=2).encode('utf-8')
        with open(run_dir / "TASK_SPEC.json", "wb") as f:
            f.write(task_spec_bytes)

        # 7. Compute and save TASK_SPEC integrity hash (anti-tamper)
        task_spec_hash = hashlib.sha256(task_spec_bytes).hexdigest()
        with open(run_dir / "TASK_SPEC.sha256", "w") as f:
            f.write(task_spec_hash)

        # Log hash in ledger for audit trail
        self._log_operation({
            "operation": "task_spec_hash",
            "run_id": run_id,
            "hash": task_spec_hash
        })

        return {
            "status": "pending",
            "run_id": run_id,
            "skill": skill_name,
            "executor": executor,
            "ledger_dir": str(run_dir),
            "task_spec_hash": task_spec_hash,
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

    def _generate_output_hashes(self, run_id: str) -> Dict:
        """Generate OUTPUT_HASHES.json for SPECTRUM-02 bundle.

        Hashes every declared durable output in TASK_SPEC.json.
        - If output is a file: hash that file
        - If output is a directory: hash every file under it

        Returns: {"valid": bool, "errors": [...], "hashes": {...}}
        """
        errors = []
        hashes = {}
        run_dir = CONTRACTS_DIR / run_id

        # Load TASK_SPEC.json
        task_spec_path = run_dir / "TASK_SPEC.json"
        if not task_spec_path.exists():
            return {
                "valid": False,
                "errors": [{
                    "code": "TASK_SPEC_MISSING",
                    "message": "TASK_SPEC.json not found",
                    "path": "/",
                    "details": {"run_id": run_id}
                }],
                "hashes": {}
            }

        with open(task_spec_path) as f:
            task_spec = json.load(f)

        outputs_spec = task_spec.get("outputs", {})
        durable_paths = outputs_spec.get("durable_paths", [])

        for idx, raw_path in enumerate(durable_paths):
            json_pointer = f"/outputs/durable_paths/{idx}"

            # Skip invalid paths (already caught by _verify_post_run_outputs)
            if Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
                continue

            abs_path = (PROJECT_ROOT / raw_path).resolve()

            # Skip if path escapes repo root
            if not self._is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
                continue

            if not abs_path.exists():
                errors.append({
                    "code": "OUTPUT_MISSING",
                    "message": f"Declared output does not exist: {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path}
                })
                continue

            if abs_path.is_file():
                # Hash single file
                file_hash = self._compute_hash(abs_path)
                # Use posix-style path relative to PROJECT_ROOT
                rel_posix = abs_path.relative_to(PROJECT_ROOT).as_posix()
                hashes[rel_posix] = f"sha256:{file_hash}"
            elif abs_path.is_dir():
                # Hash every file under directory
                for file_path in abs_path.rglob("*"):
                    if file_path.is_file():
                        file_hash = self._compute_hash(file_path)
                        rel_posix = file_path.relative_to(PROJECT_ROOT).as_posix()
                        hashes[rel_posix] = f"sha256:{file_hash}"

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "hashes": hashes
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

        completed_at = datetime.now().isoformat()

        # Helper to write STATUS.json
        def write_status(run_status: str, skill_status: str, cmp01: str, error_msg: Optional[str] = None):
            status_data = {
                "run_id": run_id,
                "status": run_status,
                "skill_status": skill_status,
                "completed_at": completed_at,
                "cmp01": cmp01
            }
            if error_msg:
                status_data["error"] = error_msg
            with open(run_dir / "STATUS.json", "w") as f:
                json.dump(status_data, f, indent=2)

        # CMP-01 Check 1: Verify TASK_SPEC integrity (anti-tamper)
        task_spec_path = run_dir / "TASK_SPEC.json"
        hash_path = run_dir / "TASK_SPEC.sha256"
        
        if hash_path.exists() and task_spec_path.exists():
            with open(hash_path) as f:
                expected_hash = f.read().strip()
            with open(task_spec_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            
            if expected_hash != actual_hash:
                tamper_errors = [{
                    "code": "TASK_SPEC_TAMPERED",
                    "message": "TASK_SPEC.json has been modified after execution started",
                    "path": "/",
                    "details": {"expected_hash": expected_hash, "actual_hash": actual_hash}
                }]
                with open(run_dir / "ERRORS.json", "w") as f:
                    json.dump({
                        "errors": ["TASK_SPEC_TAMPERED: TASK_SPEC.json integrity check failed"],
                        "structured_errors": tamper_errors
                    }, f, indent=2)
                write_status("error", "failed", "fail", "TASK_SPEC.json integrity check failed")
                return {
                    "status": "error",
                    "message": "TASK_SPEC.json integrity check failed",
                    "errors": tamper_errors
                }

        # CMP-01 Check 2: Verify declared outputs exist and are in durable roots
        output_verification = self._verify_post_run_outputs(run_id)
        if not output_verification["valid"]:
            # Persist errors and fail the run
            all_errors = (errors or []) + [
                f"{e['code']}: {e['message']}" for e in output_verification["errors"]
            ]
            with open(run_dir / "ERRORS.json", "w") as f:
                json.dump({
                    "errors": all_errors,
                    "structured_errors": output_verification["errors"]
                }, f, indent=2)
            write_status("error", "failed", "fail", "CMP-01 output verification failed")
            return {
                "status": "error",
                "message": "CMP-01 output verification failed",
                "errors": output_verification["errors"]
            }

        # Save outputs
        with open(run_dir / "OUTPUTS.json", "w") as f:
            json.dump(outputs, f, indent=2)

        # Save errors if any
        if errors:
            with open(run_dir / "ERRORS.json", "w") as f:
                json.dump({"errors": errors}, f, indent=2)

        # SPECTRUM-02: Generate OUTPUT_HASHES.json for durable bundle
        hash_result = self._generate_output_hashes(run_id)
        if not hash_result["valid"]:
            # Hash generation failed - fail closed
            all_errors = (errors or []) + [
                f"{e['code']}: {e['message']}" for e in hash_result["errors"]
            ]
            with open(run_dir / "ERRORS.json", "w") as f:
                json.dump({
                    "errors": all_errors,
                    "structured_errors": hash_result["errors"]
                }, f, indent=2)
            write_status("error", "failed", "fail", "SPECTRUM-02 hash generation failed")
            return {
                "status": "error",
                "message": "SPECTRUM-02 hash generation failed",
                "errors": hash_result["errors"]
            }

        # Write OUTPUT_HASHES.json
        output_hashes_data = {
            "validator_semver": VALIDATOR_SEMVER,
            "validator_build_id": get_validator_build_id(),
            "generated_at": datetime.now().isoformat(),
            "hashes": hash_result["hashes"]
        }
        with open(run_dir / "OUTPUT_HASHES.json", "w") as f:
            json.dump(output_hashes_data, f, indent=2)

        # Write success STATUS.json
        write_status("success", status, "pass")

        # Log completion
        self._log_operation({
            "operation": "skill_complete",
            "run_id": run_id,
            "status": status,
            "outputs": list(outputs.keys()),
            "error_count": len(errors) if errors else 0,
            "cmp01": "pass"
        })

        return {
            "status": "success",
            "run_id": run_id,
            "skill_status": status,
            "ledger_dir": str(run_dir),
            "outputs_saved": len(outputs),
            "errors_logged": len(errors) if errors else 0,
            "cmp01": "pass"
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
        """Validate JSON instance against schema.
        
        Basic validation without jsonschema library:
        - Checks required fields
        - Validates enum values for task_type
        - Returns errors list if invalid
        """
        errors = []
        
        # Check input_schema if present
        input_schema = schema.get("input_schema", schema)
        
        # Check required fields
        required_fields = input_schema.get("required", [])
        for field in required_fields:
            if field not in instance:
                errors.append(f"Missing required field: {field}")
        
        # Validate properties with enums
        properties = input_schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in instance and "enum" in field_schema:
                if instance[field] not in field_schema["enum"]:
                    errors.append(
                        f"Invalid value for '{field}': '{instance[field]}'. "
                        f"Must be one of: {field_schema['enum']}"
                    )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    # =========================================================================
    # CMP-01 PATH VALIDATION
    # Forbidden overlap containment + output existence checks
    # =========================================================================

    def _is_path_under_root(self, path: Path, root: Path) -> bool:
        """Component-safe check if path is under root (not just string prefix)."""
        try:
            # Python 3.9+ has is_relative_to
            return path.is_relative_to(root)
        except AttributeError:
            # Python 3.8 fallback
            try:
                path.relative_to(root)
                return True
            except ValueError:
                return False

    def _validate_single_path(
        self,
        raw_path: str,
        json_pointer: str,
        allowed_roots: List[str],
        root_error_code: str
    ) -> List[Dict]:
        """Validate a single path against CMP-01 rules.
        
        Returns list of error dicts (empty if valid).
        """
        errors = []

        # 1. Reject absolute paths
        if Path(raw_path).is_absolute():
            errors.append({
                "code": "PATH_ESCAPES_REPO_ROOT",
                "message": f"Absolute paths are not allowed: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "reason": "absolute_path"}
            })
            return errors

        # 2. Reject traversal segments
        path_parts = Path(raw_path).parts
        if ".." in path_parts:
            errors.append({
                "code": "PATH_CONTAINS_TRAVERSAL",
                "message": f"Path contains forbidden traversal segment '..': {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "segments": list(path_parts)}
            })
            return errors

        # 3. Resolve and check containment under PROJECT_ROOT
        abs_path = (PROJECT_ROOT / raw_path).resolve()
        if not self._is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
            errors.append({
                "code": "PATH_ESCAPES_REPO_ROOT",
                "message": f"Path escapes repository root: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "resolved": str(abs_path), "repo_root": str(PROJECT_ROOT)}
            })
            return errors

        # 4. Check forbidden overlap
        for forbidden in FORBIDDEN_ROOTS:
            forbidden_abs = (PROJECT_ROOT / forbidden).resolve()
            # Check both directions of containment
            if self._is_path_under_root(abs_path, forbidden_abs) or self._is_path_under_root(forbidden_abs, abs_path):
                errors.append({
                    "code": "FORBIDDEN_PATH_OVERLAP",
                    "message": f"Path overlaps forbidden root '{forbidden}': {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "forbidden_root": forbidden}
                })
                return errors

        # 5. Check under allowed roots
        under_allowed = False
        for root in allowed_roots:
            root_abs = (PROJECT_ROOT / root).resolve()
            if self._is_path_under_root(abs_path, root_abs):
                under_allowed = True
                break

        if not under_allowed:
            errors.append({
                "code": root_error_code,
                "message": f"Path not under any allowed root: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "allowed_roots": allowed_roots}
            })

        return errors

    def _check_containment_overlap(
        self,
        paths: List[str],
        json_pointer_base: str
    ) -> List[Dict]:
        """Check for containment overlap between paths in the same list.
        
        Policy: Exact duplicates (same resolved path) are allowed/deduped.
        Only flag when one path strictly contains another.
        """
        errors = []
        abs_paths = []

        # Store (original_index, raw_path, abs_path) to preserve correct indices
        for orig_idx, raw_path in enumerate(paths):
            if not Path(raw_path).is_absolute() and ".." not in Path(raw_path).parts:
                abs_paths.append((orig_idx, raw_path, (PROJECT_ROOT / raw_path).resolve()))

        for i, (idx_a, raw_a, abs_a) in enumerate(abs_paths):
            for j, (idx_b, raw_b, abs_b) in enumerate(abs_paths):
                if i >= j:
                    continue
                
                # Allow exact duplicates (same resolved path)
                if abs_a == abs_b:
                    continue
                
                # Check if one strictly contains the other (both directions)
                if self._is_path_under_root(abs_a, abs_b) or self._is_path_under_root(abs_b, abs_a):
                    # Use smaller original index for deterministic path pointer
                    smaller_idx = min(idx_a, idx_b)
                    errors.append({
                        "code": "PATH_OVERLAP",
                        "message": f"Paths have containment overlap: '{raw_a}' and '{raw_b}'",
                        "path": f"{json_pointer_base}/{smaller_idx}",
                        "details": {
                            "index_a": idx_a,
                            "index_b": idx_b,
                            "path_a": raw_a,
                            "path_b": raw_b
                        }
                    })

        return errors



    def _validate_jobspec_paths(self, task_spec: Dict) -> Dict:
        """Validate all paths in a JobSpec against CMP-01 rules.
        
        Checks:
        - catalytic_domains must be under CATALYTIC_ROOTS
        - outputs.durable_paths must be under DURABLE_ROOTS
        - No forbidden overlaps
        - No traversal escapes
        - No containment overlap within same list
        
        Returns: {"valid": bool, "errors": [error_dict, ...]}
        """
        errors = []

        # Validate catalytic_domains
        catalytic_domains = task_spec.get("catalytic_domains", [])
        for idx, domain in enumerate(catalytic_domains):
            path_errors = self._validate_single_path(
                domain,
                f"/catalytic_domains/{idx}",
                CATALYTIC_ROOTS,
                "CATALYTIC_OUTSIDE_ROOT"
            )
            errors.extend(path_errors)

        # Check containment overlap within catalytic_domains
        if len(catalytic_domains) > 1:
            errors.extend(self._check_containment_overlap(
                catalytic_domains,
                "/catalytic_domains"
            ))

        # Validate outputs.durable_paths
        outputs = task_spec.get("outputs", {})
        durable_paths = outputs.get("durable_paths", [])
        for idx, dpath in enumerate(durable_paths):
            path_errors = self._validate_single_path(
                dpath,
                f"/outputs/durable_paths/{idx}",
                DURABLE_ROOTS,
                "OUTPUT_OUTSIDE_DURABLE_ROOT"
            )
            errors.extend(path_errors)

        # Check containment overlap within durable_paths
        if len(durable_paths) > 1:
            errors.extend(self._check_containment_overlap(
                durable_paths,
                "/outputs/durable_paths"
            ))

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _verify_post_run_outputs(self, run_id: str) -> Dict:
        """Verify declared outputs exist after run completion.
        
        Called from skill_complete to enforce output existence.
        
        Returns: {"valid": bool, "errors": [error_dict, ...]}
        """
        errors = []
        run_dir = CONTRACTS_DIR / run_id

        # Load TASK_SPEC.json
        task_spec_path = run_dir / "TASK_SPEC.json"
        if not task_spec_path.exists():
            return {
                "valid": False,
                "errors": [{
                    "code": "TASK_SPEC_MISSING",
                    "message": f"TASK_SPEC.json not found in run directory",
                    "path": "/",
                    "details": {"run_id": run_id, "expected": str(task_spec_path)}
                }]
            }

        with open(task_spec_path) as f:
            task_spec = json.load(f)

        outputs = task_spec.get("outputs", {})
        durable_paths = outputs.get("durable_paths", [])

        for idx, raw_path in enumerate(durable_paths):
            json_pointer = f"/outputs/durable_paths/{idx}"

            # 0. Report errors for absolute/traversal paths (do not silently skip)
            if Path(raw_path).is_absolute():
                errors.append({
                    "code": "PATH_ESCAPES_REPO_ROOT",
                    "message": f"Absolute paths are not allowed: {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "reason": "absolute_path"}
                })
                continue  # Skip further checks for this entry

            if ".." in Path(raw_path).parts:
                errors.append({
                    "code": "PATH_CONTAINS_TRAVERSAL",
                    "message": f"Path contains forbidden traversal segment '..': {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "segments": list(Path(raw_path).parts)}
                })
                continue  # Skip further checks for this entry

            abs_path = (PROJECT_ROOT / raw_path).resolve()

            # 0.5 Check containment under PROJECT_ROOT (catches symlink escapes)
            if not self._is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
                errors.append({
                    "code": "PATH_ESCAPES_REPO_ROOT",
                    "message": f"Path escapes repository root (possibly via symlink): {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "resolved": str(abs_path), "repo_root": str(PROJECT_ROOT)}
                })
                continue  # Skip further checks for this entry

            # 1. Check forbidden overlap (hard stop for this entry if found)

            forbidden_hit = False
            for forbidden in FORBIDDEN_ROOTS:
                forbidden_abs = (PROJECT_ROOT / forbidden).resolve()
                if self._is_path_under_root(abs_path, forbidden_abs) or self._is_path_under_root(forbidden_abs, abs_path):
                    errors.append({
                        "code": "FORBIDDEN_PATH_OVERLAP",
                        "message": f"Output overlaps forbidden root '{forbidden}': {raw_path}",
                        "path": json_pointer,
                        "details": {"declared": raw_path, "forbidden_root": forbidden}
                    })
                    forbidden_hit = True
                    break  # Exit forbidden loop
            
            if forbidden_hit:
                continue  # Skip durable/existence checks for this entry

            # 2. Check under DURABLE_ROOTS
            under_durable = False
            for root in DURABLE_ROOTS:
                root_abs = (PROJECT_ROOT / root).resolve()
                if self._is_path_under_root(abs_path, root_abs):
                    under_durable = True
                    break

            if not under_durable:
                errors.append({
                    "code": "OUTPUT_OUTSIDE_DURABLE_ROOT",
                    "message": f"Output not under any durable root: {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "durable_roots": DURABLE_ROOTS}
                })
                continue  # Skip existence check for this entry

            # 3. Check existence on disk
            if not abs_path.exists():
                errors.append({
                    "code": "OUTPUT_MISSING",
                    "message": f"Declared output does not exist: {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "resolved": str(abs_path)}
                })

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    # =========================================================================
    # SPECTRUM-02 BUNDLE VERIFICATION
    # Adversarial resume without execution history
    # =========================================================================

    def verify_spectrum02_bundle(
        self,
        run_dir: Path,
        strict_build_id: bool = False
    ) -> Dict:
        """Verify a SPECTRUM-02 resume bundle.

        Checks:
        - TASK_SPEC.json exists
        - STATUS.json exists with status=success and cmp01=pass
        - OUTPUT_HASHES.json exists with supported validator_semver
        - validator_build_id exists and is non-empty
        - All declared hashes verify against actual files

        Args:
            run_dir: Path to the run directory containing the bundle
            strict_build_id: If True, reject if build_id != current get_validator_build_id()

        Returns: {"valid": bool, "errors": [...]}
        """
        errors = []

        # Ensure run_dir is a Path
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        # 1. Check TASK_SPEC.json exists
        task_spec_path = run_dir / "TASK_SPEC.json"
        if not task_spec_path.exists():
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "message": "TASK_SPEC.json missing",
                "path": "/",
                "details": {"expected": str(task_spec_path)}
            })
            return {"valid": False, "errors": errors}

        # 2. Check STATUS.json exists and is valid
        status_path = run_dir / "STATUS.json"
        if not status_path.exists():
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "message": "STATUS.json missing",
                "path": "/",
                "details": {"expected": str(status_path)}
            })
            return {"valid": False, "errors": errors}

        try:
            with open(status_path) as f:
                status = json.load(f)
        except json.JSONDecodeError as e:
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "message": f"STATUS.json invalid JSON: {e}",
                "path": "/",
                "details": {}
            })
            return {"valid": False, "errors": errors}

        if status.get("status") != "success":
            errors.append({
                "code": "STATUS_NOT_SUCCESS",
                "message": f"STATUS.status is '{status.get('status')}', expected 'success'",
                "path": "/status",
                "details": {"actual": status.get("status")}
            })
            return {"valid": False, "errors": errors}

        if status.get("cmp01") != "pass":
            errors.append({
                "code": "CMP01_NOT_PASS",
                "message": f"STATUS.cmp01 is '{status.get('cmp01')}', expected 'pass'",
                "path": "/cmp01",
                "details": {"actual": status.get("cmp01")}
            })
            return {"valid": False, "errors": errors}

        # 3. Check OUTPUT_HASHES.json exists and is valid
        hashes_path = run_dir / "OUTPUT_HASHES.json"
        if not hashes_path.exists():
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "message": "OUTPUT_HASHES.json missing",
                "path": "/",
                "details": {"expected": str(hashes_path)}
            })
            return {"valid": False, "errors": errors}

        try:
            with open(hashes_path) as f:
                output_hashes = json.load(f)
        except json.JSONDecodeError as e:
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "message": f"OUTPUT_HASHES.json invalid JSON: {e}",
                "path": "/",
                "details": {}
            })
            return {"valid": False, "errors": errors}

        # 4. Check validator semver is supported
        validator_semver = output_hashes.get("validator_semver")
        if validator_semver not in SUPPORTED_VALIDATOR_SEMVERS:
            errors.append({
                "code": "VALIDATOR_UNSUPPORTED",
                "message": f"validator_semver '{validator_semver}' not supported",
                "path": "/validator_semver",
                "details": {
                    "actual": validator_semver,
                    "supported": list(SUPPORTED_VALIDATOR_SEMVERS)
                }
            })
            return {"valid": False, "errors": errors}

        # 5. Check validator_build_id exists and is non-empty
        validator_build_id = output_hashes.get("validator_build_id")
        if not validator_build_id:
            errors.append({
                "code": "VALIDATOR_BUILD_ID_MISSING",
                "message": "validator_build_id is missing or empty",
                "path": "/validator_build_id",
                "details": {"actual": validator_build_id}
            })
            return {"valid": False, "errors": errors}

        # 6. Strict build ID check (optional)
        if strict_build_id:
            current_build_id = get_validator_build_id()
            if validator_build_id != current_build_id:
                errors.append({
                    "code": "VALIDATOR_BUILD_MISMATCH",
                    "message": f"validator_build_id mismatch: expected '{current_build_id}', got '{validator_build_id}'",
                    "path": "/validator_build_id",
                    "details": {
                        "expected": current_build_id,
                        "actual": validator_build_id
                    }
                })
                return {"valid": False, "errors": errors}

        # 7. Verify each hash
        hashes = output_hashes.get("hashes", {})
        for rel_path, expected_hash in hashes.items():
            abs_path = PROJECT_ROOT / rel_path

            if not abs_path.exists():
                errors.append({
                    "code": "OUTPUT_MISSING",
                    "message": f"Output file does not exist: {rel_path}",
                    "path": f"/hashes/{rel_path}",
                    "details": {"declared": rel_path, "resolved": str(abs_path)}
                })
                continue

            # Compute actual hash
            actual_hash = f"sha256:{self._compute_hash(abs_path)}"

            if actual_hash != expected_hash:
                errors.append({
                    "code": "HASH_MISMATCH",
                    "message": f"Hash mismatch for {rel_path}",
                    "path": f"/hashes/{rel_path}",
                    "details": {
                        "declared": rel_path,
                        "expected": expected_hash,
                        "actual": actual_hash
                    }
                })

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


    # =========================================================================
    # AGENT MESSAGING SYSTEM
    # Governor ↔ Ant Worker communication via MCP
    # =========================================================================


    def dispatch_task(
        self,
        task_id: str,
        task_spec: Dict,
        from_agent: str,
        to_agent: str,
        priority: int = 5
    ) -> Dict:
        """Governor dispatches task to Ant Worker via MCP message queue.

        Features:
        - Task spec validation before dispatch
        - Duplicate task detection
        - Atomic file writes
        - Priority validation (1-10)
        """
        # Validate task_spec
        validation = _validate_task_spec(task_spec)
        if not validation["valid"]:
            return {
                "status": "error",
                "message": "Task spec validation failed",
                "errors": validation["errors"]
            }

        # Validate priority
        priority = max(1, min(10, priority))

        # Check for duplicate task_id (already pending or processing)
        queue_file = self.ledger_path / "task_queue.jsonl"
        for entry in _read_jsonl_streaming(
            queue_file,
            filter_fn=lambda e: e.get("task_id") == task_id and e.get("status") in ["pending", "acknowledged", "processing"]
        ):
            return {
                "status": "error",
                "message": f"Duplicate task_id: {task_id} already exists with status '{entry.get('status')}'",
                "existing_task": entry
            }

        task_message = {
            "task_id": task_id,
            "task_spec": task_spec,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "priority": priority,
            "status": "pending",
            "dispatched_at": datetime.now().isoformat(),
            "acknowledged_at": None,
            "completed_at": None,
            "result": None
        }

        # Atomic write to queue
        if not _atomic_write_jsonl(queue_file, json.dumps(task_message)):
            return {
                "status": "error",
                "message": "Failed to write task to queue (atomic write failed)"
            }

        self._log_operation({
            "operation": "dispatch_task",
            "task_id": task_id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "priority": priority
        })

        return {"status": "dispatched", "task_id": task_id, "to_agent": to_agent, "priority": priority}

    def get_pending_tasks(self, agent_id: str, limit: int = 10) -> Dict:
        """Ant Worker checks for pending tasks assigned to it.

        Features:
        - Streaming read (memory efficient)
        - Priority sorting
        - Configurable limit
        - Robust error handling
        """
        queue_file = self.ledger_path / "task_queue.jsonl"
        pending = []

        try:
            for task in _read_jsonl_streaming(
                queue_file,
                filter_fn=lambda t: t.get("to_agent") == agent_id and t.get("status") == "pending",
                limit=limit * 2  # Get extra for sorting
            ):
                pending.append(task)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read task queue: {str(e)}",
                "agent_id": agent_id,
                "pending_count": 0,
                "tasks": []
            }

        # Sort by priority (highest first), then by dispatch time (oldest first)
        pending.sort(key=lambda x: (-x.get("priority", 5), x.get("dispatched_at", "")))

        # Apply limit after sorting
        pending = pending[:limit]

        return {
            "status": "success",
            "agent_id": agent_id,
            "pending_count": len(pending),
            "tasks": pending
        }

    def report_result(self, task_id: str, from_agent: str, status: str, result: Dict, errors: Optional[List[str]] = None) -> Dict:
        """Ant Worker reports task result back to Governor.

        Features:
        - Atomic write
        - Duplicate result detection
        - Status validation
        """
        # Validate status
        valid_statuses = ["success", "failed", "error", "timeout"]
        if status not in valid_statuses:
            return {
                "status": "error",
                "message": f"Invalid status: {status}. Must be one of: {valid_statuses}"
            }

        # Check for duplicate result
        results_file = self.ledger_path / "task_results.jsonl"
        for existing in _read_jsonl_streaming(
            results_file,
            filter_fn=lambda r: r.get("task_id") == task_id
        ):
            return {
                "status": "error",
                "message": f"Result already reported for task_id: {task_id}",
                "existing_result": existing
            }

        result_message = {
            "task_id": task_id,
            "from_agent": from_agent,
            "status": status,
            "result": result,
            "errors": errors or [],
            "reported_at": datetime.now().isoformat()
        }

        # Atomic write
        if not _atomic_write_jsonl(results_file, json.dumps(result_message)):
            return {
                "status": "error",
                "message": "Failed to write result (atomic write failed)"
            }

        # Also update task queue to mark as completed
        queue_file = self.ledger_path / "task_queue.jsonl"

        def update_task_status(entries):
            for entry in entries:
                if entry.get("task_id") == task_id:
                    entry["status"] = "completed" if status == "success" else "failed"
                    entry["completed_at"] = datetime.now().isoformat()
            return entries

        _atomic_rewrite_jsonl(queue_file, update_task_status)

        self._log_operation({
            "operation": "report_result",
            "task_id": task_id,
            "from_agent": from_agent,
            "status": status,
            "error_count": len(errors) if errors else 0
        })

        return {"status": "reported", "task_id": task_id}

    def get_results(
        self,
        task_id: Optional[str] = None,
        limit: int = MAX_RESULTS_PER_PAGE,
        offset: int = 0
    ) -> Dict:
        """Governor retrieves results from Ant Workers.

        Features:
        - Streaming read with pagination (memory efficient)
        - Optional task_id filter
        - Configurable limit and offset
        """
        results_file = self.ledger_path / "task_results.jsonl"
        results = []

        try:
            filter_fn = None
            if task_id:
                filter_fn = lambda r: r.get("task_id") == task_id

            for result in _read_jsonl_streaming(
                results_file,
                filter_fn=filter_fn,
                limit=limit,
                offset=offset
            ):
                results.append(result)

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read results: {str(e)}",
                "results": [],
                "count": 0
            }

        return {
            "status": "success",
            "results": results,
            "count": len(results),
            "limit": limit,
            "offset": offset,
            "has_more": len(results) == limit
        }

    def acknowledge_task(self, task_id: str, agent_id: str = None) -> Dict:
        """Mark task as acknowledged so it transitions to processing.

        Features:
        - Atomic rewrite (no race conditions)
        - State machine validation
        - Agent ownership verification
        """
        queue_file = self.ledger_path / "task_queue.jsonl"
        acknowledged = False
        error_msg = None
        previous_status = None

        def do_acknowledge(entries):
            nonlocal acknowledged, error_msg, previous_status
            for entry in entries:
                if entry.get("task_id") == task_id:
                    previous_status = entry.get("status")

                    # Verify agent ownership if specified
                    if agent_id and entry.get("to_agent") != agent_id:
                        error_msg = f"Task {task_id} is assigned to {entry.get('to_agent')}, not {agent_id}"
                        return entries

                    # Validate state transition
                    if previous_status != "pending":
                        error_msg = f"Cannot acknowledge task in state '{previous_status}' (must be 'pending')"
                        return entries

                    entry["status"] = "acknowledged"
                    entry["acknowledged_at"] = datetime.now().isoformat()
                    if agent_id:
                        entry["acknowledged_by"] = agent_id
                    acknowledged = True
                    break
            return entries

        if not _atomic_rewrite_jsonl(queue_file, do_acknowledge):
            return {
                "status": "error",
                "message": "Failed to acknowledge task (atomic rewrite failed)",
                "task_id": task_id
            }

        if error_msg:
            return {
                "status": "error",
                "message": error_msg,
                "task_id": task_id,
                "previous_status": previous_status
            }

        if not acknowledged:
            return {
                "status": "not_found",
                "message": f"Task {task_id} not found in queue",
                "task_id": task_id
            }

        self._log_operation({
            "operation": "acknowledge_task",
            "task_id": task_id,
            "agent_id": agent_id
        })

        return {
            "status": "acknowledged",
            "task_id": task_id,
            "previous_status": previous_status
        }

    # =========================================================================
    # CHAIN OF COMMAND ESCALATION
    # Ant → Governor → Claude → User
    # =========================================================================

    CHAIN_OF_COMMAND = ["Ant", "Governor", "Claude", "User"]

    def escalate(
        self,
        from_agent: str,
        issue: str,
        context: Dict,
        priority: int = 5
    ) -> Dict:
        """Escalate issue UP the chain of command.

        Features:
        - Atomic write
        - Priority validation
        - Chain of command enforcement
        """
        # Validate priority
        priority = max(1, min(10, priority))

        # Determine who to escalate to
        try:
            from_index = next(i for i, level in enumerate(self.CHAIN_OF_COMMAND)
                             if from_agent.startswith(level))
            to_level = self.CHAIN_OF_COMMAND[from_index + 1] if from_index + 1 < len(self.CHAIN_OF_COMMAND) else "User"
        except (StopIteration, IndexError):
            to_level = "Governor"  # Default escalation target

        escalation = {
            "escalation_id": f"esc-{uuid.uuid4().hex[:8]}",
            "from_agent": from_agent,
            "to_level": to_level,
            "issue": issue,
            "context": context,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "resolved_at": None,
            "resolution": None
        }

        escalation_file = self.ledger_path / "escalations.jsonl"
        if not _atomic_write_jsonl(escalation_file, json.dumps(escalation)):
            return {
                "status": "error",
                "message": "Failed to write escalation (atomic write failed)"
            }

        self._log_operation({
            "operation": "escalate",
            "escalation_id": escalation["escalation_id"],
            "from_agent": from_agent,
            "to_level": to_level,
            "priority": priority
        })

        return {
            "status": "escalated",
            "escalation_id": escalation["escalation_id"],
            "from": from_agent,
            "to": to_level,
            "message": f"Issue escalated to {to_level}"
        }

    def get_escalations(self, for_level: str, limit: int = 50) -> Dict:
        """Get pending escalations for a level in the chain.

        Features:
        - Streaming read
        - Priority sorting
        """
        escalation_file = self.ledger_path / "escalations.jsonl"
        pending = []

        try:
            for esc in _read_jsonl_streaming(
                escalation_file,
                filter_fn=lambda e: e.get("to_level") == for_level and e.get("status") == "pending",
                limit=limit * 2
            ):
                pending.append(esc)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read escalations: {str(e)}",
                "for_level": for_level,
                "pending_count": 0,
                "escalations": []
            }

        pending.sort(key=lambda x: (-x.get("priority", 5), x.get("created_at", "")))
        pending = pending[:limit]

        return {
            "status": "success",
            "for_level": for_level,
            "pending_count": len(pending),
            "escalations": pending
        }

    def resolve_escalation(
        self,
        escalation_id: str,
        resolved_by: str,
        resolution: str,
        action_taken: Optional[str] = None
    ) -> Dict:
        """Resolve an escalation with decision.

        Features:
        - Atomic rewrite
        - State validation
        """
        escalation_file = self.ledger_path / "escalations.jsonl"
        resolved = False
        error_msg = None

        def do_resolve(entries):
            nonlocal resolved, error_msg
            for entry in entries:
                if entry.get("escalation_id") == escalation_id:
                    if entry.get("status") != "pending":
                        error_msg = f"Escalation already in state '{entry.get('status')}'"
                        return entries

                    entry["status"] = "resolved"
                    entry["resolved_at"] = datetime.now().isoformat()
                    entry["resolved_by"] = resolved_by
                    entry["resolution"] = resolution
                    entry["action_taken"] = action_taken
                    resolved = True
                    break
            return entries

        if not _atomic_rewrite_jsonl(escalation_file, do_resolve):
            return {
                "status": "error",
                "message": "Failed to resolve escalation (atomic rewrite failed)"
            }

        if error_msg:
            return {"status": "error", "message": error_msg}

        if not resolved:
            return {"status": "error", "message": f"Escalation {escalation_id} not found"}

        self._log_operation({
            "operation": "resolve_escalation",
            "escalation_id": escalation_id,
            "resolved_by": resolved_by
        })

        return {
            "status": "resolved",
            "escalation_id": escalation_id,
            "resolved_by": resolved_by
        }

    def send_directive(
        self,
        from_level: str,
        to_agent: str,
        directive: str,
        context: Dict
    ) -> Dict:
        """Send directive DOWN the chain of command.

        Features:
        - Atomic write
        - Chain validation
        """
        directive_msg = {
            "directive_id": f"dir-{uuid.uuid4().hex[:8]}",
            "from_level": from_level,
            "to_agent": to_agent,
            "directive": directive,
            "context": context,
            "status": "pending",
            "issued_at": datetime.now().isoformat(),
            "acknowledged_at": None
        }

        directive_file = self.ledger_path / "directives.jsonl"
        if not _atomic_write_jsonl(directive_file, json.dumps(directive_msg)):
            return {
                "status": "error",
                "message": "Failed to write directive (atomic write failed)"
            }

        self._log_operation({
            "operation": "send_directive",
            "directive_id": directive_msg["directive_id"],
            "from_level": from_level,
            "to_agent": to_agent
        })

        return {
            "status": "issued",
            "directive_id": directive_msg["directive_id"],
            "to": to_agent
        }

    def get_directives(self, for_agent: str, limit: int = 50) -> Dict:
        """Get pending directives for an agent.

        Features:
        - Streaming read
        - Ordered by issue time
        """
        directive_file = self.ledger_path / "directives.jsonl"
        pending = []

        try:
            for d in _read_jsonl_streaming(
                directive_file,
                filter_fn=lambda x: x.get("to_agent") == for_agent and x.get("status") == "pending",
                limit=limit
            ):
                pending.append(d)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read directives: {str(e)}",
                "for_agent": for_agent,
                "pending_count": 0,
                "directives": []
            }

        # Sort by issue time (oldest first - FIFO)
        pending.sort(key=lambda x: x.get("issued_at", ""))

        return {
            "status": "success",
            "for_agent": for_agent,
            "pending_count": len(pending),
            "directives": pending
        }

    def acknowledge_directive(self, directive_id: str, agent_id: str = None) -> Dict:
        """Mark directive as processed.

        Features:
        - Atomic rewrite
        - Agent verification
        """
        directive_file = self.ledger_path / "directives.jsonl"
        acknowledged = False
        error_msg = None

        def do_acknowledge(entries):
            nonlocal acknowledged, error_msg
            for entry in entries:
                if entry.get("directive_id") == directive_id:
                    # Verify agent if specified
                    if agent_id and entry.get("to_agent") != agent_id:
                        error_msg = f"Directive is for {entry.get('to_agent')}, not {agent_id}"
                        return entries

                    if entry.get("status") != "pending":
                        error_msg = f"Directive already in state '{entry.get('status')}'"
                        return entries

                    entry["status"] = "processed"
                    entry["acknowledged_at"] = datetime.now().isoformat()
                    if agent_id:
                        entry["acknowledged_by"] = agent_id
                    acknowledged = True
                    break
            return entries

        if not _atomic_rewrite_jsonl(directive_file, do_acknowledge):
            return {
                "status": "error",
                "message": "Failed to acknowledge directive (atomic rewrite failed)"
            }

        if error_msg:
            return {"status": "error", "message": error_msg, "directive_id": directive_id}

        if not acknowledged:
            return {"status": "not_found", "directive_id": directive_id}

        self._log_operation({
            "operation": "acknowledge_directive",
            "directive_id": directive_id,
            "agent_id": agent_id
        })

        return {"status": "acknowledged", "directive_id": directive_id}


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
        },
        # Agent Messaging Tools (Governor ↔ Ant Worker)
        "dispatch_task": {
            "description": "Governor dispatches task to Ant Worker",
            "parameters": {
                "task_id": "Unique task identifier",
                "task_spec": "Task specification JSON",
                "from_agent": "Dispatching agent (Governor)",
                "to_agent": "Target agent (Ant-1, Ant-2)",
                "priority": "Task priority (1-10)"
            }
        },
        "get_pending_tasks": {
            "description": "Ant Worker polls for pending tasks",
            "parameters": {
                "agent_id": "ID of the polling agent"
            }
        },
        "report_result": {
            "description": "Ant Worker reports task result to Governor",
            "parameters": {
                "task_id": "Task ID being reported",
                "from_agent": "Reporting agent",
                "status": "success|failed|error",
                "result": "Result dictionary",
                "errors": "List of errors if any"
            }
        },
        "get_results": {
            "description": "Governor retrieves results from Ant Workers",
            "parameters": {
                "task_id": "Optional filter by task ID"
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
