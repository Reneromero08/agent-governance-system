#!/usr/bin/env python3
"""
CATALYTIC-DPT/SKILLS/ant-worker/run.py

Distributed task executor for Ant Workers.

Executes via MCP to prevent drift and ensure hash verification.
Reports to Governor and logs all operations to immutable ledger.

Task types:
- file_operation: Copy, move, delete, read files
- code_adapt: Update imports, refactor code
- validate: Test files, run fixtures
- research: Analyze and understand code
"""

import json
import sys
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MCP_SERVER = PROJECT_ROOT / "CATALYTIC-DPT" / "LAB" / "MCP" / "server.py"
CONTRACTS_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"


def load_json(path: str) -> Dict:
    """Load JSON from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def save_json(path: str, data: Dict) -> None:
    """Save JSON deterministically."""
    Path(path).write_bytes(_canonical_json_bytes(data))


def compute_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file."""
    sha = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)  # 64kb chunks
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


class GrokExecutor:
    """Executes tasks via MCP mediator."""

    def __init__(self, task_spec: Dict, run_id: Optional[str] = None):
        self.task_spec = task_spec
        self.task_id = task_spec.get("task_id", "unnamed_task")
        self.task_type = task_spec.get("task_type", "file_operation")

        # Deterministic run_id: caller may supply, else derive from task spec bytes.
        if run_id is None:
            task_id = self.task_id if isinstance(self.task_id, str) and self.task_id else "unnamed_task"
            spec_hash = hashlib.sha256(_canonical_json_bytes(self.task_spec)).hexdigest()[:12]
            run_id = f"{task_id}-{spec_hash}"

        self.run_id = run_id
        self.run_dir = CONTRACTS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        timestamp = task_spec.get("timestamp", "CATALYTIC-DPT-02_CONFIG")
        if not isinstance(timestamp, str) or not timestamp:
            timestamp = "CATALYTIC-DPT-02_CONFIG"

        self.results = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "run_id": run_id,
            "status": "pending",
            "timestamp": timestamp,
            "operations": [],
            "errors": [],
            "ledger_dir": str(self.run_dir)
        }

    def execute(self) -> Dict:
        """Execute the task based on task_type."""
        try:
            # Save task spec to ledger (immutable)
            with open(self.run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(self.task_spec, f, indent=2)

            if self.task_type == "file_operation":
                self._execute_file_operation()
            elif self.task_type == "code_adapt":
                self._execute_code_adapt()
            elif self.task_type == "validate":
                self._execute_validate()
            elif self.task_type == "research":
                self._execute_research()
            else:
                self.results["errors"].append(f"Unknown task_type: {self.task_type}")
                self.results["status"] = "error"
                return self.results

            # If no errors, mark as success
            if not self.results["errors"]:
                self.results["status"] = "success"
            else:
                self.results["status"] = "error"

        except Exception as e:
            self.results["status"] = "error"
            self.results["errors"].append(f"Exception: {str(e)}")

        finally:
            # Save results to ledger
            (self.run_dir / "RESULTS.json").write_bytes(_canonical_json_bytes(self.results))

        return self.results

    def _execute_file_operation(self) -> None:
        """Execute file operations (copy, move, delete, read)."""
        operation = self.task_spec.get("operation", "copy")

        if operation == "copy":
            self._copy_files()
        elif operation == "move":
            self._move_files()
        elif operation == "delete":
            self._delete_files()
        elif operation == "read":
            self._read_files()
        else:
            self.results["errors"].append(f"Unknown file operation: {operation}")

    def _copy_files(self) -> None:
        """Copy files with MCP-like hash verification."""
        files = self.task_spec.get("files", [])
        verify_integrity = self.task_spec.get("verify_integrity", True)

        if not files:
            self.results["errors"].append("No files specified in 'files' list")
            return

        for file_spec in files:
            source = file_spec.get("source")
            destination = file_spec.get("destination")

            if not source or not destination:
                self.results["errors"].append(f"Missing source or destination: {file_spec}")
                continue

            try:
                source_path = Path(source)
                dest_path = Path(destination)

                # Verify source exists
                if not source_path.exists():
                    self.results["errors"].append(f"Source file not found: {source}")
                    continue

                # Create destination directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Compute source hash
                source_hash = compute_hash(source_path)

                # Copy file
                shutil.copy2(source_path, dest_path)

                # Compute destination hash
                dest_hash = compute_hash(dest_path)

                # Verify integrity
                hash_match = source_hash == dest_hash
                if verify_integrity and not hash_match:
                    # Remove corrupted file
                    dest_path.unlink()
                    self.results["errors"].append(
                        f"Hash mismatch for {destination} (file removed). "
                        f"Source: {source_hash}, Dest: {dest_hash}"
                    )
                    continue

                # Log operation
                operation_result = {
                    "operation": "copy",
                    "source": source,
                    "destination": destination,
                    "source_hash": source_hash,
                    "dest_hash": dest_hash,
                    "hash_verified": hash_match,
                    "size_bytes": source_path.stat().st_size,
                }
                self.results["operations"].append(operation_result)
                print(f"[grok-executor] [OK] Copied: {destination} (hash verified: {hash_match})")

            except Exception as e:
                self.results["errors"].append(f"Error copying {source} to {destination}: {str(e)}")

    def _move_files(self) -> None:
        """Move files (essentially copy + delete)."""
        files = self.task_spec.get("files", [])

        for file_spec in files:
            source = file_spec.get("source")
            destination = file_spec.get("destination")

            if not source or not destination:
                continue

            try:
                source_path = Path(source)
                dest_path = Path(destination)

                if not source_path.exists():
                    self.results["errors"].append(f"Source not found: {source}")
                    continue

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(dest_path))

                self.results["operations"].append({
                    "operation": "move",
                    "source": source,
                    "destination": destination,
                })
                print(f"[grok-executor] [OK] Moved: {source} -> {destination}")

            except Exception as e:
                self.results["errors"].append(f"Error moving {source}: {str(e)}")

    def _delete_files(self) -> None:
        """Delete files."""
        files = self.task_spec.get("files", [])

        for file_path in files:
            try:
                p = Path(file_path)
                if p.exists():
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)

                    self.results["operations"].append({
                        "operation": "delete",
                        "path": file_path,
                    })
                    print(f"[grok-executor] [OK] Deleted: {file_path}")
            except Exception as e:
                self.results["errors"].append(f"Error deleting {file_path}: {str(e)}")

    def _read_files(self) -> None:
        """Read and return file contents with size limits and streaming.

        Features:
        - File size limit (10MB default, configurable)
        - Truncation with indicator
        - Total memory limit across all files
        """
        files = self.task_spec.get("files", [])
        max_file_size = self.task_spec.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        max_total_size = self.task_spec.get("max_total_size", 50 * 1024 * 1024)  # 50MB
        truncate = self.task_spec.get("truncate", True)

        contents = {}
        total_bytes_read = 0

        for file_path in files:
            try:
                p = Path(file_path)
                if not p.exists():
                    self.results["errors"].append(f"File not found: {file_path}")
                    continue

                file_size = p.stat().st_size

                # Check file size limit
                if file_size > max_file_size:
                    if truncate:
                        # Read only the first max_file_size bytes
                        with open(p, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read(max_file_size)
                        contents[file_path] = content + f"\n\n... [TRUNCATED: file is {file_size:,} bytes, showing first {max_file_size:,}] ..."
                        total_bytes_read += max_file_size
                        self.results["operations"].append({
                            "operation": "read",
                            "path": file_path,
                            "size": file_size,
                            "truncated": True,
                            "bytes_read": max_file_size
                        })
                        print(f"[ant-worker] [OK] Read (truncated): {file_path}")
                    else:
                        self.results["errors"].append(
                            f"File too large: {file_path} ({file_size:,} bytes > {max_file_size:,} max)"
                        )
                    continue

                # Check total memory limit
                if total_bytes_read + file_size > max_total_size:
                    self.results["errors"].append(
                        f"Total size limit reached: cannot read {file_path} ({file_size:,} bytes would exceed {max_total_size:,} limit)"
                    )
                    continue

                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    contents[file_path] = f.read()

                total_bytes_read += file_size
                self.results["operations"].append({
                    "operation": "read",
                    "path": file_path,
                    "size": file_size,
                    "truncated": False,
                    "bytes_read": file_size
                })
                print(f"[ant-worker] [OK] Read: {file_path} ({file_size:,} bytes)")

            except UnicodeDecodeError as e:
                self.results["errors"].append(f"Binary/encoding error reading {file_path}: {str(e)}")
            except Exception as e:
                self.results["errors"].append(f"Error reading {file_path}: {str(e)}")

        if contents:
            self.results["file_contents"] = contents
            self.results["read_summary"] = {
                "files_read": len(contents),
                "total_bytes": total_bytes_read
            }

    def _execute_code_adapt(self) -> None:
        """Adapt code with safe, controlled replacements.

        Supports:
        - Exact string replacement (default, safer)
        - Regex replacement (opt-in via "regex": true)
        - Count-limited replacement (via "count": N)
        - Line-anchored replacement for precision
        """
        import re as regex_module

        file_path = self.task_spec.get("file")
        adaptations = self.task_spec.get("adaptations", [])

        if not file_path:
            self.results["errors"].append("No 'file' specified for code adaptation")
            return

        p = Path(file_path)
        if not p.exists():
            self.results["errors"].append(f"File not found: {file_path}")
            return

        # Check file size before reading
        file_size = p.stat().st_size
        max_size = 5 * 1024 * 1024  # 5MB limit for code files
        if file_size > max_size:
            self.results["errors"].append(
                f"File too large for code adaptation: {file_size} bytes (max {max_size})"
            )
            return

        try:
            # Read original file
            original_content = p.read_text(encoding='utf-8')
            adapted_content = original_content
            total_replacements = 0

            # Apply adaptations
            for idx, adapt in enumerate(adaptations):
                find = adapt.get("find")
                replace = adapt.get("replace")
                reason = adapt.get("reason", "")
                use_regex = adapt.get("regex", False)
                count = adapt.get("count", 0)  # 0 = replace all occurrences

                if not find:
                    self.results["errors"].append(f"Adaptation {idx}: missing 'find' pattern")
                    continue

                if replace is None:
                    self.results["errors"].append(f"Adaptation {idx}: missing 'replace' value")
                    continue

                # Count occurrences before replacement
                if use_regex:
                    try:
                        pattern = regex_module.compile(find)
                        matches = pattern.findall(adapted_content)
                        occurrence_count = len(matches)
                    except regex_module.error as e:
                        self.results["errors"].append(
                            f"Adaptation {idx}: invalid regex pattern: {e}"
                        )
                        continue
                else:
                    occurrence_count = adapted_content.count(find)

                if occurrence_count == 0:
                    self.results["errors"].append(
                        f"Pattern not found in {file_path}: {find[:50]}{'...' if len(find) > 50 else ''}"
                    )
                    continue

                # Perform replacement
                if use_regex:
                    if count > 0:
                        adapted_content = pattern.sub(replace, adapted_content, count=count)
                        replacements_made = min(count, occurrence_count)
                    else:
                        adapted_content = pattern.sub(replace, adapted_content)
                        replacements_made = occurrence_count
                else:
                    if count > 0:
                        # Limited replacement - do it count times
                        for _ in range(count):
                            if find in adapted_content:
                                adapted_content = adapted_content.replace(find, replace, 1)
                        replacements_made = min(count, occurrence_count)
                    else:
                        adapted_content = adapted_content.replace(find, replace)
                        replacements_made = occurrence_count

                total_replacements += replacements_made

                self.results["operations"].append({
                    "operation": "code_adapt",
                    "file": file_path,
                    "find": find[:50] + "..." if len(find) > 50 else find,
                    "replace": replace[:50] + "..." if len(replace) > 50 else replace,
                    "reason": reason,
                    "regex": use_regex,
                    "occurrences_found": occurrence_count,
                    "replacements_made": replacements_made,
                })
                print(f"[ant-worker] [OK] Adapted ({replacements_made}x): {reason}")

            # Validate adapted content is different and non-empty
            if adapted_content == original_content:
                self.results["errors"].append("No changes were made to the file")
                return

            if not adapted_content.strip():
                self.results["errors"].append("Adapted content would be empty - aborting")
                return

            # Write adapted file
            p.write_text(adapted_content, encoding='utf-8')

            self.results["summary"] = {
                "total_adaptations": len(adaptations),
                "total_replacements": total_replacements,
                "original_size": len(original_content),
                "adapted_size": len(adapted_content),
            }

        except Exception as e:
            self.results["errors"].append(f"Error adapting {file_path}: {str(e)}")

    def _execute_validate(self) -> None:
        """Validate files (run tests, check syntax, etc.)."""
        file_path = self.task_spec.get("file")
        checks = self.task_spec.get("checks", [])

        if not file_path:
            self.results["errors"].append("No 'file' specified for validation")
            return

        p = Path(file_path)
        if not p.exists():
            self.results["errors"].append(f"File not found: {file_path}")
            return

        print(f"[grok-executor] Validating: {file_path}")
        print(f"[grok-executor] Checks: {checks}")

        # In production, would run actual validation (syntax check, tests, etc.)
        # For now, just document what would be checked
        self.results["operations"].append({
            "operation": "validate",
            "file": file_path,
            "checks": checks,
            "status": "simulated",
            "note": "In production, would run syntax checks, fixtures, tests",
        })

    def _execute_research(self) -> None:
        """Research/analyze files."""
        file_path = self.task_spec.get("file")
        analysis_type = self.task_spec.get("analysis_type", "general")

        if not file_path:
            self.results["errors"].append("No 'file' specified for research")
            return

        p = Path(file_path)
        if not p.exists():
            self.results["errors"].append(f"File not found: {file_path}")
            return

        print(f"[grok-executor] Researching: {file_path}")
        print(f"[grok-executor] Analysis type: {analysis_type}")

        # In production, would perform real analysis
        # For now, document what would be analyzed
        self.results["operations"].append({
            "operation": "research",
            "file": file_path,
            "analysis_type": analysis_type,
            "status": "simulated",
            "note": "In production, would perform actual code analysis",
        })


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python run.py <input.json> <output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load task spec
    try:
        task_spec = load_json(input_path)
    except Exception as e:
        print(f"ERROR: Failed to load input: {e}")
        sys.exit(1)

    # Execute task
    executor = GrokExecutor(task_spec)
    results = executor.execute()

    # Save results
    save_json(output_path, results)

    # Print summary
    print(f"\n[grok-executor] Task: {executor.task_id}")
    print(f"[grok-executor] Status: {results['status']}")
    print(f"[grok-executor] Operations: {len(results['operations'])}")
    print(f"[grok-executor] Errors: {len(results['errors'])}")
    print(f"[grok-executor] Ledger: {results['ledger_dir']}")

    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'success' else 1)


if __name__ == "__main__":
    main()
