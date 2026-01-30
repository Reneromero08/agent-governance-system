#!/usr/bin/env python3
"""
Skill: test-primitives

Test CAT LAB safe primitives (file locking, atomic ops, validation).

Contract-style wrapper:
  python run.py <input.json> <output.json>

Deterministic fixture support:
- If `dry_run` is true, returns mock results
- Otherwise, runs actual primitive tests
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
except ImportError:
    GuardedWriter = None


# =============================================================================
# SAFE PRIMITIVES (Ported from MCP server.py)
# =============================================================================

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


# Validator constants
VALIDATOR_SEMVER = "1.0.0"

DURABLE_ROOTS = [
    "LAW/CONTRACTS/_runs/",
    "NAVIGATION/CORTEX/_generated/",
    "MEMORY/LLM_PACKER/_packs/",
]

CATALYTIC_ROOTS = [
    "LAW/CONTRACTS/_runs/_tmp/",
    "NAVIGATION/CORTEX/_generated/_tmp/",
    "MEMORY/LLM_PACKER/_packs/_tmp/",
    "CAPABILITY/TOOLS/_tmp/",
    "CAPABILITY/MCP/_tmp/",
]

FORBIDDEN_ROOTS = [
    "LAW/CANON/",
    "AGENTS.md",
]

# Task state machine transitions
TASK_STATES = {
    "pending": ["acknowledged", "cancelled"],
    "acknowledged": ["processing", "cancelled"],
    "processing": ["completed", "failed", "timeout", "cancelled"],
    "completed": [],
    "failed": [],
    "timeout": [],
    "cancelled": [],
}


def get_validator_build_id() -> str:
    """Get deterministic validator build fingerprint."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"git:{result.stdout.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    server_path = PROJECT_ROOT / "CAPABILITY" / "MCP" / "server.py"
    if server_path.exists():
        sha = hashlib.sha256()
        with open(server_path, "rb") as f:
            sha.update(f.read())
        return f"file:{sha.hexdigest()[:12]}"

    return "unknown"


def _atomic_write_jsonl(file_path: Path, line: str) -> bool:
    """Atomically append a single line to a JSONL file."""
    try:
        line_bytes = (line.rstrip('\n') + '\n').encode('utf-8')
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='jsonl_',
            dir=file_path.parent
        )
        try:
            os.write(fd, line_bytes)
            os.fsync(fd)
        finally:
            os.close(fd)

        with open(file_path, 'a', encoding='utf-8') as f:
            _lock_file(f, exclusive=True)
            try:
                with open(temp_path, 'r', encoding='utf-8') as tmp:
                    f.write(tmp.read())
                f.flush()
                os.fsync(f.fileno())
            finally:
                _unlock_file(f)

        os.unlink(temp_path)
        return True
    except Exception:
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
    """Atomically rewrite a JSONL file with a transformation function."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.touch()

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
                            continue
            finally:
                _unlock_file(f)

        transformed = transform(entries)

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

        temp_path_obj = Path(temp_path)
        if sys.platform == 'win32':
            backup_path = file_path.with_suffix('.bak')
            try:
                if file_path.exists():
                    shutil.copy2(file_path, backup_path)
                shutil.move(str(temp_path_obj), str(file_path))
                if backup_path.exists():
                    backup_path.unlink()
            except Exception:
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    backup_path.unlink()
                raise
        else:
            os.replace(temp_path, file_path)

        return True
    except Exception:
        try:
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception:
            pass
        return False


def _read_jsonl_streaming(
    file_path: Path,
    filter_fn: Optional[Callable[[Dict], bool]] = None,
    limit: int = 100,
    offset: int = 0
) -> Iterator[Dict]:
    """Stream JSONL file with optional filtering and pagination."""
    if not file_path.exists():
        return

    count = 0
    skipped = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        _lock_file(f, exclusive=False)
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if filter_fn and not filter_fn(entry):
                    continue

                if skipped < offset:
                    skipped += 1
                    continue

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
    """Validate task_spec has required fields and valid structure."""
    import re
    errors = []

    required_fields = ["task_id", "task_type"]
    for field in required_fields:
        if field not in task_spec:
            errors.append(f"Missing required field: {field}")

    valid_task_types = ["file_operation", "code_adapt", "validate", "research"]
    if "task_type" in task_spec and task_spec["task_type"] not in valid_task_types:
        errors.append(f"Invalid task_type: {task_spec['task_type']}")

    if "task_id" in task_spec:
        task_id = task_spec["task_id"]
        if not isinstance(task_id, str) or not re.match(r'^[\w\-]+$', task_id):
            errors.append("Invalid task_id format")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def _compute_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


# =============================================================================
# TEST RUNNER
# =============================================================================

def _load_json(path: Path) -> Dict[str, Any]:
    """Load and validate JSON input."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("input must be a JSON object")
    return obj


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON output using GuardedWriter."""
    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        sys.exit(1)

    writer = GuardedWriter(
        PROJECT_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"]
    )

    rel_path = path.resolve().relative_to(PROJECT_ROOT)
    writer.mkdir_auto(str(rel_path.parent))
    content = json.dumps(obj, indent=2, sort_keys=True)
    writer.write_auto(str(rel_path), content)


def run_tests() -> Dict[str, Any]:
    """Run all primitive tests."""
    results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "details": []
    }

    def test(name: str, func):
        """Run a test and record result."""
        results["tests_run"] += 1
        try:
            func()
            results["tests_passed"] += 1
            results["details"].append(f"PASS: {name}")
        except Exception as e:
            results["tests_failed"] += 1
            results["details"].append(f"FAIL: {name}: {str(e)}")

    # Test 1: File locking
    def test_file_locking():
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                temp_path = f.name
                _lock_file(f, exclusive=True)
                f.write("test")
                _unlock_file(f)
        finally:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()

    test("File locking (Windows/Unix)", test_file_locking)

    # Test 2: Atomic JSONL write
    def test_atomic_write():
        temp_dir = Path(tempfile.mkdtemp())
        try:
            test_file = temp_dir / "test.jsonl"
            assert _atomic_write_jsonl(test_file, json.dumps({"id": 1}))
            assert _atomic_write_jsonl(test_file, json.dumps({"id": 2}))
            lines = test_file.read_text().strip().split('\n')
            assert len(lines) == 2
        finally:
            shutil.rmtree(temp_dir)

    test("Atomic JSONL write", test_atomic_write)

    # Test 3: Atomic JSONL rewrite
    def test_atomic_rewrite():
        temp_dir = Path(tempfile.mkdtemp())
        try:
            test_file = temp_dir / "test.jsonl"
            _atomic_write_jsonl(test_file, json.dumps({"status": "pending"}))
            assert _atomic_rewrite_jsonl(
                test_file,
                lambda entries: [{**e, "status": "done"} for e in entries]
            )
            line = test_file.read_text().strip()
            assert json.loads(line)["status"] == "done"
        finally:
            shutil.rmtree(temp_dir)

    test("Atomic JSONL rewrite", test_atomic_rewrite)

    # Test 4: Streaming JSONL reader
    def test_streaming():
        temp_dir = Path(tempfile.mkdtemp())
        try:
            test_file = temp_dir / "test.jsonl"
            for i in range(5):
                _atomic_write_jsonl(test_file, json.dumps({"id": i}))
            results_list = list(_read_jsonl_streaming(test_file, limit=3))
            assert len(results_list) == 3
        finally:
            shutil.rmtree(temp_dir)

    test("Streaming JSONL reader", test_streaming)

    # Test 5: Task state validation
    def test_task_states():
        assert _validate_task_state_transition("pending", "acknowledged") == True
        assert _validate_task_state_transition("pending", "completed") == False
        assert _validate_task_state_transition("completed", "failed") == False

    test("Task state transitions", test_task_states)

    # Test 6: Task spec validation
    def test_task_spec():
        valid = _validate_task_spec({"task_id": "test-1", "task_type": "validate"})
        assert valid["valid"] == True
        invalid = _validate_task_spec({"task_id": "bad@id", "task_type": "invalid"})
        assert invalid["valid"] == False

    test("Task spec validation", test_task_spec)

    # Test 7: File hashing
    def test_hash():
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"test")
        try:
            hash1 = _compute_hash(temp_path)
            assert len(hash1) == 64  # SHA-256
        finally:
            temp_path.unlink()

    test("File hashing (SHA-256)", test_hash)

    # Test 8: Constants
    def test_constants():
        assert VALIDATOR_SEMVER == "1.0.0"
        assert len(DURABLE_ROOTS) == 3
        assert len(CATALYTIC_ROOTS) == 5
        assert len(FORBIDDEN_ROOTS) == 2
        build_id = get_validator_build_id()
        assert build_id.startswith("git:") or build_id.startswith("file:") or build_id == "unknown"

    test("Constants and build ID", test_constants)

    # Build summary
    summary = f"Tests: {results['tests_passed']}/{results['tests_run']} passed"
    if results['tests_failed'] > 0:
        summary += f", {results['tests_failed']} failed"

    return {
        **results,
        "summary": summary,
        "validator_semver": VALIDATOR_SEMVER,
        "build_id": get_validator_build_id(),
        "durable_roots_count": len(DURABLE_ROOTS),
        "catalytic_roots_count": len(CATALYTIC_ROOTS)
    }


def run_mock_tests() -> Dict[str, Any]:
    """Return mock test results for deterministic testing."""
    return {
        "tests_run": 8,
        "tests_passed": 8,
        "tests_failed": 0,
        "details": [
            "PASS: File locking (Windows/Unix)",
            "PASS: Atomic JSONL write",
            "PASS: Atomic JSONL rewrite",
            "PASS: Streaming JSONL reader",
            "PASS: Task state transitions",
            "PASS: Task spec validation",
            "PASS: File hashing (SHA-256)",
            "PASS: Constants and build ID"
        ],
        "summary": "Tests: 8/8 passed",
        "validator_semver": "1.0.0",
        "build_id": "mock:deterministic",
        "durable_roots_count": 3,
        "catalytic_roots_count": 5
    }


def main(argv: list) -> int:
    """Main entry point."""
    if len(argv) != 3:
        sys.stderr.write("Usage: run.py <input.json> <output.json>\n")
        return 2

    input_path = Path(argv[1])
    output_path = Path(argv[2])

    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        inp = _load_json(input_path)
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    dry_run = inp.get("dry_run", False)
    run_actual_tests = inp.get("run_tests", True)

    if dry_run:
        result = run_mock_tests()
    elif run_actual_tests:
        result = run_tests()
    else:
        result = {"message": "Tests skipped (run_tests=false)"}

    # Merge input with result for traceability
    output = {**inp, **result}

    try:
        _write_json(output_path, output)
    except Exception as e:
        print(f"Error writing output: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
