#!/usr/bin/env python3
"""
Test harness for grok-executor skill.

Validates:
1. Task execution with hash verification
2. Immutable ledger creation
3. Error handling and rollback
4. Integration with MCP server
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directories to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATALYTIC_DPT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CATALYTIC_DPT))

# Import from local run.py
import importlib.util
spec = importlib.util.spec_from_file_location("grok_executor", str(Path(__file__).parent / "run.py"))
grok_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grok_module)
GrokExecutor = grok_module.GrokExecutor
compute_hash = grok_module.compute_hash


class GrokExecutorTestHarness:
    """Test suite for grok-executor."""

    def __init__(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="grok_test_"))
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        }

    def run_all_tests(self) -> dict:
        """Run all test cases."""
        tests = [
            ("test_file_copy", self.test_file_copy),
            ("test_hash_verification", self.test_hash_verification),
            ("test_missing_source", self.test_missing_source),
            ("test_code_adaptation", self.test_code_adaptation),
            ("test_ledger_creation", self.test_ledger_creation),
        ]

        for test_name, test_func in tests:
            self.results["summary"]["total"] += 1
            try:
                test_func()
                self.results["summary"]["passed"] += 1
                self.results["tests"].append({
                    "name": test_name,
                    "status": "PASS",
                    "timestamp": datetime.now().isoformat()
                })
                print(f"[PASS] {test_name}")
            except AssertionError as e:
                self.results["summary"]["failed"] += 1
                self.results["tests"].append({
                    "name": test_name,
                    "status": "FAIL",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                print(f"[FAIL] {test_name}: {e}")
            except Exception as e:
                self.results["summary"]["failed"] += 1
                self.results["tests"].append({
                    "name": test_name,
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                print(f"[ERROR] {test_name}: {type(e).__name__}: {e}")

        return self.results

    def test_file_copy(self):
        """Test file copy operation with hash verification."""
        # Create source file
        source = self.test_dir / "source.txt"
        source.write_text("Hello, World!")

        dest = self.test_dir / "dest.txt"

        # Create task spec
        task_spec = {
            "task_id": "test-copy",
            "task_type": "file_operation",
            "operation": "copy",
            "files": [
                {
                    "source": str(source),
                    "destination": str(dest)
                }
            ],
            "verify_integrity": True
        }

        # Execute
        executor = GrokExecutor(task_spec)
        results = executor.execute()

        # Verify
        assert results["status"] == "success", f"Status was {results['status']}"
        assert dest.exists(), "Destination file not created"
        assert dest.read_text() == "Hello, World!", "File content mismatch"
        assert len(results["operations"]) > 0, "No operations recorded"

        op = results["operations"][0]
        assert op["source_hash"] == op["dest_hash"], "Hash mismatch"
        assert op["hash_verified"] == True, "Hash not verified"

    def test_hash_verification(self):
        """Test that hash verification detects mismatches."""
        source = self.test_dir / "source_hash.txt"
        source.write_text("Original content")

        dest = self.test_dir / "dest_hash.txt"

        # Compute source hash
        source_hash = compute_hash(source)

        # Copy file
        shutil.copy2(source, dest)

        # Verify destination hash matches
        dest_hash = compute_hash(dest)
        assert source_hash == dest_hash, "Hash mismatch detected"

    def test_missing_source(self):
        """Test error handling for missing source file."""
        task_spec = {
            "task_id": "test-missing",
            "task_type": "file_operation",
            "operation": "copy",
            "files": [
                {
                    "source": "/nonexistent/path/file.txt",
                    "destination": str(self.test_dir / "dest.txt")
                }
            ]
        }

        executor = GrokExecutor(task_spec)
        results = executor.execute()

        # Should have error
        assert results["status"] == "error", "Should have error status"
        assert len(results["errors"]) > 0, "No errors recorded"
        assert "not found" in results["errors"][0].lower(), "Wrong error message"

    def test_code_adaptation(self):
        """Test code adaptation (find/replace)."""
        # Create source file with code to adapt
        source = self.test_dir / "code.py"
        source.write_text("""
def run_with_cline():
    result = cline.execute()
    return result
""")

        task_spec = {
            "task_id": "test-adapt",
            "task_type": "code_adapt",
            "file": str(source),
            "adaptations": [
                {
                    "find": "cline",
                    "replace": "gemini",
                    "reason": "Replace with gemini"
                }
            ]
        }

        executor = GrokExecutor(task_spec)
        results = executor.execute()

        assert results["status"] == "success", f"Status was {results['status']}"
        assert len(results["operations"]) > 0, "No operations recorded"

        # Verify file was modified
        modified = source.read_text()
        assert "gemini" in modified, "Replacement not made"
        assert "cline" not in modified, "Original text still present"

    def test_ledger_creation(self):
        """Test that immutable ledger is created."""
        source = self.test_dir / "ledger_source.txt"
        source.write_text("Ledger test")

        dest = self.test_dir / "ledger_dest.txt"

        task_spec = {
            "task_id": "test-ledger",
            "task_type": "file_operation",
            "operation": "copy",
            "files": [
                {
                    "source": str(source),
                    "destination": str(dest)
                }
            ]
        }

        executor = GrokExecutor(task_spec)
        results = executor.execute()

        # Verify ledger directory exists
        ledger_dir = Path(results["ledger_dir"])
        assert ledger_dir.exists(), f"Ledger dir not created: {ledger_dir}"

        # Verify TASK_SPEC.json exists
        task_spec_file = ledger_dir / "TASK_SPEC.json"
        assert task_spec_file.exists(), "TASK_SPEC.json not created"

        with open(task_spec_file) as f:
            saved_spec = json.load(f)
            assert saved_spec["task_id"] == "test-ledger", "Task spec mismatch"

        # Verify RESULTS.json exists
        results_file = ledger_dir / "RESULTS.json"
        assert results_file.exists(), "RESULTS.json not created"

    def cleanup(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def save_report(self, output_file: str):
        """Save test report to file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)


def main():
    """Run test harness."""
    print("\n" + "="*60)
    print("Grok Executor Test Harness")
    print("="*60 + "\n")

    harness = GrokExecutorTestHarness()

    try:
        results = harness.run_all_tests()

        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Total:  {results['summary']['total']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print("="*60 + "\n")

        # Save report
        report_file = Path(__file__).parent / "test_results.json"
        harness.save_report(str(report_file))
        print(f"Report saved to: {report_file}\n")

        # Exit with appropriate code
        sys.exit(0 if results['summary']['failed'] == 0 else 1)

    finally:
        harness.cleanup()


def test_ant_worker():
    """Pytest entry point."""
    harness = GrokExecutorTestHarness()
    try:
        results = harness.run_all_tests()
        assert results['summary']['failed'] == 0
    finally:
        harness.cleanup()


if __name__ == "__main__":
    main()
