#!/usr/bin/env python3
"""
Compression Benchmark Tasks

Deterministic benchmark suite for validating compression preserves task success.
Tasks are evaluated with both baseline (full context) and compressed (retrieved) context.

Requirements:
- Tasks must be deterministic (same input -> same output)
- Tasks must be reproducible from fixtures
- Compression is valid only when compressed_success_rate >= baseline_success_rate
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Project root detection
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_VERSION = "1.0.0"


class TaskType(Enum):
    """Types of benchmark tasks."""
    CODE_COMPILES = "code_compiles"
    TESTS_PASS = "tests_pass"
    BUGS_FOUND = "bugs_found"
    SEMANTIC_MATCH = "semantic_match"


@dataclass
class BenchmarkTask:
    """A single benchmark task definition."""
    task_id: str
    task_type: TaskType
    description: str
    fixture_path: Optional[Path] = None
    expected_answer: Optional[str] = None
    validator: Optional[Callable[[str, str], bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "fixture_path": str(self.fixture_path) if self.fixture_path else None,
        }


@dataclass
class TaskResult:
    """Result of running a benchmark task."""
    task_id: str
    task_type: str
    passed: bool
    tokens_used: int
    context_type: str  # "baseline" or "compressed"
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "passed": self.passed,
            "tokens_used": self.tokens_used,
            "context_type": self.context_type,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class BenchmarkResults:
    """Aggregate results from running benchmark suite."""
    benchmark_version: str
    tasks_run: int
    baseline_results: Dict[str, Any]
    compressed_results: Dict[str, Any]
    parity_achieved: bool
    task_details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "tasks_run": self.tasks_run,
            "baseline_results": self.baseline_results,
            "compressed_results": self.compressed_results,
            "parity_achieved": self.parity_achieved,
            "task_details": self.task_details,
        }


def _sha256_hex(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_tokens_proxy(text: str) -> int:
    """
    Count tokens using tiktoken if available, else word-count proxy.

    For benchmark reproducibility, this must match compression proof tokenizer.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback: word-count proxy
        return int(len(text.split()) / 0.75)


# ============================================================================
# SEMANTIC MATCH TASKS (Deterministic, no LLM required)
# ============================================================================

def validate_semantic_contains(response: str, expected: str) -> bool:
    """Check if response contains expected substring (case-insensitive)."""
    return expected.lower() in response.lower()


def validate_semantic_pattern(response: str, pattern: str) -> bool:
    """Check if response matches regex pattern."""
    return bool(re.search(pattern, response, re.IGNORECASE))


def validate_json_field(response: str, field_path: str, expected_value: Any) -> bool:
    """Check if JSON response has expected field value."""
    try:
        data = json.loads(response)
        parts = field_path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                return False
        return current == expected_value
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return False


# ============================================================================
# BENCHMARK TASK FIXTURES
# ============================================================================

def get_semantic_match_tasks() -> List[BenchmarkTask]:
    """
    Get semantic match tasks - deterministic string matching tests.

    These tasks verify that retrieval returns content containing expected keywords.
    No LLM is required; validation is pure string/regex matching.
    """
    tasks = [
        BenchmarkTask(
            task_id="semantic_001",
            task_type=TaskType.SEMANTIC_MATCH,
            description="Retrieve content about Genesis Compact",
            expected_answer="AGS BOOTSTRAP",
            validator=lambda r, e: validate_semantic_contains(r, e),
        ),
        BenchmarkTask(
            task_id="semantic_002",
            task_type=TaskType.SEMANTIC_MATCH,
            description="Retrieve content about Translation Layer",
            expected_answer="semantic",
            validator=lambda r, e: validate_semantic_contains(r, e),
        ),
        BenchmarkTask(
            task_id="semantic_003",
            task_type=TaskType.SEMANTIC_MATCH,
            description="Retrieve content about Semiotic symbols",
            expected_answer="symbol",
            validator=lambda r, e: validate_semantic_contains(r, e),
        ),
        BenchmarkTask(
            task_id="semantic_004",
            task_type=TaskType.SEMANTIC_MATCH,
            description="Retrieve ADR content",
            expected_answer="decision",
            validator=lambda r, e: validate_semantic_contains(r, e),
        ),
        BenchmarkTask(
            task_id="semantic_005",
            task_type=TaskType.SEMANTIC_MATCH,
            description="Retrieve roadmap content",
            expected_answer="phase",
            validator=lambda r, e: validate_semantic_contains(r, e),
        ),
    ]
    return tasks


def get_code_compile_tasks() -> List[BenchmarkTask]:
    """
    Get code compilation tasks.

    These tasks verify that code snippets compile/parse correctly.
    For Python: ast.parse()
    For JSON: json.loads()
    """
    # These would load from fixtures in a real implementation
    tasks = [
        BenchmarkTask(
            task_id="compile_001",
            task_type=TaskType.CODE_COMPILES,
            description="Verify Python function syntax",
            fixture_path=REPO_ROOT / "CAPABILITY" / "PRIMITIVES" / "cassette_receipt.py",
        ),
        BenchmarkTask(
            task_id="compile_002",
            task_type=TaskType.CODE_COMPILES,
            description="Verify JSON schema syntax",
            fixture_path=REPO_ROOT / "LAW" / "SCHEMAS" / "cassette_receipt.schema.json",
        ),
    ]
    return tasks


def get_test_pass_tasks() -> List[BenchmarkTask]:
    """
    Get test pass tasks.

    These tasks verify that specific test files pass when run.
    """
    tasks = [
        BenchmarkTask(
            task_id="test_001",
            task_type=TaskType.TESTS_PASS,
            description="Cassette receipt tests pass",
            fixture_path=REPO_ROOT / "CAPABILITY" / "TESTBENCH" / "mcp-capability-tests" / "test_cassette_receipt.py",
        ),
    ]
    return tasks


def get_all_benchmark_tasks() -> List[BenchmarkTask]:
    """Get all benchmark tasks for the suite."""
    tasks = []
    tasks.extend(get_semantic_match_tasks())
    tasks.extend(get_code_compile_tasks())
    tasks.extend(get_test_pass_tasks())
    return tasks


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """
    Runs benchmark tasks with baseline and compressed contexts.

    For each task:
    1. Run with baseline (full corpus) context
    2. Run with compressed (retrieved) context
    3. Compare success rates
    """

    def __init__(self, retriever: Optional[Callable[[str], str]] = None):
        """
        Initialize benchmark runner.

        Args:
            retriever: Function that takes a query and returns retrieved context.
                      If None, uses identity (full corpus) for both baseline and compressed.
        """
        self.retriever = retriever
        self._baseline_corpus: Optional[str] = None

    def set_baseline_corpus(self, corpus: str) -> None:
        """Set the baseline corpus for full-context evaluation."""
        self._baseline_corpus = corpus

    def run_task_semantic(
        self,
        task: BenchmarkTask,
        context: str,
        context_type: str,
    ) -> TaskResult:
        """Run a semantic match task."""
        tokens_used = _count_tokens_proxy(context)

        # For semantic match, check if expected answer is in context
        if task.validator and task.expected_answer:
            passed = task.validator(context, task.expected_answer)
        else:
            # Default: check if any relevant content was retrieved
            passed = len(context.strip()) > 0

        return TaskResult(
            task_id=task.task_id,
            task_type=task.task_type.value,
            passed=passed,
            tokens_used=tokens_used,
            context_type=context_type,
        )

    def run_task_compile(
        self,
        task: BenchmarkTask,
        context: str,
        context_type: str,
    ) -> TaskResult:
        """Run a code compilation task."""
        import ast

        tokens_used = _count_tokens_proxy(context)
        passed = False
        error_message = None

        if task.fixture_path and task.fixture_path.exists():
            content = task.fixture_path.read_text(encoding="utf-8")
            try:
                if task.fixture_path.suffix == ".py":
                    ast.parse(content)
                    passed = True
                elif task.fixture_path.suffix == ".json":
                    json.loads(content)
                    passed = True
                else:
                    # Unknown type, assume pass if file exists
                    passed = True
            except (SyntaxError, json.JSONDecodeError) as e:
                error_message = str(e)
        else:
            error_message = f"Fixture not found: {task.fixture_path}"

        return TaskResult(
            task_id=task.task_id,
            task_type=task.task_type.value,
            passed=passed,
            tokens_used=tokens_used,
            context_type=context_type,
            error_message=error_message,
        )

    def run_task_test(
        self,
        task: BenchmarkTask,
        context: str,
        context_type: str,
    ) -> TaskResult:
        """Run a test pass task."""
        tokens_used = _count_tokens_proxy(context)
        passed = False
        error_message = None

        if task.fixture_path and task.fixture_path.exists():
            try:
                # Run pytest on the fixture
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(task.fixture_path), "-v", "--tb=short"],
                    capture_output=True,
                    timeout=60,
                    cwd=str(REPO_ROOT),
                )
                passed = result.returncode == 0
                if not passed:
                    error_message = result.stderr.decode("utf-8", errors="replace")[:500]
            except subprocess.TimeoutExpired:
                error_message = "Test timed out after 60 seconds"
            except Exception as e:
                error_message = str(e)
        else:
            error_message = f"Test file not found: {task.fixture_path}"

        return TaskResult(
            task_id=task.task_id,
            task_type=task.task_type.value,
            passed=passed,
            tokens_used=tokens_used,
            context_type=context_type,
            error_message=error_message,
        )

    def run_task(
        self,
        task: BenchmarkTask,
        context: str,
        context_type: str,
    ) -> TaskResult:
        """Run a single task with given context."""
        if task.task_type == TaskType.SEMANTIC_MATCH:
            return self.run_task_semantic(task, context, context_type)
        elif task.task_type == TaskType.CODE_COMPILES:
            return self.run_task_compile(task, context, context_type)
        elif task.task_type == TaskType.TESTS_PASS:
            return self.run_task_test(task, context, context_type)
        else:
            return TaskResult(
                task_id=task.task_id,
                task_type=task.task_type.value,
                passed=False,
                tokens_used=0,
                context_type=context_type,
                error_message=f"Unknown task type: {task.task_type}",
            )

    def run_suite(
        self,
        tasks: Optional[List[BenchmarkTask]] = None,
    ) -> BenchmarkResults:
        """
        Run full benchmark suite comparing baseline vs compressed.

        Returns aggregate results with parity check.
        """
        if tasks is None:
            tasks = get_all_benchmark_tasks()

        baseline_corpus = self._baseline_corpus or ""

        baseline_passed = 0
        baseline_failed = 0
        compressed_passed = 0
        compressed_failed = 0
        task_details = []

        for task in tasks:
            # Run with baseline context
            baseline_result = self.run_task(task, baseline_corpus, "baseline")

            # Run with compressed context
            if self.retriever:
                compressed_context = self.retriever(task.description)
            else:
                compressed_context = baseline_corpus
            compressed_result = self.run_task(task, compressed_context, "compressed")

            # Track results
            if baseline_result.passed:
                baseline_passed += 1
            else:
                baseline_failed += 1

            if compressed_result.passed:
                compressed_passed += 1
            else:
                compressed_failed += 1

            task_details.append({
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "baseline_passed": baseline_result.passed,
                "compressed_passed": compressed_result.passed,
                "tokens_used_baseline": baseline_result.tokens_used,
                "tokens_used_compressed": compressed_result.tokens_used,
            })

        tasks_run = len(tasks)
        baseline_success_rate = baseline_passed / tasks_run if tasks_run > 0 else 0.0
        compressed_success_rate = compressed_passed / tasks_run if tasks_run > 0 else 0.0
        parity_achieved = compressed_success_rate >= baseline_success_rate

        return BenchmarkResults(
            benchmark_version=BENCHMARK_VERSION,
            tasks_run=tasks_run,
            baseline_results={
                "tasks_passed": baseline_passed,
                "tasks_failed": baseline_failed,
                "success_rate": round(baseline_success_rate, 6),
            },
            compressed_results={
                "tasks_passed": compressed_passed,
                "tasks_failed": compressed_failed,
                "success_rate": round(compressed_success_rate, 6),
            },
            parity_achieved=parity_achieved,
            task_details=task_details,
        )


def run_benchmarks(
    retriever: Optional[Callable[[str], str]] = None,
    baseline_corpus: Optional[str] = None,
) -> BenchmarkResults:
    """
    Convenience function to run benchmark suite.

    Args:
        retriever: Function that retrieves compressed context for a query
        baseline_corpus: Full corpus text for baseline evaluation

    Returns:
        BenchmarkResults with parity check
    """
    runner = BenchmarkRunner(retriever=retriever)
    if baseline_corpus:
        runner.set_baseline_corpus(baseline_corpus)
    return runner.run_suite()


if __name__ == "__main__":
    # Run benchmarks standalone for testing
    print(f"Benchmark Suite v{BENCHMARK_VERSION}")
    print("=" * 60)

    results = run_benchmarks()

    print(f"\nTasks run: {results.tasks_run}")
    print(f"\nBaseline results:")
    print(f"  Passed: {results.baseline_results['tasks_passed']}")
    print(f"  Failed: {results.baseline_results['tasks_failed']}")
    print(f"  Success rate: {results.baseline_results['success_rate']:.2%}")

    print(f"\nCompressed results:")
    print(f"  Passed: {results.compressed_results['tasks_passed']}")
    print(f"  Failed: {results.compressed_results['tasks_failed']}")
    print(f"  Success rate: {results.compressed_results['success_rate']:.2%}")

    print(f"\nParity achieved: {results.parity_achieved}")

    if not results.parity_achieved:
        print("\nWARNING: Compression validation FAILED - success rate dropped!")
        sys.exit(1)
    else:
        print("\nSUCCESS: Compression maintains task success parity")
        sys.exit(0)
