#!/usr/bin/env python3
"""
Phase 6.4 LLM Benchmark Runner

Integrates with local Nemotron model to run actual semantic Q&A tests.
Tests whether compressed context preserves task success compared to baseline.

Endpoint: http://10.5.0.2:1234 (configurable)
Model: nemotron-3-nano-30b-a3b (configurable)

Usage:
    from llm_benchmark_runner import run_llm_benchmarks
    results = run_llm_benchmarks(baseline_corpus, compressed_corpus)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Project root detection
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import requests
except ImportError:
    requests = None

# Configuration
LLM_ENDPOINT = "http://10.5.0.2:1234/v1/chat/completions"
LLM_MODEL = "nemotron-3-nano-30b-a3b"
LLM_TIMEOUT = 120  # seconds
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.3  # Lower for more deterministic responses

BENCHMARK_VERSION = "1.1.0"  # LLM-enabled version


@dataclass
class LLMTask:
    """A semantic Q&A task for LLM evaluation."""
    task_id: str
    question: str
    expected_keywords: List[str]  # Answer should contain these
    context_hint: str  # What part of codebase to focus on
    difficulty: str = "easy"  # easy, medium, hard

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "expected_keywords": self.expected_keywords,
            "context_hint": self.context_hint,
            "difficulty": self.difficulty,
        }


@dataclass
class LLMTaskResult:
    """Result of running an LLM benchmark task."""
    task_id: str
    passed: bool
    context_type: str  # "baseline" or "compressed"
    response: str
    matched_keywords: List[str]
    missing_keywords: List[str]
    latency_ms: float
    tokens_used: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "context_type": self.context_type,
            "response": self.response[:500] + "..." if len(self.response) > 500 else self.response,
            "matched_keywords": self.matched_keywords,
            "missing_keywords": self.missing_keywords,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
            "error": self.error,
        }


@dataclass
class LLMBenchmarkResults:
    """Aggregate results from LLM benchmark suite."""
    benchmark_version: str
    endpoint: str
    model: str
    tasks_run: int
    baseline_results: Dict[str, Any]
    compressed_results: Dict[str, Any]
    parity_achieved: bool
    task_details: List[Dict[str, Any]]
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "endpoint": self.endpoint,
            "model": self.model,
            "tasks_run": self.tasks_run,
            "baseline_results": self.baseline_results,
            "compressed_results": self.compressed_results,
            "parity_achieved": self.parity_achieved,
            "task_details": self.task_details,
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


def _call_llm(
    prompt: str,
    context: str,
    endpoint: str = LLM_ENDPOINT,
    model: str = LLM_MODEL,
) -> Tuple[str, int, float]:
    """
    Call the LLM with a prompt and context.

    Returns:
        Tuple of (response_text, tokens_used, latency_ms)
    """
    if requests is None:
        raise ImportError("requests library required: pip install requests")

    system_prompt = """You are a helpful assistant answering questions about a codebase.
Use ONLY the provided context to answer. Be concise and factual.
If the answer is not in the context, say "I cannot find this information in the provided context."
"""

    user_prompt = f"""Context:
{context[:32000]}

Question: {prompt}

Answer concisely based only on the context above:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    start = time.time()
    response = requests.post(endpoint, json=payload, timeout=LLM_TIMEOUT)
    latency_ms = (time.time() - start) * 1000

    response.raise_for_status()
    data = response.json()

    response_text = data["choices"][0]["message"]["content"]
    tokens_used = data.get("usage", {}).get("total_tokens", 0)

    return response_text, tokens_used, latency_ms


def _validate_response(response: str, expected_keywords: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate LLM response against expected keywords.

    Returns:
        Tuple of (passed, matched_keywords, missing_keywords)
    """
    response_lower = response.lower()
    matched = []
    missing = []

    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            matched.append(keyword)
        else:
            missing.append(keyword)

    # Pass if at least 50% of keywords matched (allows partial success)
    passed = len(matched) >= len(expected_keywords) / 2

    return passed, matched, missing


# ============================================================================
# LLM BENCHMARK TASKS
# ============================================================================

def get_llm_semantic_tasks() -> List[LLMTask]:
    """
    Get LLM-based semantic Q&A tasks.

    These tasks test whether the LLM can correctly answer questions
    about the codebase using either full or compressed context.

    Each task validated against actual file contents:
    - Task 1: GENESIS.md - bootstrap prompt for AGS
    - Task 2: memory_record.py - canonical data structure
    - Task 3: cassette_receipt.py - receipt chain with Merkle roots
    - Task 4: memory_cassette.py - cartridge export/import
    - Task 5: CORTEX/README.md - cassette network architecture
    """
    tasks = [
        LLMTask(
            task_id="llm_semantic_001",
            question="What is the Genesis Prompt used for in this system?",
            # From GENESIS.md: "bootstrap prompt for the Agent Governance System"
            # "LOAD ORDER (strict priority)", "Text is law. Code is consequence."
            expected_keywords=["bootstrap", "load", "canon", "governance"],
            context_hint="LAW/GENESIS",
            difficulty="easy",
        ),
        LLMTask(
            task_id="llm_semantic_002",
            question="What fields does the MemoryRecord data structure contain?",
            # From memory_record.py: "id, text, embeddings, payload, scores, lineage, receipts"
            # "Text is canonical (source of truth)"
            expected_keywords=["text", "embeddings", "payload", "receipts"],
            context_hint="CAPABILITY/PRIMITIVES/memory_record",
            difficulty="easy",
        ),
        LLMTask(
            task_id="llm_semantic_003",
            question="How does the cassette receipt system ensure data integrity?",
            # From cassette_receipt.py: "Receipts form a chain via parent_receipt_hash"
            # "compute_session_merkle_root"
            expected_keywords=["receipt", "hash", "chain", "merkle"],
            context_hint="cassette_receipt",
            difficulty="medium",
        ),
        LLMTask(
            task_id="llm_semantic_004",
            question="What is the cartridge export/import cycle used for?",
            # From memory_cassette.py: "Export cassette as a portable cartridge"
            # "Import a cartridge and restore cassette state"
            expected_keywords=["export", "import", "restore", "portable"],
            context_hint="cartridge",
            difficulty="medium",
        ),
        LLMTask(
            task_id="llm_semantic_005",
            question="What is the CORTEX cassette network used for?",
            # From CORTEX/README.md: "cassette network - modular SQLite databases
            # with FTS5 full-text search and vector embeddings"
            # "semantic search and navigation layer"
            expected_keywords=["cassette", "semantic", "search", "navigation"],
            context_hint="NAVIGATION/CORTEX",
            difficulty="hard",
        ),
    ]
    return tasks


# ============================================================================
# LLM BENCHMARK RUNNER
# ============================================================================

class LLMBenchmarkRunner:
    """
    Runs LLM-based benchmark tasks comparing baseline vs compressed context.

    Requires a running LLM endpoint (Nemotron or compatible).
    """

    def __init__(
        self,
        endpoint: str = LLM_ENDPOINT,
        model: str = LLM_MODEL,
    ):
        self.endpoint = endpoint
        self.model = model
        self._baseline_corpus: str = ""
        self._compressed_corpus: str = ""

    def set_baseline_corpus(self, corpus: str) -> None:
        """Set the baseline (full) corpus."""
        self._baseline_corpus = corpus

    def set_compressed_corpus(self, corpus: str) -> None:
        """Set the compressed (retrieved) corpus."""
        self._compressed_corpus = corpus

    def check_endpoint(self) -> bool:
        """Check if LLM endpoint is available."""
        if requests is None:
            return False
        try:
            response = requests.get(
                self.endpoint.replace("/chat/completions", "/models"),
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def run_task(
        self,
        task: LLMTask,
        context: str,
        context_type: str,
    ) -> LLMTaskResult:
        """Run a single LLM task."""
        try:
            response, tokens_used, latency_ms = _call_llm(
                prompt=task.question,
                context=context,
                endpoint=self.endpoint,
                model=self.model,
            )

            passed, matched, missing = _validate_response(
                response, task.expected_keywords
            )

            return LLMTaskResult(
                task_id=task.task_id,
                passed=passed,
                context_type=context_type,
                response=response,
                matched_keywords=matched,
                missing_keywords=missing,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )

        except Exception as e:
            return LLMTaskResult(
                task_id=task.task_id,
                passed=False,
                context_type=context_type,
                response="",
                matched_keywords=[],
                missing_keywords=task.expected_keywords,
                latency_ms=0,
                tokens_used=0,
                error=str(e),
            )

    def run_suite(
        self,
        tasks: Optional[List[LLMTask]] = None,
    ) -> LLMBenchmarkResults:
        """
        Run full LLM benchmark suite comparing baseline vs compressed.

        Returns aggregate results with parity check.
        """
        if tasks is None:
            tasks = get_llm_semantic_tasks()

        baseline_passed = 0
        baseline_failed = 0
        compressed_passed = 0
        compressed_failed = 0
        task_details = []
        total_latency = 0.0

        for task in tasks:
            # Run with baseline context
            baseline_result = self.run_task(
                task, self._baseline_corpus, "baseline"
            )
            total_latency += baseline_result.latency_ms

            # Run with compressed context
            compressed_result = self.run_task(
                task, self._compressed_corpus, "compressed"
            )
            total_latency += compressed_result.latency_ms

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
                "question": task.question,
                "difficulty": task.difficulty,
                "baseline_passed": baseline_result.passed,
                "baseline_matched": baseline_result.matched_keywords,
                "compressed_passed": compressed_result.passed,
                "compressed_matched": compressed_result.matched_keywords,
                "baseline_latency_ms": baseline_result.latency_ms,
                "compressed_latency_ms": compressed_result.latency_ms,
            })

        tasks_run = len(tasks)
        baseline_rate = baseline_passed / tasks_run if tasks_run > 0 else 0.0
        compressed_rate = compressed_passed / tasks_run if tasks_run > 0 else 0.0
        parity_achieved = compressed_rate >= baseline_rate

        return LLMBenchmarkResults(
            benchmark_version=BENCHMARK_VERSION,
            endpoint=self.endpoint,
            model=self.model,
            tasks_run=tasks_run,
            baseline_results={
                "tasks_passed": baseline_passed,
                "tasks_failed": baseline_failed,
                "success_rate": round(baseline_rate, 4),
            },
            compressed_results={
                "tasks_passed": compressed_passed,
                "tasks_failed": compressed_failed,
                "success_rate": round(compressed_rate, 4),
            },
            parity_achieved=parity_achieved,
            task_details=task_details,
            total_latency_ms=total_latency,
        )


def run_llm_benchmarks(
    baseline_corpus: str,
    compressed_corpus: str,
    endpoint: str = LLM_ENDPOINT,
    model: str = LLM_MODEL,
) -> LLMBenchmarkResults:
    """
    Convenience function to run LLM benchmark suite.

    Args:
        baseline_corpus: Full corpus text for baseline evaluation
        compressed_corpus: Compressed/retrieved corpus text
        endpoint: LLM API endpoint
        model: Model name

    Returns:
        LLMBenchmarkResults with parity check
    """
    runner = LLMBenchmarkRunner(endpoint=endpoint, model=model)
    runner.set_baseline_corpus(baseline_corpus)
    runner.set_compressed_corpus(compressed_corpus)
    return runner.run_suite()


def load_sample_corpus() -> Tuple[str, str]:
    """
    Load sample corpus from repository for testing.

    Corpus files selected to cover each benchmark task:
    - Task 1: GENESIS.md (bootstrap prompt)
    - Task 2: memory_record.py (data structure)
    - Task 3: cassette_receipt.py (integrity)
    - Task 4: memory_cassette.py (cartridge export/import)
    - Task 5: CORTEX/README.md (cassette network architecture)

    Returns:
        Tuple of (baseline_corpus, compressed_corpus)
    """
    # Baseline files - ordered by size (smaller first)
    # Each file maps to a specific benchmark task
    baseline_files = [
        # Task 1: Genesis bootstrap
        REPO_ROOT / "LAW" / "CANON" / "META" / "GENESIS.md",
        # Task 5: CORTEX architecture (small README)
        REPO_ROOT / "NAVIGATION" / "CORTEX" / "README.md",
        # Task 2: MemoryRecord structure
        REPO_ROOT / "CAPABILITY" / "PRIMITIVES" / "memory_record.py",
        # Task 3: Receipt chain integrity
        REPO_ROOT / "CAPABILITY" / "PRIMITIVES" / "cassette_receipt.py",
        # Task 4: Cartridge export/import (large file, loaded last)
        REPO_ROOT / "NAVIGATION" / "CORTEX" / "network" / "memory_cassette.py",
    ]

    baseline_parts = []
    for f in baseline_files:
        if f.exists():
            content = f.read_text(encoding="utf-8")
            baseline_parts.append(f"=== {f.name} ===\n{content}\n")

    baseline_corpus = "\n".join(baseline_parts)

    # Compressed corpus - first 4 files (excludes large memory_cassette.py)
    # Still tests compression: 4 small files vs 5 files including large one
    compressed_parts = baseline_parts[:4] if len(baseline_parts) >= 4 else baseline_parts
    compressed_corpus = "\n".join(compressed_parts)

    return baseline_corpus, compressed_corpus


def main() -> int:
    """Run LLM benchmarks standalone."""
    print(f"LLM Benchmark Suite v{BENCHMARK_VERSION}")
    print("=" * 60)
    print(f"Endpoint: {LLM_ENDPOINT}")
    print(f"Model: {LLM_MODEL}")
    print()

    # Check endpoint availability
    runner = LLMBenchmarkRunner()
    if not runner.check_endpoint():
        print("ERROR: LLM endpoint not available")
        print(f"Please ensure Nemotron is running at {LLM_ENDPOINT}")
        return 1

    print("Endpoint available!")
    print()

    # Load sample corpus
    print("Loading sample corpus...")
    baseline_corpus, compressed_corpus = load_sample_corpus()
    print(f"Baseline corpus: {len(baseline_corpus):,} chars")
    print(f"Compressed corpus: {len(compressed_corpus):,} chars")
    print()

    # Run benchmarks
    print("Running LLM benchmarks...")
    print("-" * 60)

    results = run_llm_benchmarks(
        baseline_corpus=baseline_corpus,
        compressed_corpus=compressed_corpus,
    )

    print()
    print(f"Tasks run: {results.tasks_run}")
    print(f"Total latency: {results.total_latency_ms:.0f}ms")
    print()
    print(f"Baseline results:")
    print(f"  Passed: {results.baseline_results['tasks_passed']}")
    print(f"  Failed: {results.baseline_results['tasks_failed']}")
    print(f"  Success rate: {results.baseline_results['success_rate']:.2%}")
    print()
    print(f"Compressed results:")
    print(f"  Passed: {results.compressed_results['tasks_passed']}")
    print(f"  Failed: {results.compressed_results['tasks_failed']}")
    print(f"  Success rate: {results.compressed_results['success_rate']:.2%}")
    print()
    print(f"Parity achieved: {results.parity_achieved}")

    print()
    print("Task Details:")
    print("-" * 60)
    for detail in results.task_details:
        status_b = "PASS" if detail["baseline_passed"] else "FAIL"
        status_c = "PASS" if detail["compressed_passed"] else "FAIL"
        print(f"  {detail['task_id']}: baseline={status_b}, compressed={status_c}")
        print(f"    Question: {detail['question'][:50]}...")
        print(f"    Baseline matched: {detail['baseline_matched']}")
        print(f"    Compressed matched: {detail['compressed_matched']}")
        print()

    # Save results
    output_path = REPO_ROOT / "NAVIGATION" / "PROOFS" / "COMPRESSION" / "LLM_BENCHMARK_RESULTS.json"
    output_path.write_text(json.dumps(results.to_dict(), indent=2), encoding="utf-8")
    print(f"Results saved to: {output_path}")

    if results.parity_achieved:
        print("\nSUCCESS: Compression maintains LLM task success parity")
        return 0
    else:
        print("\nWARNING: Compression may degrade LLM performance")
        return 1


if __name__ == "__main__":
    sys.exit(main())
