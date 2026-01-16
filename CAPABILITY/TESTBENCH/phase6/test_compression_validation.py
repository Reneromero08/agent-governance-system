#!/usr/bin/env python3
"""
Phase 6.4 Tests: Compression Validation

Tests for benchmark tasks, corpus specs, and proof runners.
"""

import json
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "NAVIGATION" / "PROOFS" / "COMPRESSION"))
sys.path.insert(0, str(PROJECT_ROOT / "NAVIGATION" / "PROOFS" / "CATALYTIC"))
sys.path.insert(0, str(PROJECT_ROOT / "NAVIGATION" / "PROOFS"))

import pytest


class TestBenchmarkTasks:
    """Test benchmark task framework."""

    def test_import_benchmark_module(self):
        """Test that benchmark module imports correctly."""
        from benchmark_tasks import (
            BENCHMARK_VERSION,
            BenchmarkRunner,
            BenchmarkTask,
            TaskType,
            get_all_benchmark_tasks,
        )

        assert BENCHMARK_VERSION is not None
        assert BenchmarkRunner is not None

    def test_get_all_benchmark_tasks(self):
        """Test that benchmark tasks are defined."""
        from benchmark_tasks import get_all_benchmark_tasks, TaskType

        tasks = get_all_benchmark_tasks()

        assert len(tasks) > 0

        # Check task types are valid
        for task in tasks:
            assert isinstance(task.task_type, TaskType)
            assert task.task_id is not None
            assert task.description is not None

    def test_semantic_match_tasks(self):
        """Test semantic match tasks."""
        from benchmark_tasks import get_semantic_match_tasks, TaskType

        tasks = get_semantic_match_tasks()

        assert len(tasks) >= 3
        for task in tasks:
            assert task.task_type == TaskType.SEMANTIC_MATCH
            assert task.expected_answer is not None

    def test_benchmark_runner_creation(self):
        """Test benchmark runner can be created."""
        from benchmark_tasks import BenchmarkRunner

        runner = BenchmarkRunner()
        assert runner is not None

    def test_benchmark_runner_with_empty_corpus(self):
        """Test benchmark runner with empty corpus."""
        from benchmark_tasks import BenchmarkRunner, get_semantic_match_tasks

        runner = BenchmarkRunner()
        runner.set_baseline_corpus("")

        tasks = get_semantic_match_tasks()[:2]
        results = runner.run_suite(tasks)

        assert results.tasks_run == len(tasks)
        assert results.benchmark_version is not None

    def test_task_result_serialization(self):
        """Test that task results serialize to dict."""
        from benchmark_tasks import TaskResult

        result = TaskResult(
            task_id="test_001",
            task_type="semantic_match",
            passed=True,
            tokens_used=100,
            context_type="baseline",
        )

        d = result.to_dict()
        assert d["task_id"] == "test_001"
        assert d["passed"] is True
        assert d["tokens_used"] == 100


class TestCorpusSpec:
    """Test corpus specification module."""

    def test_import_corpus_spec(self):
        """Test that corpus spec module imports correctly."""
        from corpus_spec import (
            BaselineCorpusSpec,
            CompressedContextSpec,
            CorpusAnchor,
            ProofCorpusSpec,
            get_default_spec,
        )

        assert BaselineCorpusSpec is not None
        assert CompressedContextSpec is not None

    def test_baseline_corpus_spec_defaults(self):
        """Test baseline corpus spec has sensible defaults."""
        from corpus_spec import BaselineCorpusSpec

        spec = BaselineCorpusSpec()

        assert spec.aggregation_mode == "sum_per_file"
        assert len(spec.include_patterns) > 0
        assert "LAW/**/*.md" in spec.include_patterns

    def test_compressed_context_spec_defaults(self):
        """Test compressed context spec has sensible defaults."""
        from corpus_spec import CompressedContextSpec

        spec = CompressedContextSpec()

        assert spec.retrieval_method == "semantic"
        assert spec.top_k == 10
        assert spec.min_similarity == 0.4
        assert spec.tie_break_order == "similarity_desc_hash_asc"

    def test_corpus_spec_serialization(self):
        """Test corpus spec serializes to dict."""
        from corpus_spec import ProofCorpusSpec

        spec = ProofCorpusSpec()
        d = spec.to_dict()

        assert "spec_version" in d
        assert "baseline" in d
        assert "compressed" in d

    def test_get_default_spec(self):
        """Test get_default_spec returns valid spec."""
        from corpus_spec import get_default_spec

        spec = get_default_spec()

        assert spec is not None
        assert spec.spec_version == "1.0.0"

    def test_tie_breaking_determinism(self):
        """Test that tie-breaking produces deterministic results."""
        from corpus_spec import CompressedContextSpec

        spec = CompressedContextSpec()

        results = [
            {"similarity": 0.8, "hash": "ccc"},
            {"similarity": 0.8, "hash": "aaa"},
            {"similarity": 0.9, "hash": "bbb"},
            {"similarity": 0.8, "hash": "bbb"},
        ]

        sorted_results = spec.apply_tie_breaking(results)

        # Should be sorted by similarity DESC, then hash ASC
        assert sorted_results[0]["hash"] == "bbb"  # 0.9, only one
        assert sorted_results[1]["hash"] == "aaa"  # 0.8, lowest hash
        assert sorted_results[2]["hash"] == "bbb"  # 0.8, middle hash
        assert sorted_results[3]["hash"] == "ccc"  # 0.8, highest hash


class TestCompressionClaimSchema:
    """Test compression claim schema updates."""

    def test_schema_has_task_performance(self):
        """Test that schema includes task_performance field."""
        schema_path = (
            PROJECT_ROOT
            / "THOUGHT"
            / "LAB"
            / "CAT_CHAT"
            / "SCHEMAS"
            / "compression_claim.schema.json"
        )

        if not schema_path.exists():
            pytest.skip("Schema file not found")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        assert "task_performance" in schema["properties"]

        tp = schema["properties"]["task_performance"]
        assert tp["type"] == "object"
        assert "benchmark_version" in tp["properties"]
        assert "tasks_run" in tp["properties"]
        assert "baseline_results" in tp["properties"]
        assert "compressed_results" in tp["properties"]
        assert "parity_achieved" in tp["properties"]

    def test_task_details_schema(self):
        """Test task_details array schema."""
        schema_path = (
            PROJECT_ROOT
            / "THOUGHT"
            / "LAB"
            / "CAT_CHAT"
            / "SCHEMAS"
            / "compression_claim.schema.json"
        )

        if not schema_path.exists():
            pytest.skip("Schema file not found")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        tp = schema["properties"]["task_performance"]
        task_details = tp["properties"]["task_details"]

        assert task_details["type"] == "array"
        item_props = task_details["items"]["properties"]
        assert "task_id" in item_props
        assert "task_type" in item_props
        assert "baseline_passed" in item_props
        assert "compressed_passed" in item_props


class TestProofRunnerIntegration:
    """Test proof runner integration (import only, no actual proof runs)."""

    def test_import_proof_runner(self):
        """Test that proof runner imports correctly."""
        from proof_runner import (
            ProofRunner,
            get_proof_artifacts,
            run_proofs_for_pack,
        )

        assert ProofRunner is not None
        assert run_proofs_for_pack is not None
        assert get_proof_artifacts is not None

    def test_proof_runner_creation(self):
        """Test proof runner can be created."""
        from proof_runner import ProofRunner

        runner = ProofRunner(pack_id="test-pack", bundle_id="a" * 64)

        assert runner.pack_id == "test-pack"
        assert runner.bundle_id == "a" * 64
        assert runner.timestamp is not None

    def test_get_proof_artifacts_returns_list(self):
        """Test get_proof_artifacts returns a list."""
        from proof_runner import get_proof_artifacts

        artifacts = get_proof_artifacts()

        assert isinstance(artifacts, list)
        # Artifacts may or may not exist depending on whether proofs have been run


class TestDeterminism:
    """Test determinism guarantees for proofs."""

    def test_token_counting_determinism(self):
        """Test that token counting is deterministic."""
        from benchmark_tasks import _count_tokens_proxy

        text = "This is a deterministic test string for token counting."

        count1 = _count_tokens_proxy(text)
        count2 = _count_tokens_proxy(text)

        assert count1 == count2

    def test_corpus_anchor_hash_determinism(self):
        """Test that corpus anchor hash is deterministic."""
        from corpus_spec import _sha256_hex

        text = "Deterministic content for hashing"

        hash1 = _sha256_hex(text)
        hash2 = _sha256_hex(text)

        assert hash1 == hash2
        assert len(hash1) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
