"""
Tests for Phase I.1: Metrics Collector
======================================

Tests for StepMetrics, TurnMetrics, SessionMetrics, and MetricsCollector.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from catalytic_chat.metrics_collector import (
    StepMetrics,
    TurnMetrics,
    SessionMetrics,
    MetricsCollector,
    create_metrics_collector,
    compute_content_hash,
)


# =============================================================================
# StepMetrics Tests
# =============================================================================

class TestStepMetrics:
    """Tests for StepMetrics dataclass."""

    def test_step_metrics_creation(self):
        """Test basic StepMetrics creation."""
        step = StepMetrics(
            step_name="spc_resolve",
            start_time_ns=1000000,
            end_time_ns=2000000,
            bytes_in=100,
            bytes_out=50,
            cache_hit=False,
            source="spc",
            success=True,
            metadata={"symbol": "C"},
        )

        assert step.step_name == "spc_resolve"
        assert step.source == "spc"
        assert step.success is True

    def test_step_metrics_latency(self):
        """Test latency calculation."""
        step = StepMetrics(
            step_name="test",
            start_time_ns=0,
            end_time_ns=5_000_000,  # 5ms
            bytes_in=0,
            bytes_out=0,
            cache_hit=False,
            source="test",
            success=True,
        )

        assert step.latency_ms == 5.0

    def test_step_metrics_compression_ratio(self):
        """Test compression ratio calculation."""
        step = StepMetrics(
            step_name="compress",
            start_time_ns=0,
            end_time_ns=1000,
            bytes_in=1000,
            bytes_out=100,
            cache_hit=False,
            source="turn_compressor",
            success=True,
        )

        assert step.compression_ratio == 10.0

    def test_step_metrics_zero_output(self):
        """Test compression ratio with zero output."""
        step = StepMetrics(
            step_name="fail",
            start_time_ns=0,
            end_time_ns=1000,
            bytes_in=100,
            bytes_out=0,
            cache_hit=False,
            source="test",
            success=False,
        )

        assert step.compression_ratio == 0.0

    def test_step_metrics_to_dict(self):
        """Test serialization to dictionary."""
        step = StepMetrics(
            step_name="spc_resolve",
            start_time_ns=0,
            end_time_ns=1_000_000,
            bytes_in=100,
            bytes_out=50,
            cache_hit=True,
            source="spc",
            success=True,
            metadata={"pointer": "C3"},
        )

        d = step.to_dict()

        assert d["step_name"] == "spc_resolve"
        assert d["latency_ms"] == 1.0
        assert d["bytes_in"] == 100
        assert d["bytes_out"] == 50
        assert d["cache_hit"] is True
        assert d["source"] == "spc"
        assert d["success"] is True
        assert d["compression_ratio"] == 2.0
        assert d["metadata"] == {"pointer": "C3"}


# =============================================================================
# TurnMetrics Tests
# =============================================================================

class TestTurnMetrics:
    """Tests for TurnMetrics dataclass."""

    def test_turn_metrics_creation(self):
        """Test basic TurnMetrics creation."""
        turn = TurnMetrics(
            turn_index=1,
            timestamp="2026-01-19T12:00:00Z",
        )

        assert turn.turn_index == 1
        assert turn.total_bytes_expanded == 0
        assert turn.cache_hit_rate == 0.0

    def test_turn_metrics_add_step(self):
        """Test adding steps to turn metrics."""
        turn = TurnMetrics(turn_index=1, timestamp="2026-01-19T12:00:00Z")

        step1 = StepMetrics(
            step_name="spc_resolve",
            start_time_ns=0,
            end_time_ns=1_000_000,
            bytes_in=100,
            bytes_out=50,
            cache_hit=True,
            source="spc",
            success=True,
        )

        step2 = StepMetrics(
            step_name="turn_compress",
            start_time_ns=1_000_000,
            end_time_ns=3_000_000,
            bytes_in=500,
            bytes_out=100,
            cache_hit=False,
            source="turn_compressor",
            success=True,
        )

        turn.add_step(step1)
        turn.add_step(step2)

        assert len(turn.steps) == 2
        assert turn.total_bytes_expanded == 600  # 100 + 500
        assert turn.total_bytes_compressed == 150  # 50 + 100
        assert turn.cache_hits == 1
        assert turn.cache_misses == 1
        assert turn.spc_hits == 1

    def test_turn_metrics_total_latency(self):
        """Test total latency calculation."""
        turn = TurnMetrics(turn_index=1, timestamp="2026-01-19T12:00:00Z")

        for i in range(3):
            step = StepMetrics(
                step_name=f"step_{i}",
                start_time_ns=0,
                end_time_ns=1_000_000,  # 1ms each
                bytes_in=0,
                bytes_out=0,
                cache_hit=False,
                source="test",
                success=True,
            )
            turn.add_step(step)

        assert turn.total_latency_ms == 3.0

    def test_turn_metrics_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        turn = TurnMetrics(turn_index=1, timestamp="2026-01-19T12:00:00Z")

        # Add 3 hits, 1 miss
        for i, hit in enumerate([True, True, True, False]):
            step = StepMetrics(
                step_name=f"step_{i}",
                start_time_ns=0,
                end_time_ns=1000,
                bytes_in=0,
                bytes_out=0,
                cache_hit=hit,
                source="test",
                success=True,
            )
            turn.add_step(step)

        assert turn.cache_hit_rate == 0.75

    def test_turn_metrics_resolution_chain_tracking(self):
        """Test resolution chain source tracking."""
        turn = TurnMetrics(turn_index=1, timestamp="2026-01-19T12:00:00Z")

        sources = ["spc", "cassette", "spc", "vector_fallback", "cas"]

        for i, source in enumerate(sources):
            step = StepMetrics(
                step_name=f"step_{i}",
                start_time_ns=0,
                end_time_ns=1000,
                bytes_in=0,
                bytes_out=0,
                cache_hit=False,
                source=source,
                success=True,
            )
            turn.add_step(step)

        assert turn.spc_hits == 2
        assert turn.cassette_hits == 1
        assert turn.vector_fallback_hits == 1
        assert turn.cas_hits == 1

    def test_turn_metrics_to_dict(self):
        """Test serialization to dictionary."""
        turn = TurnMetrics(turn_index=5, timestamp="2026-01-19T12:00:00Z")
        turn.e_score_mean = 0.75
        turn.tokens_in_context = 5000

        d = turn.to_dict()

        assert d["turn_index"] == 5
        assert d["e_score_mean"] == 0.75
        assert d["tokens_in_context"] == 5000
        assert "steps" in d
        assert "resolution_chain" in d


# =============================================================================
# SessionMetrics Tests
# =============================================================================

class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_session_metrics_creation(self):
        """Test basic SessionMetrics creation."""
        session = SessionMetrics(
            session_id="test-session-001",
            started_at="2026-01-19T12:00:00Z",
        )

        assert session.session_id == "test-session-001"
        assert session.total_turns == 0

    def test_session_metrics_add_turn(self):
        """Test adding turns to session metrics."""
        session = SessionMetrics(
            session_id="test-session",
            started_at="2026-01-19T12:00:00Z",
        )

        turn1 = TurnMetrics(turn_index=1, timestamp="2026-01-19T12:00:00Z")
        turn1.total_bytes_expanded = 1000
        turn1.total_bytes_compressed = 200
        turn1.cache_hits = 3
        turn1.cache_misses = 1
        turn1.spc_hits = 2
        turn1.e_score_mean = 0.8

        turn2 = TurnMetrics(turn_index=2, timestamp="2026-01-19T12:01:00Z")
        turn2.total_bytes_expanded = 2000
        turn2.total_bytes_compressed = 400
        turn2.cache_hits = 5
        turn2.cache_misses = 2
        turn2.cassette_hits = 3
        turn2.e_score_mean = 0.6

        session.add_turn(turn1)
        session.add_turn(turn2)

        assert session.total_turns == 2
        assert session.total_bytes_expanded == 3000
        assert session.total_bytes_compressed == 600
        assert session.total_cache_hits == 8
        assert session.total_cache_misses == 3
        assert session.total_spc_hits == 2
        assert session.total_cassette_hits == 3

    def test_session_metrics_compression_ratio(self):
        """Test overall compression ratio."""
        session = SessionMetrics(
            session_id="test-session",
            started_at="2026-01-19T12:00:00Z",
        )

        session.total_bytes_expanded = 10000
        session.total_bytes_compressed = 1000

        assert session.overall_compression_ratio == 10.0

    def test_session_metrics_mean_e_score(self):
        """Test mean E-score calculation."""
        session = SessionMetrics(
            session_id="test-session",
            started_at="2026-01-19T12:00:00Z",
        )

        session.e_score_samples = [0.5, 0.7, 0.8, 0.6]

        assert session.mean_e_score == 0.65

    def test_session_metrics_invariant_tracking(self):
        """Test invariant check tracking."""
        session = SessionMetrics(
            session_id="test-session",
            started_at="2026-01-19T12:00:00Z",
        )

        session.record_invariant_check("INV-CATALYTIC-01", True)
        session.record_invariant_check("INV-CATALYTIC-04", False, {"reason": "budget exceeded"})

        assert session.invariant_checks["INV-CATALYTIC-01"] is True
        assert session.invariant_checks["INV-CATALYTIC-04"] is False
        assert len(session.invariant_violations) == 1
        assert session.invariant_violations[0]["invariant_id"] == "INV-CATALYTIC-04"

    def test_session_metrics_e_score_histogram(self):
        """Test E-score histogram computation."""
        session = SessionMetrics(
            session_id="test-session",
            started_at="2026-01-19T12:00:00Z",
        )

        # Add samples across different bins
        session.e_score_samples = [0.1, 0.15, 0.3, 0.5, 0.55, 0.7, 0.75, 0.9, 0.95]

        histogram = session._compute_e_score_histogram()

        assert histogram["0.0-0.2"] == 2  # 0.1, 0.15
        assert histogram["0.2-0.4"] == 1  # 0.3
        assert histogram["0.4-0.6"] == 2  # 0.5, 0.55
        assert histogram["0.6-0.8"] == 2  # 0.7, 0.75
        assert histogram["0.8-1.0"] == 2  # 0.9, 0.95

    def test_session_metrics_to_dict(self):
        """Test serialization to dictionary."""
        session = SessionMetrics(
            session_id="test-session-001",
            started_at="2026-01-19T12:00:00Z",
        )

        session.total_bytes_expanded = 5000
        session.total_bytes_compressed = 1000
        session.total_cache_hits = 10
        session.record_invariant_check("INV-01", True)

        d = session.to_dict()

        assert d["session_id"] == "test-session-001"
        assert d["overall_compression_ratio"] == 5.0
        assert d["invariant_checks"]["INV-01"] is True
        assert "e_score_histogram" in d


# =============================================================================
# MetricsCollector Tests
# =============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_collector_creation(self):
        """Test basic collector creation."""
        collector = MetricsCollector(session_id="test-session")

        assert collector.session_id == "test-session"
        assert collector.get_session_metrics().total_turns == 0

    def test_collector_start_end_turn(self):
        """Test turn lifecycle."""
        collector = MetricsCollector(session_id="test-session")

        turn = collector.start_turn()
        assert turn.turn_index == 1

        completed = collector.end_turn(
            e_mean=0.75,
            tokens_context=5000,
            tokens_pointer_set=2000
        )

        assert completed.e_score_mean == 0.75
        assert completed.tokens_in_context == 5000
        assert collector.get_session_metrics().total_turns == 1

    def test_collector_measure_step_context_manager(self):
        """Test measure_step context manager."""
        collector = MetricsCollector(session_id="test-session")
        collector.start_turn()

        with collector.measure_step("spc_resolve", source="spc") as step:
            # Simulate work
            time.sleep(0.001)
            step.set_bytes(100, 50)
            step.set_cache_hit(True)
            step.add_metadata("pointer", "C3")

        turn = collector.get_current_turn()

        assert len(turn.steps) == 1
        assert turn.steps[0].step_name == "spc_resolve"
        assert turn.steps[0].bytes_in == 100
        assert turn.steps[0].bytes_out == 50
        assert turn.steps[0].cache_hit is True
        assert turn.steps[0].source == "spc"
        assert turn.steps[0].metadata["pointer"] == "C3"
        assert turn.steps[0].latency_ms > 0

    def test_collector_measure_step_failure(self):
        """Test measure_step with exception."""
        collector = MetricsCollector(session_id="test-session")
        collector.start_turn()

        with pytest.raises(ValueError):
            with collector.measure_step("failing_step", source="test") as step:
                step.set_bytes(100, 0)
                raise ValueError("Simulated failure")

        turn = collector.get_current_turn()

        # Step should still be recorded
        assert len(turn.steps) == 1
        assert turn.steps[0].success is False

    def test_collector_multiple_turns(self):
        """Test multiple turn workflow."""
        collector = MetricsCollector(session_id="test-session")

        for i in range(5):
            collector.start_turn()

            with collector.measure_step(f"step_{i}", source="spc") as step:
                step.set_bytes(100 * (i + 1), 50 * (i + 1))

            collector.end_turn(e_mean=0.5 + i * 0.1)

        session = collector.get_session_metrics()

        assert session.total_turns == 5
        assert len(session.e_score_samples) == 5

    def test_collector_invariant_recording(self):
        """Test invariant check recording."""
        collector = MetricsCollector(session_id="test-session")

        collector.record_invariant_check("INV-CATALYTIC-01", True)
        collector.record_invariant_check("INV-CATALYTIC-04", False, {"budget_used": 15000})

        session = collector.get_session_metrics()

        assert session.invariant_checks["INV-CATALYTIC-01"] is True
        assert session.invariant_checks["INV-CATALYTIC-04"] is False
        assert len(session.invariant_violations) == 1

    def test_collector_export_to_json(self):
        """Test exporting metrics to JSON file."""
        collector = MetricsCollector(session_id="test-session")

        collector.start_turn()
        with collector.measure_step("test_step", source="test") as step:
            step.set_bytes(100, 50)
        collector.end_turn(e_mean=0.7)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            collector.export_to_json(output_path)

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert data["session_id"] == "test-session"
            assert data["total_turns"] == 1

    def test_collector_get_summary(self):
        """Test getting summary dictionary."""
        collector = MetricsCollector(session_id="test-session")

        collector.start_turn()
        collector.end_turn(e_mean=0.5)

        summary = collector.get_summary()

        assert isinstance(summary, dict)
        assert summary["session_id"] == "test-session"
        assert summary["total_turns"] == 1

    def test_collector_end_turn_without_start_raises(self):
        """Test that ending turn without starting raises error."""
        collector = MetricsCollector(session_id="test-session")

        with pytest.raises(RuntimeError, match="No turn in progress"):
            collector.end_turn()


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_metrics_collector(self):
        """Test factory function."""
        collector = create_metrics_collector("my-session")

        assert isinstance(collector, MetricsCollector)
        assert collector.session_id == "my-session"

    def test_compute_content_hash(self):
        """Test content hash computation."""
        content = "Hello, world!"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_compute_content_hash_deterministic(self):
        """Test that hash is deterministic."""
        content = "The quick brown fox"

        hashes = [compute_content_hash(content) for _ in range(100)]

        assert all(h == hashes[0] for h in hashes)


# =============================================================================
# Integration Tests
# =============================================================================

class TestMetricsIntegration:
    """Integration tests for metrics collection workflow."""

    def test_full_session_workflow(self):
        """Test complete session metrics workflow."""
        collector = MetricsCollector(session_id="integration-test")

        # Simulate 10 turns
        for turn_num in range(1, 11):
            collector.start_turn()

            # Simulate resolution chain
            sources = ["spc", "cassette", "cas"]
            for i, source in enumerate(sources):
                with collector.measure_step(f"resolve_{i}", source=source) as step:
                    step.set_bytes(100 + turn_num * 10, 50 + turn_num * 5)
                    step.set_cache_hit(i % 2 == 0)

            # Simulate turn compression
            with collector.measure_step("turn_compress", source="turn_compressor") as step:
                step.set_bytes(500, 50)

            collector.end_turn(
                e_mean=0.5 + turn_num * 0.04,
                tokens_context=5000 + turn_num * 100,
                tokens_pointer_set=2000 + turn_num * 50,
            )

        # Record invariant checks
        collector.record_invariant_check("INV-CATALYTIC-01", True)
        collector.record_invariant_check("INV-CATALYTIC-04", True)
        collector.record_invariant_check("INV-CATALYTIC-07", True)

        session = collector.get_session_metrics()

        # Verify session metrics
        assert session.total_turns == 10
        assert session.total_resolutions > 0
        assert session.cache_hit_rate > 0
        assert session.overall_compression_ratio > 1  # Should compress

        # Verify invariant tracking
        assert len(session.invariant_checks) == 3
        assert all(session.invariant_checks.values())

        # Verify E-score samples
        assert len(session.e_score_samples) == 10
        assert 0.5 <= session.mean_e_score <= 1.0

    def test_metrics_json_roundtrip(self):
        """Test that metrics survive JSON serialization."""
        collector = MetricsCollector(session_id="json-test")

        collector.start_turn()
        with collector.measure_step("step1", source="spc") as step:
            step.set_bytes(100, 25)
        collector.end_turn(e_mean=0.8)

        # Serialize to JSON
        summary = collector.get_summary()
        json_str = json.dumps(summary)

        # Deserialize
        parsed = json.loads(json_str)

        assert parsed["session_id"] == "json-test"
        assert parsed["total_turns"] == 1
        assert parsed["overall_compression_ratio"] == 4.0  # 100/25
