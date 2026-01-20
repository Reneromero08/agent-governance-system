"""
Metrics Collector - Phase I.1
=============================

Unified telemetry for CAT Chat operations.

Captures:
- Per-step metrics (resolution, compression, partitioning)
- Per-turn aggregations
- Per-session summaries
- Cache hit rates and timing

All metrics are logged to session_events for deterministic replay and audit.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager


# =============================================================================
# Step-Level Metrics
# =============================================================================

@dataclass
class StepMetrics:
    """
    Metrics for a single operation step.

    Captures timing, byte sizes, cache status, and source for each
    discrete operation in the resolution/compression chain.
    """
    step_name: str              # e.g., "spc_resolve", "cassette_fts", "turn_compress"
    start_time_ns: int          # Nanosecond precision start
    end_time_ns: int            # Nanosecond precision end
    bytes_in: int               # Input size in bytes
    bytes_out: int              # Output size in bytes
    cache_hit: bool             # Whether result was cached
    source: str                 # Where data came from (spc, cassette, cas, vector, etc.)
    success: bool               # Whether step succeeded
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return (self.end_time_ns - self.start_time_ns) / 1_000_000

    @property
    def compression_ratio(self) -> float:
        """Ratio of input to output (> 1 means compression)."""
        if self.bytes_out == 0:
            return 0.0
        return self.bytes_in / self.bytes_out

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "step_name": self.step_name,
            "latency_ms": round(self.latency_ms, 3),
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "cache_hit": self.cache_hit,
            "source": self.source,
            "success": self.success,
            "compression_ratio": round(self.compression_ratio, 2) if self.bytes_out > 0 else None,
            "metadata": self.metadata,
        }


# =============================================================================
# Turn-Level Metrics
# =============================================================================

@dataclass
class TurnMetrics:
    """
    Aggregated metrics for one conversation turn.

    A turn = user query + assistant response + all context operations.
    """
    turn_index: int
    timestamp: str
    steps: List[StepMetrics] = field(default_factory=list)

    # Byte metrics
    total_bytes_expanded: int = 0     # Total bytes of expanded content
    total_bytes_compressed: int = 0   # Total bytes after compression

    # Token metrics (estimated)
    tokens_in_context: int = 0        # Working set tokens
    tokens_in_pointer_set: int = 0    # Pointer set tokens

    # Quality metrics
    e_score_mean: float = 0.0         # Mean E-score of working set
    e_score_min: float = 0.0          # Min E-score in working set
    e_score_max: float = 0.0          # Max E-score in working set

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    # Resolution chain metrics
    spc_hits: int = 0
    cassette_hits: int = 0
    local_index_hits: int = 0
    docs_index_hits: int = 0
    cas_hits: int = 0
    vector_fallback_hits: int = 0

    @property
    def total_latency_ms(self) -> float:
        """Total latency across all steps."""
        return sum(s.latency_ms for s in self.steps)

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio for this turn."""
        if self.total_bytes_compressed == 0:
            return 0.0
        return self.total_bytes_expanded / self.total_bytes_compressed

    def add_step(self, step: StepMetrics) -> None:
        """Add a step and update aggregates."""
        self.steps.append(step)

        # Update byte totals
        self.total_bytes_expanded += step.bytes_in
        self.total_bytes_compressed += step.bytes_out

        # Update cache counters
        if step.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Update source counters
        if step.source == "spc":
            self.spc_hits += 1
        elif step.source == "cassette":
            self.cassette_hits += 1
        elif step.source == "local_index" or step.source == "symbol_registry":
            self.local_index_hits += 1
        elif step.source == "docs_index":
            self.docs_index_hits += 1
        elif step.source == "cas":
            self.cas_hits += 1
        elif step.source == "vector_fallback":
            self.vector_fallback_hits += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "total_bytes_expanded": self.total_bytes_expanded,
            "total_bytes_compressed": self.total_bytes_compressed,
            "compression_ratio": round(self.compression_ratio, 2) if self.total_bytes_compressed > 0 else None,
            "tokens_in_context": self.tokens_in_context,
            "tokens_in_pointer_set": self.tokens_in_pointer_set,
            "e_score_mean": round(self.e_score_mean, 4),
            "e_score_min": round(self.e_score_min, 4),
            "e_score_max": round(self.e_score_max, 4),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "resolution_chain": {
                "spc_hits": self.spc_hits,
                "cassette_hits": self.cassette_hits,
                "local_index_hits": self.local_index_hits,
                "docs_index_hits": self.docs_index_hits,
                "cas_hits": self.cas_hits,
                "vector_fallback_hits": self.vector_fallback_hits,
            },
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


# =============================================================================
# Session-Level Metrics
# =============================================================================

@dataclass
class SessionMetrics:
    """
    Aggregated metrics for an entire session.

    Tracks cumulative statistics across all turns.
    """
    session_id: str
    started_at: str
    turns: List[TurnMetrics] = field(default_factory=list)

    # Cumulative byte metrics
    total_bytes_expanded: int = 0
    total_bytes_compressed: int = 0

    # Cumulative cache metrics
    total_cache_hits: int = 0
    total_cache_misses: int = 0

    # Cumulative resolution chain metrics
    total_spc_hits: int = 0
    total_cassette_hits: int = 0
    total_local_index_hits: int = 0
    total_docs_index_hits: int = 0
    total_cas_hits: int = 0
    total_vector_fallback_hits: int = 0

    # E-score tracking
    e_score_samples: List[float] = field(default_factory=list)

    # Invariant tracking
    invariant_checks: Dict[str, bool] = field(default_factory=dict)
    invariant_violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_turns(self) -> int:
        """Number of turns in session."""
        return len(self.turns)

    @property
    def overall_compression_ratio(self) -> float:
        """Overall compression ratio for entire session."""
        if self.total_bytes_compressed == 0:
            return 0.0
        return self.total_bytes_expanded / self.total_bytes_compressed

    @property
    def cache_hit_rate(self) -> float:
        """Overall cache hit rate."""
        total = self.total_cache_hits + self.total_cache_misses
        return self.total_cache_hits / total if total > 0 else 0.0

    @property
    def total_latency_ms(self) -> float:
        """Total latency across all turns."""
        return sum(t.total_latency_ms for t in self.turns)

    @property
    def mean_e_score(self) -> float:
        """Mean E-score across all samples."""
        if not self.e_score_samples:
            return 0.0
        return sum(self.e_score_samples) / len(self.e_score_samples)

    @property
    def total_resolutions(self) -> int:
        """Total resolution attempts."""
        return (
            self.total_spc_hits +
            self.total_cassette_hits +
            self.total_local_index_hits +
            self.total_docs_index_hits +
            self.total_cas_hits +
            self.total_vector_fallback_hits
        )

    def add_turn(self, turn: TurnMetrics) -> None:
        """Add a turn and update cumulative metrics."""
        self.turns.append(turn)

        # Update byte totals
        self.total_bytes_expanded += turn.total_bytes_expanded
        self.total_bytes_compressed += turn.total_bytes_compressed

        # Update cache totals
        self.total_cache_hits += turn.cache_hits
        self.total_cache_misses += turn.cache_misses

        # Update resolution chain totals
        self.total_spc_hits += turn.spc_hits
        self.total_cassette_hits += turn.cassette_hits
        self.total_local_index_hits += turn.local_index_hits
        self.total_docs_index_hits += turn.docs_index_hits
        self.total_cas_hits += turn.cas_hits
        self.total_vector_fallback_hits += turn.vector_fallback_hits

        # Collect E-score samples
        if turn.e_score_mean > 0:
            self.e_score_samples.append(turn.e_score_mean)

    def record_invariant_check(
        self,
        invariant_id: str,
        passed: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an invariant check result."""
        self.invariant_checks[invariant_id] = passed
        if not passed:
            self.invariant_violations.append({
                "invariant_id": invariant_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": details or {},
            })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "total_turns": self.total_turns,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "total_bytes_expanded": self.total_bytes_expanded,
            "total_bytes_compressed": self.total_bytes_compressed,
            "overall_compression_ratio": round(self.overall_compression_ratio, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "total_cache_hits": self.total_cache_hits,
            "total_cache_misses": self.total_cache_misses,
            "mean_e_score": round(self.mean_e_score, 4),
            "resolution_chain": {
                "total_resolutions": self.total_resolutions,
                "spc_hits": self.total_spc_hits,
                "cassette_hits": self.total_cassette_hits,
                "local_index_hits": self.total_local_index_hits,
                "docs_index_hits": self.total_docs_index_hits,
                "cas_hits": self.total_cas_hits,
                "vector_fallback_hits": self.total_vector_fallback_hits,
            },
            "invariant_checks": self.invariant_checks,
            "invariant_violations": self.invariant_violations,
            "e_score_histogram": self._compute_e_score_histogram(),
        }

    def _compute_e_score_histogram(self) -> Dict[str, int]:
        """Compute E-score distribution histogram."""
        if not self.e_score_samples:
            return {}

        bins = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for score in self.e_score_samples:
            if score < 0.2:
                bins["0.0-0.2"] += 1
            elif score < 0.4:
                bins["0.2-0.4"] += 1
            elif score < 0.6:
                bins["0.4-0.6"] += 1
            elif score < 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1

        return bins


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Central metrics collection for CAT Chat.

    Usage:
        collector = MetricsCollector(session_id="session_123")

        # Start a turn
        turn = collector.start_turn()

        # Record steps
        with collector.measure_step("spc_resolve", source="spc") as step:
            result = spc_bridge.resolve(pointer)
            step.set_bytes(len(pointer), len(result))
            step.set_cache_hit(False)

        # End turn with E-scores
        collector.end_turn(e_mean=0.7, tokens_context=5000)

        # Get session summary
        summary = collector.get_session_metrics()
    """

    def __init__(
        self,
        session_id: str,
        token_estimator: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize metrics collector.

        Args:
            session_id: Session ID for tracking
            token_estimator: Function to estimate tokens from text (default: len//4)
        """
        self.session_id = session_id
        self.token_estimator = token_estimator or (lambda s: len(s) // 4)

        self._session = SessionMetrics(
            session_id=session_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        self._current_turn: Optional[TurnMetrics] = None
        self._turn_index = 0

    def start_turn(self) -> TurnMetrics:
        """
        Start a new turn for metric collection.

        Returns:
            TurnMetrics instance for this turn
        """
        self._turn_index += 1
        self._current_turn = TurnMetrics(
            turn_index=self._turn_index,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return self._current_turn

    def end_turn(
        self,
        e_mean: float = 0.0,
        e_min: float = 0.0,
        e_max: float = 0.0,
        tokens_context: int = 0,
        tokens_pointer_set: int = 0
    ) -> TurnMetrics:
        """
        End the current turn and record to session.

        Args:
            e_mean: Mean E-score of working set
            e_min: Min E-score in working set
            e_max: Max E-score in working set
            tokens_context: Tokens in working set
            tokens_pointer_set: Tokens in pointer set

        Returns:
            Completed TurnMetrics
        """
        if self._current_turn is None:
            raise RuntimeError("No turn in progress - call start_turn() first")

        turn = self._current_turn

        # Record E-scores
        turn.e_score_mean = e_mean
        turn.e_score_min = e_min
        turn.e_score_max = e_max

        # Record token counts
        turn.tokens_in_context = tokens_context
        turn.tokens_in_pointer_set = tokens_pointer_set

        # Add to session
        self._session.add_turn(turn)

        self._current_turn = None
        return turn

    @contextmanager
    def measure_step(
        self,
        step_name: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager to measure a single operation step.

        Usage:
            with collector.measure_step("spc_resolve", source="spc") as step:
                result = spc_bridge.resolve(pointer)
                step.set_bytes(len(pointer), len(result))
                step.set_cache_hit(False)
        """
        step = _StepBuilder(step_name, source, metadata or {})
        step._start_time_ns = time.perf_counter_ns()

        try:
            yield step
            step._success = True
        except Exception:
            step._success = False
            raise
        finally:
            step._end_time_ns = time.perf_counter_ns()
            metrics = step.build()

            if self._current_turn:
                self._current_turn.add_step(metrics)

    def record_step(self, step: StepMetrics) -> None:
        """Directly record a pre-built step metric."""
        if self._current_turn:
            self._current_turn.add_step(step)

    def record_invariant_check(
        self,
        invariant_id: str,
        passed: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an invariant verification result."""
        self._session.record_invariant_check(invariant_id, passed, details)

    def get_current_turn(self) -> Optional[TurnMetrics]:
        """Get the current turn metrics (if in progress)."""
        return self._current_turn

    def get_session_metrics(self) -> SessionMetrics:
        """Get the complete session metrics."""
        return self._session

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary dict suitable for logging."""
        return self._session.to_dict()

    def export_to_json(self, path: Path) -> None:
        """Export session metrics to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._session.to_dict(), f, indent=2)


class _StepBuilder:
    """Builder for step metrics within measure_step context."""

    def __init__(self, step_name: str, source: str, metadata: Dict[str, Any]):
        self._step_name = step_name
        self._source = source
        self._metadata = metadata
        self._start_time_ns = 0
        self._end_time_ns = 0
        self._bytes_in = 0
        self._bytes_out = 0
        self._cache_hit = False
        self._success = True

    def set_bytes(self, bytes_in: int, bytes_out: int) -> None:
        """Set input and output byte counts."""
        self._bytes_in = bytes_in
        self._bytes_out = bytes_out

    def set_cache_hit(self, hit: bool) -> None:
        """Set whether this was a cache hit."""
        self._cache_hit = hit

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the step."""
        self._metadata[key] = value

    def build(self) -> StepMetrics:
        """Build the StepMetrics instance."""
        return StepMetrics(
            step_name=self._step_name,
            start_time_ns=self._start_time_ns,
            end_time_ns=self._end_time_ns,
            bytes_in=self._bytes_in,
            bytes_out=self._bytes_out,
            cache_hit=self._cache_hit,
            source=self._source,
            success=self._success,
            metadata=self._metadata,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_metrics_collector(
    session_id: str,
    token_estimator: Optional[Callable[[str], int]] = None
) -> MetricsCollector:
    """Create a new metrics collector for a session."""
    return MetricsCollector(session_id, token_estimator)


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


__all__ = [
    "StepMetrics",
    "TurnMetrics",
    "SessionMetrics",
    "MetricsCollector",
    "create_metrics_collector",
    "compute_content_hash",
]
