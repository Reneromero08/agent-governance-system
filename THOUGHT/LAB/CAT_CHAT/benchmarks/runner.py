"""
Benchmark Runner - Phase I.2
============================

Executes benchmark scenarios and collects metrics.

Supports:
- Catalytic mode (with compression)
- Baseline mode (no compression)
- A/B comparison
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from .scenarios import BenchmarkScenario, ConversationTurn, PlantedFact


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """
    Results from running a benchmark scenario.
    """
    scenario_name: str
    mode: str  # "catalytic" or "baseline"
    started_at: str
    completed_at: str

    # Token metrics
    total_tokens_used: int = 0
    peak_context_tokens: int = 0
    average_context_tokens: float = 0.0

    # Compression metrics
    bytes_expanded: int = 0
    bytes_stored: int = 0
    compression_ratio: float = 0.0

    # Quality metrics
    recall_rate: float = 0.0
    recall_details: List[Dict[str, Any]] = field(default_factory=list)
    e_score_mean: float = 0.0
    e_score_samples: List[float] = field(default_factory=list)

    # Performance metrics
    total_latency_ms: float = 0.0
    per_turn_latency_ms: List[float] = field(default_factory=list)

    # Resource metrics
    peak_memory_mb: float = 0.0

    # Error tracking
    errors: List[str] = field(default_factory=list)
    turns_completed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_name": self.scenario_name,
            "mode": self.mode,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "turns_completed": self.turns_completed,
            "tokens": {
                "total_used": self.total_tokens_used,
                "peak_context": self.peak_context_tokens,
                "average_context": round(self.average_context_tokens, 1),
            },
            "compression": {
                "bytes_expanded": self.bytes_expanded,
                "bytes_stored": self.bytes_stored,
                "ratio": round(self.compression_ratio, 2),
            },
            "quality": {
                "recall_rate": round(self.recall_rate, 4),
                "e_score_mean": round(self.e_score_mean, 4),
                "recall_details": self.recall_details,
            },
            "performance": {
                "total_latency_ms": round(self.total_latency_ms, 1),
                "avg_turn_latency_ms": round(
                    sum(self.per_turn_latency_ms) / len(self.per_turn_latency_ms), 1
                ) if self.per_turn_latency_ms else 0,
            },
            "errors": self.errors,
        }


# =============================================================================
# Simulated Chat Interfaces
# =============================================================================

class MockCatalyticChat:
    """
    Mock catalytic chat for benchmark testing.

    Simulates compression behavior with configurable parameters.
    """

    def __init__(
        self,
        compression_ratio: float = 5.0,
        base_latency_ms: float = 50.0,
        e_score_mean: float = 0.7,
    ):
        self.compression_ratio = compression_ratio
        self.base_latency_ms = base_latency_ms
        self.e_score_mean = e_score_mean

        self.total_bytes_expanded = 0
        self.total_bytes_stored = 0
        self.context_tokens = 0
        self.peak_context_tokens = 0
        self.turn_latencies: List[float] = []
        self.e_scores: List[float] = []
        self.history: List[str] = []

    def respond(self, query: str) -> Dict[str, Any]:
        """Process a query and return metrics."""
        start_time = time.perf_counter()

        # Simulate response generation
        response = f"Response to: {query[:50]}..."
        response_bytes = len(response.encode())

        # Simulate compression
        expanded = len(query.encode()) + response_bytes
        stored = int(expanded / self.compression_ratio)

        self.total_bytes_expanded += expanded
        self.total_bytes_stored += stored

        # Simulate context growth with compression
        self.context_tokens += stored // 4  # ~4 bytes per token
        self.peak_context_tokens = max(self.peak_context_tokens, self.context_tokens)

        # Simulate E-score
        import random
        e_score = max(0.0, min(1.0, self.e_score_mean + random.gauss(0, 0.1)))
        self.e_scores.append(e_score)

        # Record latency
        latency = (time.perf_counter() - start_time) * 1000 + self.base_latency_ms
        self.turn_latencies.append(latency)

        self.history.append(query)

        return {
            "response": response,
            "bytes_expanded": expanded,
            "bytes_stored": stored,
            "e_score": e_score,
            "latency_ms": latency,
        }

    def check_recall(self, fact: PlantedFact) -> bool:
        """Check if a planted fact would be recalled."""
        # In real implementation, this would check E-score against fact content
        # For mock, use E-score to simulate recall probability
        import random
        return random.random() < self.e_score_mean


class MockBaselineChat:
    """
    Mock baseline chat (no compression) for comparison.

    Keeps full history in context without compression.
    """

    def __init__(self, base_latency_ms: float = 100.0):
        self.base_latency_ms = base_latency_ms

        self.total_bytes = 0
        self.context_tokens = 0
        self.peak_context_tokens = 0
        self.turn_latencies: List[float] = []
        self.history: List[str] = []

    def respond(self, query: str) -> Dict[str, Any]:
        """Process a query and return metrics."""
        start_time = time.perf_counter()

        # Simulate response generation
        response = f"Response to: {query[:50]}..."

        # No compression - full content stays in context
        turn_bytes = len(query.encode()) + len(response.encode())
        self.total_bytes += turn_bytes

        # Context grows linearly without compression
        self.context_tokens += turn_bytes // 4
        self.peak_context_tokens = max(self.peak_context_tokens, self.context_tokens)

        # Record latency (baseline is slower due to larger context)
        context_penalty = self.context_tokens / 1000  # Slower with more context
        latency = (time.perf_counter() - start_time) * 1000 + self.base_latency_ms + context_penalty
        self.turn_latencies.append(latency)

        self.history.append(query)

        return {
            "response": response,
            "bytes": turn_bytes,
            "latency_ms": latency,
        }

    def check_recall(self, fact: PlantedFact) -> bool:
        """Baseline always has full context, so recall is high."""
        import random
        return random.random() < 0.95  # 95% recall with full context


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Runs benchmark scenarios and collects results.

    Usage:
        runner = BenchmarkRunner()

        # Run catalytic mode
        catalytic_result = runner.run(scenario, mode="catalytic")

        # Run baseline mode
        baseline_result = runner.run(scenario, mode="baseline")

        # Compare results
        comparison = runner.compare(catalytic_result, baseline_result)
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        catalytic_chat_factory: Optional[Callable] = None,
        baseline_chat_factory: Optional[Callable] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            output_dir: Directory for benchmark results
            catalytic_chat_factory: Factory for catalytic chat instances
            baseline_chat_factory: Factory for baseline chat instances
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "_generated" / "benchmark_results"
        self.output_dir = Path(output_dir)

        self.catalytic_chat_factory = catalytic_chat_factory or (lambda: MockCatalyticChat())
        self.baseline_chat_factory = baseline_chat_factory or (lambda: MockBaselineChat())

    def run(
        self,
        scenario: BenchmarkScenario,
        mode: str = "catalytic",
    ) -> BenchmarkResult:
        """
        Run a benchmark scenario.

        Args:
            scenario: Benchmark scenario to run
            mode: "catalytic" or "baseline"

        Returns:
            BenchmarkResult with all metrics
        """
        if mode not in ("catalytic", "baseline"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'catalytic' or 'baseline'")

        started_at = datetime.now(timezone.utc).isoformat()

        # Create chat instance
        if mode == "catalytic":
            chat = self.catalytic_chat_factory()
        else:
            chat = self.baseline_chat_factory()

        result = BenchmarkResult(
            scenario_name=scenario.name,
            mode=mode,
            started_at=started_at,
            completed_at="",
        )

        context_tokens_sum = 0
        recall_successes = 0
        recall_attempts = 0

        try:
            for turn in scenario.turns:
                # Process turn
                try:
                    turn_result = chat.respond(turn.user_query)
                    result.turns_completed += 1

                    # Track context tokens
                    if hasattr(chat, 'context_tokens'):
                        context_tokens_sum += chat.context_tokens

                    # Track latency
                    if 'latency_ms' in turn_result:
                        result.per_turn_latency_ms.append(turn_result['latency_ms'])

                    # Track E-scores (catalytic only)
                    if mode == "catalytic" and 'e_score' in turn_result:
                        result.e_score_samples.append(turn_result['e_score'])

                except Exception as e:
                    result.errors.append(f"Turn {turn.turn_index}: {str(e)}")

                # Check recall for planted facts from earlier turns
                facts_before = scenario.get_facts_before(turn.turn_index)
                for fact in facts_before:
                    # Only test recall if keywords appear in current query
                    if any(kw.lower() in turn.user_query.lower() for kw in fact.keywords):
                        recall_attempts += 1
                        if chat.check_recall(fact):
                            recall_successes += 1
                            result.recall_details.append({
                                "fact_id": fact.fact_id,
                                "turn": turn.turn_index,
                                "recalled": True,
                            })
                        else:
                            result.recall_details.append({
                                "fact_id": fact.fact_id,
                                "turn": turn.turn_index,
                                "recalled": False,
                            })

        except Exception as e:
            result.errors.append(f"Scenario error: {str(e)}")

        # Collect final metrics
        result.completed_at = datetime.now(timezone.utc).isoformat()

        if mode == "catalytic":
            result.bytes_expanded = chat.total_bytes_expanded
            result.bytes_stored = chat.total_bytes_stored
            result.compression_ratio = (
                chat.total_bytes_expanded / chat.total_bytes_stored
                if chat.total_bytes_stored > 0 else 0
            )
            result.peak_context_tokens = chat.peak_context_tokens
            result.e_score_mean = (
                sum(chat.e_scores) / len(chat.e_scores)
                if chat.e_scores else 0
            )
        else:
            result.bytes_expanded = chat.total_bytes
            result.bytes_stored = chat.total_bytes  # No compression
            result.compression_ratio = 1.0
            result.peak_context_tokens = chat.peak_context_tokens

        result.total_tokens_used = result.peak_context_tokens
        result.average_context_tokens = (
            context_tokens_sum / result.turns_completed
            if result.turns_completed > 0 else 0
        )

        result.recall_rate = (
            recall_successes / recall_attempts
            if recall_attempts > 0 else 1.0
        )

        result.total_latency_ms = sum(result.per_turn_latency_ms)

        return result

    def run_comparison(
        self,
        scenario: BenchmarkScenario,
    ) -> Dict[str, Any]:
        """
        Run scenario in both modes and return comparison.

        Args:
            scenario: Benchmark scenario to run

        Returns:
            Comparison dict with both results and analysis
        """
        catalytic_result = self.run(scenario, mode="catalytic")
        baseline_result = self.run(scenario, mode="baseline")

        return {
            "scenario": scenario.to_dict(),
            "catalytic": catalytic_result.to_dict(),
            "baseline": baseline_result.to_dict(),
            "comparison": self._compute_comparison(catalytic_result, baseline_result),
        }

    def _compute_comparison(
        self,
        catalytic: BenchmarkResult,
        baseline: BenchmarkResult,
    ) -> Dict[str, Any]:
        """Compute comparison metrics between modes."""
        def safe_div(a, b):
            return a / b if b > 0 else 0

        return {
            "compression_improvement": {
                "bytes_saved": baseline.bytes_stored - catalytic.bytes_stored,
                "bytes_saved_pct": safe_div(
                    baseline.bytes_stored - catalytic.bytes_stored,
                    baseline.bytes_stored
                ) * 100,
                "compression_ratio": catalytic.compression_ratio,
            },
            "token_improvement": {
                "tokens_saved": baseline.peak_context_tokens - catalytic.peak_context_tokens,
                "tokens_saved_pct": safe_div(
                    baseline.peak_context_tokens - catalytic.peak_context_tokens,
                    baseline.peak_context_tokens
                ) * 100,
            },
            "quality": {
                "recall_catalytic": catalytic.recall_rate,
                "recall_baseline": baseline.recall_rate,
                "recall_delta": catalytic.recall_rate - baseline.recall_rate,
            },
            "performance": {
                "latency_catalytic_ms": catalytic.total_latency_ms,
                "latency_baseline_ms": baseline.total_latency_ms,
                "latency_improvement_pct": safe_div(
                    baseline.total_latency_ms - catalytic.total_latency_ms,
                    baseline.total_latency_ms
                ) * 100,
            },
        }

    def save_results(
        self,
        results: Dict[str, Any],
        name: Optional[str] = None,
    ) -> Path:
        """
        Save benchmark results to file.

        Args:
            results: Results dictionary
            name: Optional name for the results file

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name:
            filename = f"{name}_{timestamp}.json"
        else:
            filename = f"benchmark_{timestamp}.json"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def run_benchmark(
    scenario: BenchmarkScenario,
    mode: str = "catalytic",
) -> BenchmarkResult:
    """Run a single benchmark scenario."""
    runner = BenchmarkRunner()
    return runner.run(scenario, mode)


def run_comparison(
    scenario: BenchmarkScenario,
) -> Dict[str, Any]:
    """Run scenario comparison between catalytic and baseline."""
    runner = BenchmarkRunner()
    return runner.run_comparison(scenario)


__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "MockCatalyticChat",
    "MockBaselineChat",
    "run_benchmark",
    "run_comparison",
]
