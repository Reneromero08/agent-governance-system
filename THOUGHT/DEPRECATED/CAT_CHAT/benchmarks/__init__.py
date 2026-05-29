"""
Benchmarks Package - Phase I.2
==============================

Compression benchmarks for proving catalytic compression claims.

Provides:
- Deterministic benchmark scenarios
- Benchmark runner for A/B comparison
- Report generation
"""

from .scenarios import (
    BenchmarkScenario,
    ConversationTurn,
    PlantedFact,
    get_scenario,
    list_scenarios,
    SHORT_CONVERSATION,
    MEDIUM_CONVERSATION,
    LONG_CONVERSATION,
    SOFTWARE_ARCHITECTURE,
    DENSE_TECHNICAL,
)

from .runner import (
    BenchmarkResult,
    BenchmarkRunner,
    run_benchmark,
    run_comparison,
)

from .reporter import (
    ComparisonReport,
    BenchmarkReporter,
    generate_comparison_report,
)

__all__ = [
    # Scenarios
    "BenchmarkScenario",
    "ConversationTurn",
    "PlantedFact",
    "get_scenario",
    "list_scenarios",
    "SHORT_CONVERSATION",
    "MEDIUM_CONVERSATION",
    "LONG_CONVERSATION",
    "SOFTWARE_ARCHITECTURE",
    "DENSE_TECHNICAL",
    # Runner
    "BenchmarkResult",
    "BenchmarkRunner",
    "run_benchmark",
    "run_comparison",
    # Reporter
    "ComparisonReport",
    "BenchmarkReporter",
    "generate_comparison_report",
]
