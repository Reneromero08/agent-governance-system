#!/usr/bin/env python3
"""
Speed Benchmarks for Cassette Network

Measures query latency and throughput to validate performance claims.
"""
import json
import pytest
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add paths for imports
COMPRESSION_DIR = Path(__file__).resolve().parent
CASSETTE_NETWORK_DIR = COMPRESSION_DIR.parent
REPO_ROOT = CASSETTE_NETWORK_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT))

# Cassette network imports
CORTEX_NETWORK = REPO_ROOT / "NAVIGATION" / "CORTEX" / "network"
sys.path.insert(0, str(CORTEX_NETWORK))


@pytest.fixture(scope="module")
def geometric_network():
    """Load the geometric cassette network."""
    try:
        from geometric_cassette import GeometricCassetteNetwork

        network = GeometricCassetteNetwork.from_config(
            project_root=REPO_ROOT
        )

        if not network.cassettes:
            pytest.skip("No geometric cassettes loaded")

        return network
    except ImportError as e:
        pytest.skip(f"Geometric network not available: {e}")


@pytest.fixture(scope="module")
def benchmark_queries():
    """Standard queries for benchmarking."""
    return [
        "What are the system invariants?",
        "How do I bootstrap an agent?",
        "What is the genesis prompt?",
        "How does verification work?",
        "What are the contract rules?",
        "Explain catalytic computing",
        "What is the authority gradient?",
        "Where are receipts stored?",
        "What is the compression protocol?",
        "How do I recover from invariant violation?",
    ]


@pytest.mark.benchmark
class TestQueryLatency:
    """Benchmark query response times."""

    def test_single_query_latency(self, geometric_network, benchmark_queries):
        """Measure latency for single queries."""
        latencies = []

        for query in benchmark_queries:
            start = time.perf_counter()
            results = geometric_network.query_all_text(query, k=10)
            elapsed = (time.perf_counter() - start) * 1000  # ms

            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        max_latency = max(latencies)

        print(f"\n=== Query Latency Benchmark ===")
        print(f"Queries: {len(benchmark_queries)}")
        print(f"Average: {avg_latency:.1f}ms")
        print(f"P50: {p50:.1f}ms")
        print(f"P95: {p95:.1f}ms")
        print(f"Max: {max_latency:.1f}ms")

        # Performance targets
        assert avg_latency < 500, f"Average latency {avg_latency:.1f}ms > 500ms target"
        assert p95 < 1000, f"P95 latency {p95:.1f}ms > 1000ms target"

    def test_cold_start_latency(self, benchmark_queries):
        """Measure cold start (network load + first query)."""
        from geometric_cassette import GeometricCassetteNetwork

        start = time.perf_counter()
        network = GeometricCassetteNetwork.from_config(project_root=REPO_ROOT)
        load_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        results = network.query_all_text(benchmark_queries[0], k=10)
        first_query = (time.perf_counter() - start) * 1000

        total_cold_start = load_time + first_query

        print(f"\n=== Cold Start Benchmark ===")
        print(f"Network load: {load_time:.0f}ms")
        print(f"First query: {first_query:.1f}ms")
        print(f"Total cold start: {total_cold_start:.0f}ms")

        # Cold start should be under 10 seconds
        assert total_cold_start < 10000, f"Cold start {total_cold_start:.0f}ms > 10s"

    def test_throughput(self, geometric_network, benchmark_queries):
        """Measure queries per second."""
        num_iterations = 5
        total_queries = len(benchmark_queries) * num_iterations

        start = time.perf_counter()
        for _ in range(num_iterations):
            for query in benchmark_queries:
                geometric_network.query_all_text(query, k=10)
        elapsed = time.perf_counter() - start

        qps = total_queries / elapsed

        print(f"\n=== Throughput Benchmark ===")
        print(f"Total queries: {total_queries}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Throughput: {qps:.1f} queries/sec")

        # Should handle at least 4 queries/sec (vector search across 11K+ docs)
        assert qps >= 4, f"Throughput {qps:.1f} qps < 4 qps target"


@pytest.mark.benchmark
class TestIndexStats:
    """Benchmark index characteristics."""

    def test_corpus_stats(self, geometric_network):
        """Report corpus statistics."""
        total_docs = 0
        cassette_stats = []

        for cassette_id, cassette in geometric_network.cassettes.items():
            stats = cassette.get_stats()
            doc_count = stats.get("total_chunks", 0)
            total_docs += doc_count
            cassette_stats.append({
                "cassette": cassette_id,
                "docs": doc_count
            })

        print(f"\n=== Corpus Statistics ===")
        print(f"Total cassettes: {len(geometric_network.cassettes)}")
        print(f"Total documents: {total_docs}")
        for cs in sorted(cassette_stats, key=lambda x: -x["docs"]):
            print(f"  {cs['cassette']}: {cs['docs']} docs")

        assert total_docs > 0, "No documents in index"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
