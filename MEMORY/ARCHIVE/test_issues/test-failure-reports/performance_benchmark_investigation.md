# Performance Benchmark Investigation Report

**Date**: 2026-01-25
**Status**: Investigation Complete
**Verdict**: The xfail is OVERLY CONSERVATIVE - Performance is EXCELLENT

---

## Executive Summary

The `test_single_query_latency` test in `test_speed_benchmarks.py` is marked `xfail` with reason "Performance benchmark - environment dependent, may exceed thresholds on loaded systems." This investigation found:

1. **Current performance is excellent**: 15-28ms average latency (target <100ms)
2. **The test actually PASSES** (XPASS) when run
3. **The xfail was added defensively** for CI stability, not due to actual failures
4. **The 100ms target is realistic** and based on the documented roadmap
5. **Optimizations ARE enabled**: Vectorized numpy queries achieving 16x speedup

**Recommendation**: Remove the xfail - the performance is consistently well under threshold.

---

## Investigation Details

### 1. Test File Analysis

**File**: `D:\CCC 2.0\AI\agent-governance-system\CAPABILITY\TESTBENCH\cassette_network\compression\test_speed_benchmarks.py`

The test measures query latency across 10 benchmark queries:
```python
# Performance targets (per Success Metrics in CASSETTE_NETWORK_ROADMAP.md)
assert avg_latency < 100, f"Average latency {avg_latency:.1f}ms > 100ms target"
assert p95 < 200, f"P95 latency {p95:.1f}ms > 200ms target"
```

The xfail was added in commit `97367950` on 2026-01-25:
```
test: mark performance benchmark as xfail for CI stability

The query latency benchmark can exceed thresholds on loaded systems.
Mark as xfail to prevent blocking pushes while preserving the metric.
```

### 2. Actual Performance Measurements

**Test Run Results** (2026-01-25):
```
=== Query Latency Benchmark ===
Queries: 10
Average: 15.0ms
P50: 14.2ms
P95: 20.1ms
Max: 20.1ms
```

**Benchmark Suite Results** (benchmark_success_metrics.py):
```
Search Latency (<100ms target)...
   PASS: avg=27.5ms, p50=24.0ms, p95=43.2ms
```

**Manual Verification** (3 queries):
```
Query 1: 17.4ms
Query 2: 23.0ms
Query 3: 16.1ms
Average: 18.8ms
```

**Performance is 3-6x BETTER than required.**

### 3. Where Does the 100ms Target Come From?

The 100ms target is documented in `MEMORY\ARCHIVE\cassette-network-research\CASSETTE_NETWORK_ROADMAP.md`:

```markdown
## Success Metrics

**Status**: ALL METRICS PASSING (2026-01-18)

### Performance
- [x] Search latency: <100ms across all cassettes - Actual: 8.3ms avg (vectorized queries)
- [x] Indexing throughput: >100 chunks/sec per cassette - Actual: 140+ chunks/sec
- [x] Network overhead: <10ms per cassette query - Actual: <1ms (negligible)
```

The target is based on:
- Industry standard for interactive search (<100ms for user-perceived responsiveness)
- Validation that queries across 11,781 documents in 9 cassettes remain interactive
- Original implementation (before optimization) ran at ~132ms, so 100ms was a reasonable ceiling

### 4. Why Was It Slow Before? What Fixed It?

**Before (commit 18e06f17)**: Loop-based queries at ~132ms
**After (commit 6ee85e6a)**: Vectorized queries at 8.3ms

The optimization in `geometric_cassette.py`:

```python
def _query_geometric_vectorized(self, query_state, k):
    """Vectorized query using numpy matrix operations."""
    # Single matrix-vector multiply: E_scores = M @ q
    # (N dot products in one op instead of Python loop)
    E_scores = self._vector_matrix @ query_state.vector

    # O(N) argpartition instead of O(N log N) sort
    if k < len(E_scores):
        top_k_indices = np.argpartition(E_scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(E_scores[top_k_indices])[::-1]]
```

**Key optimizations**:
1. Pre-build N x D matrix at index load time
2. Single numpy matrix multiply replaces 11K+ Python loop iterations
3. `np.argpartition` for O(N) top-k selection vs O(N log N) full sort
4. Results: 132ms -> 8.3ms (16x speedup)

**These optimizations are ENABLED in production.**

### 5. Is the Test Measuring the Right Thing?

**Yes.** The test correctly:
- Warms up the embedding model first (excludes cold-start from latency)
- Measures actual `query_all_text()` calls with realistic queries
- Tests across all 9 cassettes with 11,781+ documents
- Calculates avg, p50, p95, and max latencies

The warm-up is important because embedding generation (~200ms for first query due to model loading) should not be counted in query latency. The test correctly excludes this.

### 6. What About CI Environment Variability?

The xfail reason mentions "loaded systems." Analysis:

**Current headroom**: 15ms avg vs 100ms target = 85ms buffer (6.6x margin)
**Even worst case observed**: 43ms p95 vs 200ms target = 157ms buffer (4.7x margin)

A CI system would need to be **extremely** loaded to push 15ms queries above 100ms. This level of variance (6x degradation) is unrealistic for properly provisioned CI.

### 7. Success Metrics Summary

| Metric | Target | Actual | Status | Headroom |
|--------|--------|--------|--------|----------|
| Search Latency (avg) | <100ms | 15-28ms | PASS | 72-85ms |
| Search Latency (p95) | <200ms | 20-43ms | PASS | 157-180ms |
| Cold Start | <5000ms | ~2300ms | PASS | 2700ms |
| Throughput | >4 qps | ~50 qps | PASS | 12x margin |

---

## Root Cause Analysis

**The xfail is NOT due to an actual performance problem.**

Root cause: Defensive marking to prevent CI flakes. However:
1. The performance has 6x margin over threshold
2. The test currently reports XPASS (expected fail, actually passed)
3. The xfail message does not reflect actual failure data

**There is no performance bug to fix.**

---

## Recommendation: The Real Fix

### Immediate Action: Remove the xfail

The test should not be xfailed. Change:

```python
# BEFORE (overly conservative)
@pytest.mark.xfail(reason="Performance benchmark - environment dependent, may exceed thresholds on loaded systems")
def test_single_query_latency(self, geometric_network, benchmark_queries):
```

To:

```python
# AFTER (correct)
def test_single_query_latency(self, geometric_network, benchmark_queries):
```

### If CI Flakiness is a Concern

If there's actual evidence of CI failures (there isn't in the git history), alternatives:
1. **Increase threshold**: 100ms -> 150ms (still reasonable for "interactive")
2. **Add retry**: `@pytest.mark.flaky(reruns=1)` with pytest-rerunfailures
3. **Skip in constrained CI**: `@pytest.mark.skipif(os.environ.get("CI_CONSTRAINED"))`

But given 6x margin, these are unnecessary.

---

## Architecture Notes

### Why Performance is Good

1. **Vectorized Queries**: Single numpy `@` operation replaces Python loops
2. **Pre-built Index**: Matrix built at load time, not query time
3. **Efficient Top-K**: `argpartition` is O(N) vs O(N log N) sort
4. **Model Caching**: `all-MiniLM-L6-v2` loaded once, reused for all queries
5. **Geometric Queries**: E (Born rule) = single dot product, O(1) per document

### Cassette Network Statistics

- 9 active cassettes
- 11,781 total documents indexed
- 384-dimensional embeddings (MiniLM)
- Vectorized matrix: 11,781 x 384 = ~18MB float32

### One Actual Issue Found

The `indexing_throughput` metric FAILS (45.8 chunks/sec vs >100 target). This is because:
- The test measures `reasoner.initialize()` calls (embedding generation)
- Embedding is I/O bound, not the geometric queries
- This is a separate concern from query latency

---

## Files Reviewed

1. `CAPABILITY/TESTBENCH/cassette_network/compression/test_speed_benchmarks.py` - The test
2. `CAPABILITY/TESTBENCH/cassette_network/benchmark_success_metrics.py` - Comprehensive benchmark
3. `NAVIGATION/CORTEX/network/geometric_cassette.py` - Core implementation
4. `CAPABILITY/PRIMITIVES/geometric_reasoner.py` - Embedding engine
5. `MEMORY/ARCHIVE/cassette-network-research/CASSETTE_NETWORK_ROADMAP.md` - Target definitions
6. `NAVIGATION/CORTEX/network/cassettes.json` - Cassette configuration

## Git History

- `97367950` - Added xfail (2026-01-25)
- `6ee85e6a` - 16x speedup optimization (2026-01-18)
- `18e06f17` - Initial compression tests (earlier)

---

## Conclusion

The performance benchmark test is xfailed unnecessarily. The cassette network achieves:
- **15-28ms average query latency** (target: <100ms)
- **16x speedup** from vectorized numpy operations
- **11,781 documents** searchable interactively

**The "fix" is removing the xfail marker, not changing performance targets or code.**

The system is performing exactly as designed and documented.
