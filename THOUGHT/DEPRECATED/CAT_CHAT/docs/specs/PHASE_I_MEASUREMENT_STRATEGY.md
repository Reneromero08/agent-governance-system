# Phase I: Measurement & Benchmarking - Implementation Strategy

**Status:** Ready for Implementation
**Priority:** P3
**Purpose:** Prove catalytic compression with reproducible numbers
**Depends On:** Phases A-H (all complete)

---

## Executive Summary

Phase I transforms CAT Chat from "trust us, it compresses" to "here are the numbers." The infrastructure already captures metrics at multiple levels - this phase aggregates, benchmarks, and verifies them into reproducible proof.

**Three Deliverables:**
1. **I.1 Per-Step Metrics** - Detailed telemetry for every operation
2. **I.2 Compression Benchmarks** - Comparison against baseline (no compression)
3. **I.3 Invariant Verification Suite** - Automated proof of all 7 catalytic invariants

---

## Current State Analysis

### Existing Metrics Infrastructure (Already Built)

| Component | Location | Metrics Captured |
|-----------|----------|------------------|
| SPC Compression | `spc_bridge.py` | tokens_expanded, tokens_pointers, tokens_saved, CDR |
| Turn Compression | `turn_compressor.py` | original_tokens, pointer_tokens, compression_ratio |
| Context Partitioning | `context_partitioner.py` | E-scores, budget_used, working_set size |
| Auto-Context Loop | `auto_context_manager.py` | E_mean, tokens_in_context, compression_ratio |
| Budget Tracking | `adaptive_budget.py` | utilization_pct, tokens_remaining |
| Retrieval Chain | `cortex_expansion_resolver.py` | hit counts by source, retrieval_time_ms |
| Docs Index | `docs_index.py` | files indexed, total size, FTS stats |
| Session Events | `session_capsule.py` | Hash-chained log of all events |

### What's Missing (Phase I Scope)

1. **Aggregation Layer** - No unified metrics collector
2. **Baseline Comparison** - No "control group" without compression
3. **Invariant Verification** - Checks exist but aren't comprehensive
4. **Benchmark Runner** - No reproducible benchmark harness
5. **Report Generation** - No human-readable output

---

## Implementation Plan

### I.1 Per-Step Metrics Collector

**Goal:** Capture detailed telemetry for every operation with timing and resource usage.

#### I.1.1 Create `MetricsCollector` Class

**File:** `catalytic_chat/metrics_collector.py`

```python
@dataclass
class StepMetrics:
    """Metrics for a single operation step."""
    step_name: str           # e.g., "spc_resolve", "vector_search"
    start_time_ns: int       # Nanosecond precision timing
    end_time_ns: int
    bytes_in: int            # Input size
    bytes_out: int           # Output size
    cache_hit: bool          # Whether result was cached
    source: str              # Where data came from
    metadata: dict           # Step-specific details

@dataclass
class TurnMetrics:
    """Aggregated metrics for one turn."""
    turn_index: int
    steps: list[StepMetrics]
    total_bytes_expanded: int
    total_bytes_compressed: int
    cache_hit_rate: float
    e_score_mean: float
    latency_ms: float

@dataclass
class SessionMetrics:
    """Aggregated metrics for entire session."""
    session_id: str
    turns: list[TurnMetrics]
    total_turns: int
    total_bytes_expanded: int
    total_bytes_compressed: int
    overall_compression_ratio: float
    cache_hit_rate: float
    e_score_distribution: dict  # histogram
    invariant_checks: dict      # pass/fail per invariant
```

#### I.1.2 Instrumentation Points

Wrap existing code with metric collection at these points:

| Operation | File | Metrics to Capture |
|-----------|------|-------------------|
| SPC Resolution | `spc_bridge.py:resolve()` | timing, tokens_in/out, symbol used |
| Cassette FTS | `cassette_client.py:search()` | timing, hits, query_terms |
| Local Index | `symbol_resolver.py:resolve()` | timing, cache_hit, symbol |
| CAS Lookup | `cas_resolver.py:get()` | timing, hash, found/miss |
| Vector Search | `vector_fallback.py:search()` | timing, k, similarity_threshold |
| Context Partition | `context_partitioner.py:partition()` | E-scores, budget_used, evicted |
| Turn Compress | `turn_compressor.py:compress()` | original_tokens, pointer_tokens |
| Turn Hydrate | `turn_compressor.py:hydrate()` | pointer_tokens, expanded_tokens |

#### I.1.3 Persistence to Session Events

Add new event types to `session_capsule.py`:

```python
EVENT_STEP_METRICS = "step_metrics"      # Per-step telemetry
EVENT_TURN_METRICS = "turn_metrics"      # Per-turn aggregation
EVENT_SESSION_METRICS = "session_metrics"  # Session summary
```

Log metrics to hash-chained session_events for deterministic replay and audit.

---

### I.2 Compression Benchmarks vs Baseline

**Goal:** Quantify compression benefit with A/B comparison.

#### I.2.1 Baseline Mode Implementation

**File:** `catalytic_chat/baseline_mode.py`

Create "no compression" mode for fair comparison:

```python
class BaselineChat:
    """Chat without catalytic compression - control group."""

    def respond(self, query: str) -> BaselineResult:
        """Process query with full context (no eviction/compression)."""
        # - Keep ALL turns in context (no turn compression)
        # - No pointer sets (everything materialized)
        # - No SPC (full @symbol expansion)
        # - Track: total_tokens_used, latency, memory
```

#### I.2.2 Benchmark Scenarios

**File:** `benchmarks/scenarios.py`

Create deterministic benchmark scenarios:

| Scenario | Turns | Description | Planted Facts |
|----------|-------|-------------|---------------|
| `short_conversation` | 10 | Quick sanity check | 2 |
| `medium_conversation` | 30 | Typical usage | 5 |
| `long_conversation` | 100 | Memory stress | 15 |
| `software_architecture` | 63 | Real domain (from stress test) | 12 |
| `dense_technical` | 50 | High-compression potential | 10 |

Each scenario defines:
- Fixed seed for reproducibility
- Conversation script (user queries + expected topics)
- Planted facts with known retrieval points
- Success criteria (recall rate, compression ratio)

#### I.2.3 Benchmark Runner

**File:** `benchmarks/runner.py`

```python
@dataclass
class BenchmarkResult:
    scenario_name: str
    mode: str  # "catalytic" or "baseline"

    # Token metrics
    total_tokens_used: int
    peak_context_tokens: int

    # Compression metrics
    bytes_expanded: int
    bytes_stored: int
    compression_ratio: float

    # Quality metrics
    recall_rate: float  # % planted facts retrieved
    e_score_mean: float

    # Performance metrics
    total_latency_ms: float
    per_turn_latency_ms: list[float]

    # Resource metrics
    peak_memory_mb: float

def run_benchmark(scenario: Scenario, mode: str) -> BenchmarkResult:
    """Run scenario in specified mode, return metrics."""
```

#### I.2.4 Comparison Report

**Output:** `_generated/benchmark_results/`

```
benchmark_results/
    {timestamp}/
        config.json           # Run configuration
        catalytic_results.json
        baseline_results.json
        comparison.md         # Human-readable report
        raw_events.jsonl      # Full event trace
```

**Comparison Report Format:**

```markdown
# Benchmark Comparison Report

## Summary
| Metric | Baseline | Catalytic | Improvement |
|--------|----------|-----------|-------------|
| Peak Context Tokens | 45,000 | 12,000 | 73.3% reduction |
| Total Bytes | 180,000 | 48,000 | 73.3% compression |
| Recall Rate | 100% | 95% | -5% (acceptable) |
| Mean Latency | 1,200ms | 450ms | 62.5% faster |

## Per-Scenario Results
...

## Compression Breakdown
- SPC Compression: 56,370x (for loaded symbols)
- Turn Compression: 8.5x average
- Context Eviction: 3.2x (working set vs full history)
```

---

### I.3 Catalytic Invariant Verification Suite

**Goal:** Automated verification of all 7 catalytic invariants.

#### I.3.1 Invariant Test Framework

**File:** `catalytic_chat/invariant_verifier.py`

```python
@dataclass
class InvariantResult:
    invariant_id: str  # "INV-CATALYTIC-01" through "INV-CATALYTIC-07"
    name: str
    passed: bool
    evidence: dict     # Proof of compliance or violation
    timestamp: str

class InvariantVerifier:
    """Comprehensive verification of all catalytic invariants."""

    def verify_all(self, session_id: str) -> list[InvariantResult]:
        """Run all invariant checks on a session."""
        return [
            self.verify_inv_01_restoration(session_id),
            self.verify_inv_02_verification(session_id),
            self.verify_inv_03_reversibility(session_id),
            self.verify_inv_04_clean_space_bound(session_id),
            self.verify_inv_05_fail_closed(session_id),
            self.verify_inv_06_determinism(session_id),
            self.verify_inv_07_auto_context(session_id),
        ]
```

#### I.3.2 Invariant Verification Methods

**INV-CATALYTIC-01 (Restoration):**
```python
def verify_inv_01_restoration(self, session_id: str) -> InvariantResult:
    """File states before/after must be identical (or explicitly committed)."""
    # 1. Snapshot all files at session start
    # 2. Run session to completion
    # 3. Compare file states
    # 4. Any changes must have corresponding commit events
```

**INV-CATALYTIC-02 (Verification):**
```python
def verify_inv_02_verification(self, session_id: str) -> InvariantResult:
    """Proof size = O(1) per domain (single Merkle root)."""
    # 1. Extract all receipts from session
    # 2. Verify each has merkle_root
    # 3. Verify merkle_root size is constant (32 bytes SHA256)
```

**INV-CATALYTIC-03 (Reversibility):**
```python
def verify_inv_03_reversibility(self, session_id: str) -> InvariantResult:
    """restore(snapshot) = original (byte-identical)."""
    # 1. Create snapshot at turn N
    # 2. Continue session
    # 3. Restore snapshot
    # 4. Verify byte-identical to original state
```

**INV-CATALYTIC-04 (Clean Space Bound):**
```python
def verify_inv_04_clean_space_bound(self, session_id: str) -> InvariantResult:
    """Context uses pointers, not full content."""
    # 1. Extract all EVENT_PARTITION events
    # 2. Verify budget_used <= budget_total for ALL turns
    # 3. Verify no BudgetExceededError was suppressed
```

**INV-CATALYTIC-05 (Fail-Closed):**
```python
def verify_inv_05_fail_closed(self, session_id: str) -> InvariantResult:
    """Restoration failure = hard exit."""
    # 1. Inject corruption into session state
    # 2. Attempt restore
    # 3. Verify exception raised (not silent failure)
```

**INV-CATALYTIC-06 (Determinism):**
```python
def verify_inv_06_determinism(self, session_id: str) -> InvariantResult:
    """Identical inputs = identical Merkle root."""
    # 1. Run bundle twice with identical inputs
    # 2. Compare merkle_roots
    # 3. Verify byte-identical
```

**INV-CATALYTIC-07 (Auto-Context):**
```python
def verify_inv_07_auto_context(self, session_id: str) -> InvariantResult:
    """Working set managed by system, not manual references."""
    # 1. Scan all user queries for @symbol references
    # 2. Verify none exist (user doesn't manually reference)
    # 3. Verify EVENT_PARTITION events show E-score based decisions
```

#### I.3.3 Invariant Test Suite

**File:** `tests/test_invariants.py`

```python
class TestCatalyticInvariants:
    """Comprehensive invariant verification tests."""

    def test_inv_01_restoration_clean_session(self):
        """Files unchanged after normal session."""

    def test_inv_01_restoration_with_commit(self):
        """File changes only via explicit commit."""

    def test_inv_02_verification_proof_size(self):
        """Merkle root is O(1) regardless of content size."""

    def test_inv_03_reversibility_snapshot_restore(self):
        """Snapshot restoration is byte-identical."""

    def test_inv_04_clean_space_all_turns(self):
        """Budget never exceeded across 100-turn session."""

    def test_inv_05_fail_closed_corruption(self):
        """Corrupted state triggers exception."""

    def test_inv_06_determinism_replay(self):
        """Same inputs produce same merkle root."""

    def test_inv_07_auto_context_no_manual(self):
        """No manual @references in benchmark scenarios."""
```

---

## File Structure

```
THOUGHT/LAB/CAT_CHAT/
    catalytic_chat/
        metrics_collector.py     # NEW - I.1
        baseline_mode.py         # NEW - I.2
        invariant_verifier.py    # NEW - I.3

    benchmarks/                  # NEW directory
        __init__.py
        scenarios.py             # Benchmark scenarios
        runner.py                # Benchmark execution
        reporter.py              # Report generation
        fixtures/                # Deterministic inputs
            short_conversation.json
            medium_conversation.json
            long_conversation.json
            software_architecture.json
            dense_technical.json

    tests/
        test_metrics_collector.py   # NEW
        test_benchmarks.py           # NEW
        test_invariants.py           # NEW

    _generated/
        benchmark_results/          # NEW - output directory
```

---

## Implementation Order

### Phase I.1: Per-Step Metrics (Estimated: Small)

1. Create `metrics_collector.py` with `StepMetrics`, `TurnMetrics`, `SessionMetrics`
2. Add instrumentation to existing resolution chain
3. Add new event types to `session_capsule.py`
4. Create `test_metrics_collector.py` with unit tests
5. Verify metrics capture in existing stress test

**Exit Criteria:** Every resolution step logged with timing and bytes

### Phase I.2: Compression Benchmarks (Estimated: Medium)

1. Create `baseline_mode.py` with no-compression chat
2. Create `benchmarks/` directory structure
3. Implement `scenarios.py` with 5 deterministic scenarios
4. Implement `runner.py` benchmark harness
5. Implement `reporter.py` comparison report generator
6. Run full benchmark suite, verify reproducibility

**Exit Criteria:** Comparison report shows compression ratios with statistical confidence

### Phase I.3: Invariant Verification (Estimated: Medium)

1. Create `invariant_verifier.py` with all 7 verification methods
2. Create `test_invariants.py` with comprehensive test suite
3. Integrate with benchmark runner for continuous verification
4. Generate invariant compliance report

**Exit Criteria:** All 7 invariants verified with automated tests

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Turn Compression Ratio | >= 5x | `original_tokens / pointer_tokens` |
| SPC Compression | >= 1000x | Per-symbol expansion ratio |
| Recall Rate | >= 90% | Planted facts retrieved correctly |
| Context Budget Compliance | 100% | No budget exceeded events |
| Invariant Pass Rate | 100% | All 7 invariants verified |
| Benchmark Reproducibility | 100% | Same inputs = same outputs |

---

## Dependencies

### Internal (All Complete)

- Phase A: Session persistence (checkpoints)
- Phase C: Auto-context loop (E-score partitioning)
- Phase D: SPC integration (compression metrics)
- Phase E: Vector fallback (retrieval chain)
- Phase G: Bundle replay (determinism verification)

### External (None Required)

Phase I uses only existing infrastructure. No new dependencies.

---

## CLI Commands

Add to `catalytic_chat/cli.py`:

```bash
# Run benchmarks
cat-chat benchmark run --scenario software_architecture
cat-chat benchmark run --all
cat-chat benchmark compare --results-dir _generated/benchmark_results/2026-01-20

# View metrics
cat-chat metrics show --session <session_id>
cat-chat metrics export --session <session_id> --format json

# Verify invariants
cat-chat invariants verify --session <session_id>
cat-chat invariants verify --all-sessions
cat-chat invariants report
```

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Baseline mode complexity | Low | Medium | Use existing chat, disable compression features |
| Benchmark flakiness | Medium | Low | Fixed seeds, deterministic scenarios |
| Invariant false positives | Low | High | Multiple verification approaches |
| Performance overhead | Low | Low | Metrics collection is O(1) per step |

---

## Timeline Recommendation

1. **Week 1:** I.1 (Metrics Collector)
   - Day 1-2: Core dataclasses and collector
   - Day 3-4: Instrumentation integration
   - Day 5: Tests and verification

2. **Week 2:** I.2 (Benchmarks)
   - Day 1-2: Baseline mode and scenarios
   - Day 3-4: Runner and reporter
   - Day 5: Full benchmark run

3. **Week 3:** I.3 (Invariant Suite)
   - Day 1-3: Verifier implementation
   - Day 4-5: Test suite and integration

**Total Estimated Effort:** 3 weeks (Medium as stated in roadmap)

---

## Exit Criteria (from Roadmap)

> Compression claims backed by reproducible benchmarks

Specifically:
- [ ] Per-step metrics captured for every operation
- [ ] Benchmark comparison shows quantified compression benefit
- [ ] All 7 catalytic invariants verified with automated tests
- [ ] Reports generated and stored in `_generated/benchmark_results/`
- [ ] CLI commands for running benchmarks and viewing metrics