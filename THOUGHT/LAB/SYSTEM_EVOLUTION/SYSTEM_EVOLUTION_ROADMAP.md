# System Evolution Roadmap (Phase 10)

**Status:** NOT STARTED (Blocked on Phase 9)
**Created:** 2026-01-25
**Goal:** Post-substrate optimization for performance, scale, and intelligence
**Priority:** P3 (low until core phases complete)

---

## Core Principles

### Post-Substrate Philosophy
Phase 10 assumes a stable, working substrate from Phases 1-9:
- CAS integrity (Phase 2)
- Deterministic chat (Phase 3)
- Catalytic restore (Phase 4)
- Vector/symbol integration (Phase 5)
- Cassette network (Phase 6)
- ELO scoring (Phase 7)
- Resident AI (Phase 8)
- Swarm architecture (Phase 9)

### Optimization vs. Functionality
This phase adds NO new core functionality - only:
- Performance improvements
- Scale enhancements
- UX improvements
- Research explorations

---

## Phase 10.1: Performance Foundation (Omega.1)

**Goal:** Optimize query performance and visibility.

### 10.1.1 Incremental Indexing (Omega.1.1)

**Problem:** Full vector index rebuilds are expensive.

**Solution:** Incremental updates based on file change detection.

**Tasks:**
- [ ] Implement file change detector (mtime/hash comparison)
- [ ] Add incremental vector update to cassette write path
- [ ] Track index staleness metrics
- [ ] Add `--incremental` flag to indexing commands
- [ ] Benchmark: target <10% of full rebuild time for <5% changes

**Exit Criteria:**
- [ ] Incremental indexing reduces rebuild time by 90%+
- [ ] Staleness metrics visible in dashboard
- [ ] Tests verify correctness vs. full rebuild

### 10.1.2 Query Result Caching (Omega.1.2)

**Problem:** Repeated queries hit the vector store unnecessarily.

**Solution:** LRU cache for semantic queries.

**Tasks:**
- [ ] Implement query hash computation (query + filters -> hash)
- [ ] Add LRU cache with configurable TTL
- [ ] Add cache invalidation on index update
- [ ] Track cache hit/miss metrics
- [ ] Benchmark: target >80% hit rate for repeated sessions

**Exit Criteria:**
- [ ] Cache reduces average query time by 50%+
- [ ] Cache invalidation is correct (no stale results)
- [ ] Hit/miss metrics logged

### 10.1.3 Compression Metrics Dashboard (Omega.1.3)

**Problem:** No visibility into compression effectiveness.

**Solution:** Dashboard showing compression metrics across system.

**Tasks:**
- [ ] Collect metrics: SCL compression ratio, SPC pointer density, ELO tier distribution
- [ ] Implement CLI dashboard (ASCII charts)
- [ ] Add export to JSON for external visualization
- [ ] Track trends over time (daily snapshots)

**Exit Criteria:**
- [ ] Dashboard shows current compression state
- [ ] Trends visible over 7+ days
- [ ] Export works for Grafana/etc.

---

## Phase 10.2: Scale & Governance (Omega.2)

**Goal:** Enable multi-cassette deployments and temporal queries.

### 10.2.1 Multi-Cassette Federation (Omega.2.1)

**Problem:** Single-cassette architecture limits scale.

**Solution:** Federated queries across multiple cassettes.

**Tasks:**
- [ ] Design federation protocol (cassette discovery, routing)
- [ ] Implement federated query dispatcher
- [ ] Merge and rank results from multiple cassettes
- [ ] Handle cassette partitioning strategies (by topic, time, etc.)
- [ ] Add federation health monitoring

**Exit Criteria:**
- [ ] Queries span multiple cassettes transparently
- [ ] Ranking is consistent across federated results
- [ ] Health monitoring detects failed cassettes

### 10.2.2 Temporal Queries (Time Travel) (Omega.2.2)

**Problem:** Can't query historical states.

**Solution:** Query cassettes at specific timestamps.

**Tasks:**
- [ ] Add timestamp parameter to query interface
- [ ] Implement point-in-time vector retrieval
- [ ] Support receipt chain replay for audits
- [ ] Add time-range queries (between T1 and T2)
- [ ] Benchmark: query latency vs. temporal depth

**Exit Criteria:**
- [ ] Historical queries return correct results
- [ ] Receipt chain replay produces identical state
- [ ] Time-range queries work for audit use cases

### 10.2.3 Receipt Compression (Omega.2.3)

**Problem:** Receipt chains grow indefinitely.

**Solution:** Compress old receipts while preserving verifiability.

**Tasks:**
- [ ] Design receipt Merkle tree pruning strategy
- [ ] Implement receipt aggregation (batch receipts into summaries)
- [ ] Preserve verifiability via Merkle proofs
- [ ] Add compression policy (age-based, ELO-based)
- [ ] Benchmark: space savings vs. verification latency

**Exit Criteria:**
- [ ] Receipts compress by 90%+ after 30 days
- [ ] Merkle proofs still verify correctly
- [ ] Compression policy is configurable

---

## Phase 10.3: Intelligence & UX (Omega.3)

**Goal:** Automated intelligence and improved user experience.

### 10.3.1 Automatic Symbol Extraction (Omega.3.1)

**Problem:** Symbols are manually defined.

**Solution:** Automatically extract symbols from high-ELO content.

**Tasks:**
- [ ] Implement entity extraction (NER) on governance documents
- [ ] Score extracted entities by frequency and context
- [ ] Propose new symbols for codebook
- [ ] Human-in-the-loop approval workflow
- [ ] Track symbol adoption metrics

**Exit Criteria:**
- [ ] System proposes 10+ useful symbols per week
- [ ] Approval workflow is smooth
- [ ] Adopted symbols improve compression

### 10.3.2 Smart Slice Prediction (Omega.3.2)

**Problem:** Users must manually specify slices.

**Solution:** Predict optimal slices based on query context.

**Tasks:**
- [ ] Train slice predictor on historical queries
- [ ] Implement context-aware slice suggestions
- [ ] Add confidence scores to predictions
- [ ] Fallback to manual specification when uncertain
- [ ] Benchmark: prediction accuracy vs. user satisfaction

**Exit Criteria:**
- [ ] Predictions are correct >80% of the time
- [ ] Users can override predictions easily
- [ ] Latency overhead is <100ms

### 10.3.3 Provenance Graph Visualization (Omega.3.3)

**Problem:** Receipt chains are hard to understand.

**Solution:** Visual graph of provenance relationships.

**Tasks:**
- [ ] Design graph schema (nodes = receipts, edges = dependencies)
- [ ] Implement D3.js or similar visualization
- [ ] Add filtering by time, entity, operation
- [ ] Support zoom/pan for large graphs
- [ ] Export to SVG/PNG

**Exit Criteria:**
- [ ] Visualization renders correctly for 1000+ receipts
- [ ] Filtering is responsive
- [ ] Export works

### 10.3.4 Zero-Knowledge Proofs Research (Omega.3.4)

**Problem:** Some audits require privacy.

**Solution:** Research ZK proofs for governance verification.

**Tasks:**
- [ ] Literature review: ZK-SNARKs, ZK-STARKs for audit
- [ ] Identify applicable governance use cases
- [ ] Prototype simple ZK proof (e.g., "file existed at time T")
- [ ] Evaluate computational overhead
- [ ] Document feasibility and trade-offs

**Exit Criteria:**
- [ ] Research report completed
- [ ] At least one prototype proof working
- [ ] Feasibility assessment documented

---

## Dependencies

| Dependency | Status | Impact |
|------------|--------|--------|
| Phase 5 (Vectors) | COMPLETE | Required for 10.1.1 |
| Phase 6 (Cassettes) | COMPLETE | Required for 10.2.1 |
| Phase 7 (ELO) | CORE COMPLETE | Required for 10.1.3 |
| Phase 9 (Swarm) | NOT STARTED | Soft dependency |

---

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Incremental index time | <10% of full rebuild | For <5% changes |
| Query cache hit rate | >80% | Repeated sessions |
| Compression dashboard | 3 metrics visible | SCL, SPC, ELO |
| Federation latency | <2x single cassette | Cross-cassette queries |
| Temporal query accuracy | 100% | Historical correctness |
| Receipt compression | 90%+ after 30 days | With verifiability |
| Symbol prediction | 10+/week adopted | Auto-extraction |
| Slice prediction | >80% accuracy | Context-aware |

---

## Changelog

| Date | Event |
|------|-------|
| 2026-01-25 | Initial roadmap created from AGS_ROADMAP_MASTER Phase 10 |

---

## References

- **Phase 10 Source:** [AGS_ROADMAP_MASTER.md](../../../AGS_ROADMAP_MASTER.md) lines 411-427
- **Cassette Network:** [CASSETTE_NETWORK_SPEC.md](../../../LAW/CANON/SEMANTIC/CASSETTE_NETWORK_SPEC.md)
- **Vector ELO:** [VECTOR_ELO_SPEC.md](../../../LAW/CANON/SEMANTIC/VECTOR_ELO_SPEC.md)
- **ZK-SNARKs:** Ben-Sasson et al. (2014) - "Succinct Non-Interactive Zero Knowledge"
- **ZK-STARKs:** Ben-Sasson et al. (2018) - "Scalable, transparent, and post-quantum secure"
