# Vector ELO Roadmap

**Status:** Active (Implementation Phase)
**Created:** 2026-01-02
**Goal:** Implement systemic intuition via ELO scoring for vectors, files, and symbols. Prune short-term memory based on ELO. R-gated to prevent echo chambers.

---

## Core Formulas

### ELO Update (Standard)
```python
expected = 1 / (1 + 10**((opponent_elo - entity_elo) / 400))
new_elo = old_elo + K * (outcome - expected)

# K = 16 (new entities), 8 (established)
# outcome = 1 (accessed), 0 (not accessed)
```

### Forgetting Curve (Ebbinghaus)
```python
retention = e^(-days_since_access / half_life)
decayed_elo = base_elo + (current_elo - base_elo) * retention

# half_life = 30 days (tunable)
# base_elo = 800 (floor)
```

### R-Gated ELO (Echo Chamber Prevention)
```python
# Only boost ELO for R-passing content
if R > threshold:
    elo_delta = K * (outcome - expected)
else:
    elo_delta = K * (outcome - expected) * 0.25  # Penalize low-R access
```

---

## Phase E.1: Logging Infrastructure (P0) - MODULES DONE

**Goal:** Log everything to enable ELO calculation.

### Tasks
- [x] **E.1.1**: SearchLogger module (`CAPABILITY/PRIMITIVES/search_logger.py`)
  - Log: session_id, timestamp, tool, query, results (hash, path, rank, similarity)
  - Storage: `NAVIGATION/CORTEX/_generated/search_log.jsonl`
  - [ ] **PENDING:** Wire into MCP server search tools
- [x] **E.1.2**: SessionAuditor module (`CAPABILITY/PRIMITIVES/session_auditor.py`)
  - Log: session_id, agent_id, start/end time, files accessed, symbols expanded
  - Storage: `NAVIGATION/CORTEX/_generated/session_audit.jsonl`
  - [ ] **PENDING:** Wire into MCP session handling
- [x] **E.1.3**: critic.py check for search protocol compliance (`CAPABILITY/TOOLS/critic.py`)
  - Flag: keyword search for conceptual queries
  - Flag: missing session_id
- [x] **E.1.4**: `elo_scores.db` created (`NAVIGATION/CORTEX/_generated/elo_scores.db`)
  - Tables: `vector_elo`, `file_elo`, `symbol_elo`, `adr_elo`
  - Access layer: `CAPABILITY/PRIMITIVES/elo_db.py`

### Exit Criteria
- [ ] All MCP searches are logged (pending integration)
- [ ] Session audits capture all file/symbol access (pending integration)
- [x] critic.py enforces search protocol
- [x] elo_scores.db schema ready

---

## Phase E.2: ELO Calculation Engine (P1) - DONE

**Goal:** Implement the ELO scoring algorithm.

### Tasks
- [x] **E.2.1**: `elo_engine.py` implemented (`CAPABILITY/PRIMITIVES/elo_engine.py`)
  - `update_elo(entity_type, entity_id, outcome, opponent_elo)` → new_elo
  - `get_tier(elo)` → HIGH/MEDIUM/LOW/VERY_LOW
  - `decay_elo(entity_type, entity_id, days_since_access)` → decayed_elo
  - K=16 new, K=8 established; half_life=30 days; floor=800
- [x] **E.2.2**: Batch ELO update method
  - `process_search_log(log_path, processed_until)` → update elo_scores.db
  - Supports resume from timestamp
- [x] **E.2.3**: ELO tier classification
  - HIGH (1600+), MEDIUM (1200-1599), LOW (800-1199), VERY_LOW (<800)
- [x] **E.2.4**: ELO update logging
  - Logs to `NAVIGATION/CORTEX/_generated/elo_updates.jsonl`

### Exit Criteria
- [x] ELO scores update correctly on access
- [x] Forgetting curve decays unused entities
- [x] Tier classification matches expected behavior
- [x] All updates logged

---

## Phase E.3: Memory Pruning (Short-Term Memory) (P1) - DONE

**Goal:** Prune short-term memory based on ELO.

### Tasks
- [x] **E.3.1**: Short-term memory scope defined in `prune_memory.py`
  - INBOX/reports/* (session reports)
  - THOUGHT/LAB/*/scratch/* (experiment scratch files)
  - NAVIGATION/CORTEX/_generated/*.jsonl (logs, excluding elo_scores.db)
- [x] **E.3.2**: Pruning policy implemented
  - VERY_LOW ELO (<800) + not accessed in 30 days → archive
  - LOW ELO (800-1199) + not accessed in 14 days → flag for compress
  - MEDIUM+ ELO → retain
- [x] **E.3.3**: `prune_memory.py` (`CAPABILITY/TOOLS/prune_memory.py`)
  - `list_prune_candidates()` → {archive, compress, retain}
  - `archive_file()` → moves to `MEMORY/ARCHIVE/pruned/` with manifest
  - `execute_pruning(dry_run)` → dry_run mode supported
  - `generate_report()` → human-readable report
- [x] **E.3.4**: Pruning report available via `generate_report()`

### Exit Criteria
- [x] Short-term memory is bounded (not infinite growth)
- [x] Pruned content is archived (not deleted)
- [x] Pruning is logged and auditable
- [x] High-ELO content never pruned

---

## Phase E.4: LITE Pack Integration (P2) - DONE

**Goal:** Use ELO scores to filter LITE pack content.

### Tasks
- [x] **E.4.1**: `lite_pack.py` created (`CAPABILITY/TOOLS/lite_pack.py`)
  - Standalone LitePackGenerator class (not wired to Engine/packer)
  - Blocks INBOX/, THOUGHT/LAB/, _generated/ from packs
- [x] **E.4.2**: Filter by ELO tier:
  - HIGH (1600+): Include completely
  - MEDIUM (1200-1599): Summarize (Python/JS signatures, markdown headers)
  - LOW (800-1199): Pointer only
  - VERY_LOW (<800): Excluded
- [x] **E.4.3**: ELO metadata in pack manifest
  - Shows: file_path, elo_score, tier, blocked_paths
- [x] **E.4.4**: Token savings estimation via `estimate_token_savings()`

### Exit Criteria
- [x] LITE packs use ELO for content selection
- [x] Pack manifest shows ELO metadata
- [x] Compression ratio calculated
- [x] High-ELO content always included

---

## Phase E.5: Search Result Ranking (P2) - MODULES DONE

**Goal:** Boost search results by ELO score.

### Tasks
- [x] **E.5.1**: EloRanker module (`CAPABILITY/TOOLS/elo_ranker.py`)
  - `compute_final_score(similarity, elo)` = similarity * 0.7 + (elo / 2000) * 0.3
  - `boost_semantic_search(results)` → re-ranked results
  - [ ] **PENDING:** Wire into MCP semantic_search
- [x] **E.5.2**: `boost_cortex_query(results)` → ELO as secondary sort
  - [ ] **PENDING:** Wire into MCP cortex_query
- [x] **E.5.3**: ELO in result metadata (RankedResult dataclass)
  - Shows: file_path, content_hash, similarity, elo_score, elo_tier, final_score, rank
- [x] **E.5.4**: `get_quality_stats(results)` for benchmarking

### Exit Criteria
- [x] Search ranking logic implemented
- [ ] Search results boosted by ELO (pending MCP integration)
- [x] Quality stats available for measurement

---

## Phase E.6: Visualization & Monitoring (P3) - CLI DONE

**Goal:** Dashboard for ELO visibility.

### Tasks
- [x] **E.6.1**: CLI dashboard (`CAPABILITY/TOOLS/elo_dashboard.py`)
  - `display_top_entities(entity_type, limit)` → top N by ELO
  - `display_histogram(entity_type)` → ASCII histogram
  - `display_tier_summary(entity_type)` → tier breakdown
  - `display_recent_activity(limit)` → recently accessed
  - `display_recent_updates(limit)` → from elo_updates.jsonl
  - `run_interactive()` → REPL mode
- [ ] **E.6.2**: Export to Prometheus/Grafana (not implemented)
- [ ] **E.6.3**: Add ELO alerts (not implemented)

### Exit Criteria
- [x] Dashboard shows ELO distribution
- [ ] Metrics exported to monitoring stack (future)
- [ ] Alerts configured (future)

---

## Phase E.X: Eigenvalue Alignment Protocol (Research Complete)

**Status:** VALIDATED (2026-01-08)
**Goal:** Cross-model semantic alignment via eigenvalue spectrum invariance.

### Discovery

The **eigenvalue spectrum** of an anchor word distance matrix is invariant across embedding models (r = 0.99+), even when raw distance matrices are uncorrelated or inverted.

| Model Pair | Raw Distance Correlation | Eigenvalue Correlation |
|------------|-------------------------|------------------------|
| MiniLM ↔ E5-large | -0.05 | **0.9869** |
| MiniLM ↔ MPNET | 0.914 | 0.9954 |
| MiniLM ↔ BGE | 0.277 | 0.9895 |
| MiniLM ↔ GTE | 0.198 | 0.9865 |

### Proven Method

1. Compute squared distance matrix D² for anchor words
2. Apply classical MDS: B = -½ J D² J (double-centered Gram)
3. Eigendecompose: B = VΛV^T
4. Get MDS coordinates: X = V√Λ
5. Procrustes rotation: R = argmin ||X₁R - X₂||
6. Align new points via Gower's out-of-sample formula

### Proof Result

- Raw MDS similarity: -0.0053
- After Procrustes alignment: **0.8377**
- Improvement: **+0.8430**

### Existing Artifacts

| File | Description |
|------|-------------|
| `experiments/semantic_anchor_test.py` | Cross-model distance matrix testing |
| `experiments/invariant_search.py` | Invariant discovery |
| `experiments/eigen_alignment_proof.py` | MDS + Procrustes proof |
| `research/cassette-network/01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md` | Proof report |
| `research/cassette-network/01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md` | Hypothesis doc |
| `research/cassette-network/OPUS_EIGEN_SPECTRUM_ALIGNMENT_PROTOCOL_PACK.md` | Full protocol spec |

### Future Work (E.X.1)

- [ ] **E.X.1.1**: Implement full protocol per OPUS pack spec
  - Protocol message types: ANCHOR_SET, SPECTRUM_SIGNATURE, ALIGNMENT_MAP
  - CLI: `anchors build`, `signature compute`, `map fit`, `map apply`
- [ ] **E.X.1.2**: Benchmark with 8/16/32/64 anchor sets
- [ ] **E.X.1.3**: Test neighborhood overlap@k on held-out set
- [ ] **E.X.1.4**: Compare with vec2vec (arXiv:2505.12540) neural approach
- [ ] **E.X.1.5**: Integrate as cassette handshake artifact

### Related Papers

- arXiv:2405.07987 - Platonic Representation Hypothesis
- arXiv:2505.12540 - vec2vec (neural approach)

---

## Dependencies

- **Lane P (LLM Packer):** Phase E.4 depends on packer updates
- **Lane M (Cassette Network):** ELO scores stored in cassettes
- **Lane S (SPECTRUM):** ELO logging integrated with audit trail

---

## Success Metrics

| Metric | Target |
|--------|--------|
| ELO convergence | Variance <10% after 100 sessions |
| LITE pack accuracy | 90%+ accessed files are high-ELO |
| Search efficiency | 80%+ top-5 results are high-ELO |
| Token savings | 80%+ smaller LITE packs |
| Memory bound | Short-term memory <10GB |

---

## Changelog

| Date | Event |
|------|-------|
| 2026-01-11 | **Phase 5 COMPLETE** - All 529 tests pass, Global DoD met (tests/receipts/reports) |
| 2026-01-11 | Phase 5.3.6: PAPER_SPC.md research skeleton |
| 2026-01-11 | Phase 5.3.5: SPC semantic density proof harness |
| 2026-01-11 | Phase 5.3.4: TOKENIZER_ATLAS with single-token enforcement |
| 2026-01-11 | Phase 5.3.1-5.3.3: Created comprehensive test suites (215 tests) |
| 2026-01-09 | Phase 5.2.5: SCL CLI implementation |
| 2026-01-08 | Phase 5.2.1-5.2.4: SCL macro grammar, codebook, resolution, validator |
| 2026-01-07 | Phase 5.1: Vector embeddings and semantic discovery infrastructure |

---

## References

- **ELO:** Elo (1978) - "The Rating of Chessplayers, Past and Present"
- **Forgetting Curve:** Ebbinghaus (1885) - Exponential memory decay
- **Free Energy:** Friston (2010) - "The free-energy principle: a unified brain theory?"
- **R Formula:** See `THOUGHT/LAB/FORMULA/research/questions/INDEX.md` - log(R) = -F + const proven
