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

## Phase E.1: Logging Infrastructure (P0)

**Goal:** Log everything to enable ELO calculation.

### Tasks
- [ ] **E.1.1**: Add search logging to MCP server
  - Log: session_id, timestamp, tool, query, results (hash, path, rank, similarity)
  - Storage: `NAVIGATION/CORTEX/_generated/search_log.jsonl`
- [ ] **E.1.2**: Add session audit logging
  - Log: session_id, agent_id, start/end time, files accessed, symbols expanded
  - Storage: `NAVIGATION/CORTEX/_generated/session_audit.jsonl`
- [ ] **E.1.3**: Add critic.py check for search protocol compliance
  - Flag: keyword search for conceptual queries
  - Flag: missing session_id
- [ ] **E.1.4**: Create `elo_scores.db` (SQLite)
  - Tables: `vector_elo`, `file_elo`, `symbol_elo`, `adr_elo`
  - Schema: `entity_id, elo_score, access_count, last_accessed, created_at`

### Exit Criteria
- All MCP searches are logged
- Session audits capture all file/symbol access
- critic.py enforces search protocol
- elo_scores.db schema ready

---

## Phase E.2: ELO Calculation Engine (P1)

**Goal:** Implement the ELO scoring algorithm.

### Tasks
- [ ] **E.2.1**: Implement `elo_engine.py`:
  - `update_elo(entity_id, outcome, expected)` → new_elo
  - `get_elo(entity_id)` → current_elo
  - `get_top_k_by_elo(entity_type, k)` → ranked list
  - `decay_elo(days_since_access)` → apply forgetting curve
- [ ] **E.2.2**: Batch ELO update script
  - Process `search_log.jsonl` → update `elo_scores.db`
  - Run daily or per-session
- [ ] **E.2.3**: Implement ELO tier classification
  - HIGH (1600+), MEDIUM (1200-1599), LOW (800-1199), VERY LOW (<800)
- [ ] **E.2.4**: Add ELO update logging
  - Log: timestamp, entity_id, elo_old, elo_new, delta, reason
  - Storage: `NAVIGATION/CORTEX/_generated/elo_updates.jsonl`

### Exit Criteria
- ELO scores update correctly on access
- Forgetting curve decays unused entities
- Tier classification matches expected behavior
- All updates logged

---

## Phase E.3: Memory Pruning (Short-Term Memory) (P1)

**Goal:** Prune short-term memory based on ELO.

### Tasks
- [ ] **E.3.1**: Define short-term memory scope
  - INBOX/reports/* (session reports)
  - THOUGHT/LAB/*/scratch/* (experiment scratch files)
  - NAVIGATION/CORTEX/_generated/*.jsonl (logs)
- [ ] **E.3.2**: Implement pruning policy
  - VERY LOW ELO (<800) + not accessed in 30 days → archive
  - LOW ELO (800-1199) + not accessed in 14 days → compress (summarize)
  - MEDIUM+ ELO → retain
- [ ] **E.3.3**: Implement pruning script (`prune_memory.py`)
  - List candidates for pruning
  - Archive to `MEMORY/ARCHIVE/pruned/` with manifest
  - Log pruning actions
- [ ] **E.3.4**: Add pruning report to session audit
  - Show: entities pruned, space saved, ELO distribution

### Exit Criteria
- Short-term memory is bounded (not infinite growth)
- Pruned content is archived (not deleted)
- Pruning is logged and auditable
- High-ELO content never pruned

---

## Phase E.4: LITE Pack Integration (P2)

**Goal:** Use ELO scores to filter LITE pack content.

### Tasks
- [ ] **E.4.1**: Update `Engine/packer/lite.py` to query `elo_scores.db`
- [ ] **E.4.2**: Filter by ELO tier:
  - HIGH (1600+): Include completely
  - MEDIUM (1200-1599): Summarize (signatures only)
  - LOW (<1200): Omit with pointer
- [ ] **E.4.3**: Add ELO metadata to pack manifest
  - Show: file_path, elo_score, tier
- [ ] **E.4.4**: Benchmark LITE pack size vs FULL pack
  - Goal: 80%+ smaller due to ELO filtering

### Exit Criteria
- LITE packs use ELO for content selection
- Pack manifest shows ELO metadata
- 80%+ size reduction achieved
- High-ELO content always included

---

## Phase E.5: Search Result Ranking (P2)

**Goal:** Boost search results by ELO score.

### Tasks
- [ ] **E.5.1**: Update `semantic_search` to boost by ELO
  - `final_score = similarity * 0.7 + (elo / 2000) * 0.3`
- [ ] **E.5.2**: Update `cortex_query` to sort by ELO (secondary)
- [ ] **E.5.3**: Add ELO to search result metadata
  - Show: hash, path, similarity, elo_score, tier
- [ ] **E.5.4**: Benchmark search quality
  - Goal: 80%+ of top-5 results are high-ELO

### Exit Criteria
- Search results boosted by ELO
- High-ELO content surfaces first
- Search quality improved (measured by relevance)

---

## Phase E.6: Visualization & Monitoring (P3)

**Goal:** Dashboard for ELO visibility.

### Tasks
- [ ] **E.6.1**: Build ELO dashboard (web UI or CLI)
  - Top 100 entities by ELO
  - ELO distribution histogram
  - ELO trend over time
- [ ] **E.6.2**: Export to Prometheus/Grafana
  - Metrics: elo_avg, elo_max, elo_min, access_count
- [ ] **E.6.3**: Add ELO alerts
  - Alert: Entity drops below threshold
  - Alert: Memory pruning approaching limit

### Exit Criteria
- Dashboard shows ELO distribution
- Metrics exported to monitoring stack
- Alerts configured

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
