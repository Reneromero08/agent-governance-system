# Vector ELO Scoring System (Free Energy Principle)

**Status:** Design Spec  
**Created:** 2026-01-02  
**Purpose:** Implement systemic intuition via ELO scoring for vectors, files, and symbols based on usage patterns.

---

## Core Concept

**Free Energy Principle for AGS:**
- Vectors/files that are **accessed frequently** = **lower surprise** = **higher ELO**
- Higher ELO content gets **prioritized** in LITE packs, context assembly, and search results
- System learns what's "important" by observing its own behavior (systemic intuition)

**Analogy:** Like a brain's predictive processing—frequently-used patterns become "cheaper" to access.

---

## ELO Score Formula

```python
ELO_new = ELO_old + K * (outcome - expected)

where:
- K = learning rate (16 for new vectors, 8 for established)
- outcome = 1 (accessed), 0 (not accessed)
- expected = sigmoid(ELO_old / 400)  # probability of access
```

**Initial ELO:**
- **HIGH tier (1600):** LAW/CANON/*, AGENTS.md, NAVIGATION/MAPS/*
- **MEDIUM tier (1200):** CAPABILITY/SKILLS/*, LAW/CONTEXT/decisions/*
- **LOW tier (800):** THOUGHT/LAB/*, MEMORY/ARCHIVE/*, fixtures

---

## What Gets Scored

### 1. **Vectors (Embeddings)**
- Every semantic search increments ELO for retrieved vectors
- Top-K results get higher boost (+16 for rank 1, +8 for rank 2-5, +4 for rank 6-10)

### 2. **Files**
- Every file read (via `view_file`, `grep_search`, etc.) increments ELO
- Files referenced in successful runs get bonus ELO

### 3. **Symbols**
- Every `@Symbol` expansion increments ELO
- Symbols used in successful task completions get bonus ELO

### 4. **ADRs / Context Records**
- Every ADR read increments ELO
- ADRs cited in new ADRs get bonus ELO (citation graph)

---

## Logging Requirements

### 1. **MCP Server Logging**
Every search/retrieval must log:
```json
{
  "session_id": "uuid",
  "timestamp": "ISO8601",
  "tool": "semantic_search | grep_search | cortex_query",
  "query": "user query text",
  "results": [
    {"hash": "sha256", "file_path": "...", "rank": 1, "similarity": 0.85},
    ...
  ]
}
```

**Storage:** `NAVIGATION/CORTEX/_generated/search_log.jsonl` (append-only)

### 2. **Session Audit Logging**
Every session must log:
```json
{
  "session_id": "uuid",
  "agent_id": "antigravity | user",
  "start_time": "ISO8601",
  "end_time": "ISO8601",
  "files_accessed": ["path1", "path2", ...],
  "symbols_expanded": ["@Symbol1", "@Symbol2", ...],
  "adrs_read": ["ADR-027", ...],
  "search_queries": 42,
  "semantic_searches": 30,
  "keyword_searches": 12
}
```

**Storage:** `NAVIGATION/CORTEX/_generated/session_audit.jsonl` (append-only)

### 3. **ELO Update Log**
Every ELO update must log:
```json
{
  "timestamp": "ISO8601",
  "entity_type": "vector | file | symbol | adr",
  "entity_id": "hash | path | @Symbol | ADR-XXX",
  "elo_old": 1200,
  "elo_new": 1216,
  "delta": +16,
  "reason": "semantic_search_rank_1 | file_read | symbol_expansion | adr_citation"
}
```

**Storage:** `NAVIGATION/CORTEX/_generated/elo_updates.jsonl` (append-only)

---

## ELO Tier Thresholds

| Tier | ELO Range | LITE Pack Behavior | Example Content |
|------|-----------|-------------------|-----------------|
| **HIGH** | 1600+ | Include completely (full integrity) | LAW/CANON/*, AGENTS.md, frequently-used ADRs |
| **MEDIUM** | 1200-1599 | Summarize (signatures only) | CAPABILITY/SKILLS/*, less-used ADRs |
| **LOW** | 800-1199 | Omit with pointer | THOUGHT/LAB/*, MEMORY/ARCHIVE/* |
| **VERY LOW** | <800 | Exclude entirely | Unused fixtures, old logs |

---

## Implementation Phases

### Phase 1: Logging Infrastructure (P0)
- [ ] Add search logging to MCP server (`search_log.jsonl`)
- [ ] Add session audit logging (`session_audit.jsonl`)
- [ ] Add ELO update logging (`elo_updates.jsonl`)
- [ ] Create `NAVIGATION/CORTEX/_generated/elo_scores.db` (SQLite)
  - Tables: `vector_elo`, `file_elo`, `symbol_elo`, `adr_elo`

### Phase 2: ELO Calculation Engine (P1)
- [ ] Implement `elo_engine.py`:
  - `update_elo(entity_id, outcome, expected)` → new_elo
  - `get_elo(entity_id)` → current_elo
  - `get_top_k_by_elo(entity_type, k)` → ranked list
- [ ] Batch ELO updates (process `search_log.jsonl` → update `elo_scores.db`)
- [ ] Run daily (or per-session) to keep ELO scores fresh

### Phase 3: LITE Pack Integration (P2)
- [ ] Update `Engine/packer/lite.py` to query `elo_scores.db`
- [ ] Filter files by ELO tier:
  - HIGH (1600+): Include completely
  - MEDIUM (1200-1599): Summarize
  - LOW (<1200): Omit with pointer
- [ ] Add ELO metadata to pack manifest

### Phase 4: Search Result Ranking (P2)
- [ ] Update `semantic_search` to boost results by ELO
  - `final_score = similarity * 0.7 + (elo / 2000) * 0.3`
- [ ] Update `cortex_query` to sort by ELO (secondary sort after relevance)

### Phase 5: Visualization & Monitoring (P3)
- [ ] Build ELO dashboard (web UI):
  - Top 100 vectors by ELO
  - ELO distribution histogram
  - ELO trend over time (per file/symbol)
- [ ] Export to Prometheus/Grafana for monitoring

---

## Free Energy Principle Interpretation

**Prediction Error Minimization:**
- System predicts which vectors/files will be accessed (based on ELO)
- Actual access updates ELO (reduces prediction error)
- Over time, system learns "what matters" (systemic intuition)

**Surprise = -log(P(access)):**
- High ELO → High P(access) → Low surprise
- Low ELO → Low P(access) → High surprise

**Active Inference:**
- System actively prioritizes high-ELO content (LITE packs, search ranking)
- This creates a feedback loop: high-ELO content gets accessed more → ELO increases further

**Result:** System self-organizes around frequently-used patterns (free energy minimization).

---

## Success Metrics

- **ELO convergence:** After 100 sessions, ELO scores stabilize (variance <10%)
- **LITE pack accuracy:** 90%+ of accessed files are in LITE pack (high-ELO content)
- **Search efficiency:** 80%+ of top-5 search results are high-ELO
- **Token savings:** LITE packs are 80%+ smaller than FULL packs (due to ELO filtering)

---

## References

- **Free Energy Principle:** Friston (2010) - "The free-energy principle: a unified brain theory?"
- **Predictive Processing:** Clark (2013) - "Whatever next? Predictive brains, situated agents, and the future of cognitive science"
- **ELO Rating System:** Elo (1978) - "The Rating of Chessplayers, Past and Present"

---

## Next Steps

1. **Implement Phase 1** (Logging) - Add to Lane P (LLM Packer) or Lane M (Cassette Network)
2. **Design ELO schema** - Define SQLite tables for `elo_scores.db`
3. **Prototype ELO engine** - Test on small dataset (100 vectors, 10 sessions)
4. **Integrate with LITE packer** - Use ELO scores to filter content
5. **Monitor and tune** - Adjust K (learning rate) and thresholds based on results
