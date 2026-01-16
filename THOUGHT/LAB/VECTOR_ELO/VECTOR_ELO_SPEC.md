# Vector ELO Scoring System (R-Gated)

**Status:** Design Spec
**Created:** 2026-01-02
**Purpose:** Implement systemic intuition via ELO scoring for vectors, files, and symbols. R-gated to prevent echo chambers.

---

## Core Concept

**ELO tracks usage. R gates truth.**

- ELO measures what you ACCESS (pragmatic importance)
- R measures what is TRUE (semantic quality) - see `THOUGHT/LAB/FORMULA/`
- Combined: High R + High ELO = true AND important content

**Echo Chamber Prevention:**
Low R + High ELO = frequently accessed noise. R-gating prevents this from dominating.

---

## ELO Score Formula

### Standard ELO Update
```python
expected = 1 / (1 + 10**((opponent_elo - entity_elo) / 400))
new_elo = old_elo + K * (outcome - expected)

# K = 16 (new entities), 8 (established)
# outcome = 1 (accessed), 0 (not accessed)
```

### Forgetting Curve (Ebbinghaus Decay)
```python
retention = exp(-days_since_access / half_life)
decayed_elo = base_elo + (current_elo - base_elo) * retention

# half_life = 30 days
# base_elo = 800 (floor)
```

### R-Gated ELO Update
```python
if R > threshold:
    elo_delta = K * (outcome - expected)      # Full boost
else:
    elo_delta = K * (outcome - expected) * 0.25  # Penalized (low-truth access)
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

See `VECTOR_ELO_ROADMAP.md` for full details.

### Phase E.1: Logging Infrastructure (P0)
- [ ] Add search logging to MCP server (`search_log.jsonl`)
- [ ] Add session audit logging (`session_audit.jsonl`)
- [ ] Create `elo_scores.db` (SQLite)

### Phase E.2: ELO Calculation Engine (P1)
- [ ] Implement `elo_engine.py` with:
  - Standard ELO update
  - Forgetting curve decay
  - R-gated updates (penalize low-R access)
- [ ] Batch processing from logs

### Phase E.3: Memory Pruning (P1)
- [ ] Define short-term memory scope
- [ ] Prune by: Low ELO + Low R + stale = archive
- [ ] Never prune High R content

### Phase E.4: LITE Pack Integration (P2)
- [ ] Filter by ELO tier (HIGH/MEDIUM/LOW)
- [ ] R-gate included content

### Phase E.5: Search Result Ranking (P2)
- [ ] `final_score = R_gate(content) * (similarity * 0.7 + elo_norm * 0.3)`

### Phase E.6: Visualization (P3)
- [ ] ELO dashboard with R correlation view

---

## R vs ELO: Complementary Signals

| R Value | ELO Value | Interpretation | Action |
|---------|-----------|----------------|--------|
| High R + High ELO | True AND frequently used | **Best content** | Prioritize |
| High R + Low ELO | True but rarely accessed | Latent value | Keep |
| Low R + High ELO | Frequently accessed noise | **Echo chamber** | Penalize |
| Low R + Low ELO | Noise you don't use | Prune candidate | Archive |

**Key Insight:** R is axiomatically necessary (proven uniqueness via A1-A4). ELO is a heuristic. Use R to gate what ELO tracks.

---

## Free Energy Principle Interpretation

**Already proven:** log(R) = -F + const (see Q9 in FORMULA research)

**Prediction Error Minimization:**
- System predicts which vectors/files will be accessed (based on ELO)
- Actual access updates ELO (reduces prediction error)
- R-gating ensures prediction targets TRUTH, not just frequency

**Surprise = -log(P(access)):**
- High ELO → High P(access) → Low surprise
- Low ELO → Low P(access) → High surprise

**Active Inference:**
- System actively prioritizes high-ELO content (LITE packs, search ranking)
- R-gating prevents feedback loops from amplifying noise

**Result:** System self-organizes around frequently-used TRUE patterns.

---

## Success Metrics

- **ELO convergence:** After 100 sessions, ELO scores stabilize (variance <10%)
- **LITE pack accuracy:** 90%+ of accessed files are in LITE pack (high-ELO content)
- **Search efficiency:** 80%+ of top-5 search results are high-ELO
- **Token savings:** LITE packs are 80%+ smaller than FULL packs (due to ELO filtering)

---

## References

- **ELO:** Elo (1978) - "The Rating of Chessplayers, Past and Present"
- **Forgetting Curve:** Ebbinghaus (1885) - Exponential memory decay
- **Free Energy:** Friston (2010) - "The free-energy principle: a unified brain theory?"
- **R Formula:** `THOUGHT/LAB/FORMULA/research/questions/INDEX.md` - log(R) = -F + const proven

---

## Next Steps

1. **Implement E.1** (Logging Infrastructure) - see VECTOR_ELO_ROADMAP.md
2. **Implement E.2** (ELO Engine with R-gating + forgetting curve)
3. **Integrate E.3** (Memory pruning based on R x ELO)
4. **Monitor and tune** - Adjust K, half_life, R threshold based on results
