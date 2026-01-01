# Vector ELO Roadmap

**Status:** Active (Research Phase)  
**Created:** 2026-01-02  
**Goal:** Implement systemic intuition via ELO scoring for vectors, files, and symbols. Prune short-term memory based on ELO. Free energy principle for AGS.

---

## Phase E.0: Research (SOTA ELO & Ranking Systems)

**Goal:** Survey state-of-the-art ranking/scoring systems to inform ELO implementation.

### Research Areas

#### 1. Classic ELO & Extensions
- [ ] **Original ELO** - Elo (1978): Chess rating system, pairwise comparisons
- [ ] **Glicko / Glicko-2** - Glickman (1999, 2001): Adds rating deviation (uncertainty)
- [ ] **TrueSkill** - Microsoft (2006): Bayesian skill rating, multiplayer games
- [ ] **TrueSkill 2** - Microsoft (2018): Improved for modern games

#### 2. X (Twitter) Algorithm (Open Source)
- [ ] **X Recommendation Algorithm** - Open sourced 2023
  - Heavy Ranker (neural network)
  - Trust and Safety Layer
  - "For You" timeline scoring
  - GitHub: `twitter/the-algorithm`
- [ ] **Key components:**
  - Tweet embedding similarity
  - User engagement prediction
  - Time decay factors
  - Trust scores (author reputation)

#### 3. Modern Ranking Systems
- [ ] **Google PageRank** - Brin & Page (1998): Link-based authority
- [ ] **YouTube Recommendation** - Deep neural network for watch time prediction
- [ ] **TikTok "For You"** - Interest graph + engagement signals
- [ ] **Reddit Hot/Best** - Wilson score confidence interval
- [ ] **Hacker News** - Gravity + time decay (simple, effective)

#### 4. Learning to Rank (LTR)
- [ ] **RankNet** - Burges et al. (2005): Pairwise neural ranking
- [ ] **LambdaRank** - Burges et al. (2006): Gradient-based ranking
- [ ] **LambdaMART** - Wu et al. (2010): Gradient boosted trees for ranking
- [ ] **BERT for Ranking** - Nogueira & Cho (2019): Cross-encoder ranking

#### 5. Free Energy Principle / Active Inference
- [ ] **Friston Free Energy** - Friston (2010): Predictive processing, surprise minimization
- [ ] **Active Inference** - Friston et al. (2017): Action as inference
- [ ] **Predictive Coding** - Clark (2013): Brain as prediction machine
- [ ] **Relevance Realization** - Vervaeke (2017): Salience detection, pruning

#### 6. Memory Pruning & Forgetting
- [ ] **Ebbinghaus Forgetting Curve** - Exponential decay
- [ ] **Spaced Repetition** - Leitner (1972): Optimal review intervals
- [ ] **Neural Episodic Control** - Pritzel et al. (2017): Differentiable memory
- [ ] **Episodic Memory in LLMs** - MemGPT, Generative Agents

### Deliverables
- [ ] Research summary document (10-20 pages)
- [ ] Annotated bibliography (key papers with notes)
- [ ] Comparison matrix (ELO vs Glicko vs TrueSkill vs LTR)
- [ ] Design decision: Which approach for AGS?

### Exit Criteria
- Clear understanding of SOTA ranking systems
- X algorithm reviewed and applicable patterns identified
- Decision on ELO formula (classic vs Glicko vs custom)
- Decision on memory pruning strategy (forgetting curve vs hard threshold)

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

## References

- **X Algorithm:** https://github.com/twitter/the-algorithm
- **ELO:** Elo (1978) - "The Rating of Chessplayers, Past and Present"
- **Glicko-2:** Glickman (2001) - "Parameter estimation in large dynamic paired comparison experiments"
- **TrueSkill:** Herbrich et al. (2006) - "TrueSkill: A Bayesian skill rating system"
- **Free Energy:** Friston (2010) - "The free-energy principle: a unified brain theory?"
- **Spaced Repetition:** Leitner (1972) - "Learn to learn"
