# VECTOR_ELO Lab Changelog

ELO scoring system implementation history.

---

## [3.7.48] - 2026-01-18

### E.1.2 COMPLETE - SessionAuditor MCP Integration

**Status:** SessionAuditor now wired into AGSMCPServer for complete file/symbol/search tracking.

**MCP Integration (E.1.2):**
- `CAPABILITY/MCP/server.py` - SessionAuditor wired into AGSMCPServer
  - `_init_session_auditor()`, `_end_session_audit()`, `_track_tool_access()`
  - Maps tool calls to ELO events: canon_read, context_search, codebook_lookup, cassette queries

**Verified:**
- session_auditor_available: True on server init
- Files, ADRs, symbols, and searches tracked correctly
- Session audit written to session_audit.jsonl on shutdown

---

## [3.7.47] - 2026-01-18

### Phase 7 COMPLETE - MCP Integration (E.1.1 + E.5)

**Status:** SearchLogger and EloRanker wired into SemanticMCPAdapter.

**MCP Integration:**
- SearchLogger logging all semantic searches with session attribution
- EloRanker annotating results with elo_score and elo_tier metadata

**Design Decision: ELO as Suggestion Only**
- ELO is metadata only - does NOT affect search ranking
- Formula: final_score = similarity (ELO has zero weight)
- Prevents echo chambers and "lost treasures"

---

## [3.7.46] - 2026-01-18

### Phase 7 MODULES COMPLETE - Vector ELO Implementation

All standalone modules implemented and tested across 4 parallel waves:

**Wave 1 - Logging Infrastructure (E.1):**
- search_logger.py, session_auditor.py, elo_db.py, critic.py
- elo_scores.db with 4 tables (vector/file/symbol/adr)

**Wave 2 - ELO Engine (E.2):**
- elo_engine.py with standard ELO update, Ebbinghaus forgetting curve, tier classification

**Wave 3 - Integration Tools (E.3-E.5):**
- prune_memory.py (archives VERY_LOW + 30 days stale)
- lite_pack.py (HIGH: full, MEDIUM: summary, LOW: pointer, VERY_LOW: exclude)
- elo_ranker.py (annotate_results, suggestion-only mode)

**Wave 4 - Visualization (E.6):**
- elo_dashboard.py CLI with interactive mode

**Implementation Stats:** 4,709 LOC across 9 subagents

---

## [3.2.2] - 2026-01-02

### Lane E: Initial Setup

**Added:**
- **Lane E: Vector ELO Scoring** - Systemic intuition prototype
- ROADMAP.md - 7-phase implementation plan (E.0 through E.6)

**Research Areas Defined:**
- Classic ELO & Extensions (Glicko, TrueSkill)
- X (Twitter) Algorithm, PageRank, Learning to Rank
- Free Energy Principle / Active Inference
- Memory Pruning & Forgetting (Ebbinghaus, Spaced Repetition)

---

*SCL/SPC/compression changelog entries moved to THOUGHT/LAB/COMPRESSION/CHANGELOG.md*
*Eigen-alignment changelog entries moved to THOUGHT/LAB/EIGEN_ALIGNMENT/CHANGELOG.md*
