<!-- CONTENT_HASH: PENDING -->

# CAT_CHAT: Full Technical Report

**Last updated:** 2026-05-17
**Scope:** Analysis of CAT_CHAT as documented in the markdown files located in
`THOUGHT/LAB/CAT_CHAT/`, including its contract, roadmap, specs, and
relationship to the main AGS system.

---

## 1. What It Is (from the docs)

The canonical definition lives in `CAT_CHAT_README_1.1.md`:

> CAT_CHAT is a deterministic chat substrate: models write compact, structured
> messages that reference canonical material via **symbols**, and workers
> expand only **bounded slices** of source text when needed. The system is
> designed to be reproducible, auditable, and fail-closed.

The roadmap (`CAT_CHAT_ROADMAP_2.0.md`) expands on the philosophy:

> CAT = Catalytic. This is not just bounded chat -- it's chat that operates on
> catalytic computing principles:
> - **Clean Space:** Bounded context window (tokens)
> - **Catalytic Space:** Large disk state that MUST restore exactly after use
> - **Compression:** Symbols/pointers instead of full content (56,370x for
>   a single symbol)
> - **Verification:** Every expansion produces hash-verified receipts
> - **Fail-Closed:** Restoration failure = hard exit, never silent

It lives in `THOUGHT/LAB/CAT_CHAT/` as a sandboxed research project, per
`GRADUATION_PATH.md`:

> **Status:** FUTURE REFERENCE -- CAT_CHAT stays in LAB for now.

---

## 2. The Documentation Ecosystem

### 2.1 Root-Level Docs

The authoritative, actively maintained docs at the CAT_CHAT root:

| File | What It Covers |
|------|----------------|
| `CAT_CHAT_README_1.1.md` | Overview, core concepts, quickstart, project layout, ground rules |
| `CAT_CHAT_ROADMAP_2.0.md` | 718-line master plan: 10 phases (A-J), status tracking, research validation, graduation criteria, mathematical foundation |
| `CAT_CHAT_CHANGELOG.md` | 1784-line version history from v1.0.0 through v1.3.1 |
| `PHASE_J_IMPLEMENTATION_PLAN.md` | Detailed plan for the recursive E-score hierarchy (Phase J) |

### 2.2 The `docs/` Directory

Living support documents:

| File | Content |
|------|---------|
| `CAT_CHAT_CONTRACT.md` | **Core contract**: immutable JSON schemas for Section, Symbol, Message, Expansion, Receipt; budget definitions; error policy; determinism requirements |
| `CAT_CHAT_USAGE_GUIDE.md` | 323-line CLI walkthrough for Windows PowerShell from fresh clone through bundle/receipt/trust operations |
| `GRADUATION_PATH.md` | 3 graduation options, schema mapping, graduation criteria |
| `WRITE_ISOLATION.md` | Isolation model: reads from main AGS cassettes, writes only to `_generated/` |
| `verification_context_management.md` | Priority tiers, truncation rules, fail-closed design decisions |
| `DATABASE_MAINTENANCE.md` | Schema notes and maintenance guidance |
| `MESSAGE_STORAGE_ANALYSIS.md` | Analysis of message storage patterns |
| `ADR-attestation.md` | Architecture Decision Record for the attestation protocol |

### 2.3 The `docs/specs/` Directory

Four formal, versioned specifications:

| Spec | Version | Coverage |
|------|---------|----------|
| `BUNDLE_SPEC.md` | 5.0.0 | Bundle format, hashing, completeness gates |
| `RECEIPT_SPEC.md` | 1.0.0 | Receipt format, chain integrity, Merkle root |
| `TRUST_SPEC.md` | 1.0.0 | Trust policies, validator pinning, scope-based attestation |
| `EXECUTION_SPEC.md` | 1.0.0 | Execution semantics, fail-closed behavior, exit codes |

Core principles shared across all specs (from `SPEC_INDEX.md`):
1. Determinism: same inputs produce byte-identical outputs
2. Boundedness: all artifacts have explicit size limits (no `slice=ALL`)
3. Fail-Closed: invalid state halts execution, never continues silently
4. Hash Verification: all content verified via SHA-256
5. Canonical JSON: `sort_keys=True, separators=(",",":")` with trailing newline

### 2.4 The `archive/` Directory

50+ historical docs preserved for audit: older roadmaps (v1, v1.1), commit
plans, phase completion reports, and status snapshots dating back to December
2025.

### 2.5 The `docs/specs/` Directory

Four formal, versioned specifications:

| Spec | Version | Coverage |
|------|---------|----------|
| `BUNDLE_SPEC.md` | 5.0.0 | Bundle format, hashing, completeness gates |
| `RECEIPT_SPEC.md` | 1.0.0 | Receipt format, chain integrity, Merkle root |
| `TRUST_SPEC.md` | 1.0.0 | Trust policies, validator pinning, scope-based attestation |
| `EXECUTION_SPEC.md` | 1.0.0 | Execution semantics, fail-closed behavior, exit codes |

Core principles shared across all specs (from `SPEC_INDEX.md`):
1. Determinism: same inputs produce byte-identical outputs
2. Boundedness: all artifacts have explicit size limits (no `slice=ALL`)
3. Fail-Closed: invalid state halts execution, never continues silently
4. Hash Verification: all content verified via SHA-256
5. Canonical JSON: `sort_keys=True, separators=(",",":")` with trailing newline

---

## 3. The Core Contract (CAT_CHAT_CONTRACT.md)

The contract defines 5 core objects with fail-closed semantics.

### 3.1 Section

A canonical content unit extracted from source files:

```json
{
  "section_id": "sha256_hash",
  "file_path": "string",
  "heading_path": ["heading1", "heading2"],
  "line_start": 0,
  "line_end": 42,
  "content_hash": "sha256_hash"
}
```

**Constraints:**
- `section_id` = SHA-256 of `file_path:line_start:line_end:content_hash`
- `file_path` = absolute path relative to repo root
- `line_start` <= `line_end`
- `content_hash` = SHA-256 of exact content (normalized line endings)

**Fail-closed:** missing file -> FAIL, invalid line range -> FAIL, content
hash mismatch -> FAIL.

### 3.2 Symbol

A compact reference to a Section or file:

```json
{
  "symbol_id": "@NAMESPACE/NAME",
  "target_type": "SECTION|FILE|HEADING",
  "target_ref": "section_id_or_file_path",
  "default_slice_policy": "lines[a:b]"
}
```

**Constraints:** starts with `@`, valid `target_ref`, valid slice.

**Namespace convention:** `@CANON/...`, `@CONTRACTS/...`, `@TOOLS/...`,
`@SKILLS/...`

**Fail-closed:** invalid format -> FAIL, unknown target -> FAIL, unresolvable
target -> FAIL.

### 3.3 Message

Model output requesting work with explicit resource references:

```json
{
  "intent": "string",
  "refs": ["@Symbol1", "@Symbol2"],
  "ops": [
    {
      "type": "READ|WRITE|EXECUTE",
      "target": "@Symbol",
      "params": {}
    }
  ],
  "budgets": {
    "max_symbols": 10,
    "max_sections": 5,
    "max_bytes_expanded": 10000,
    "max_expands_per_step": 3
  },
  "required_outputs": ["output1", "output2"]
}
```

**Fail-closed:** empty intent -> FAIL, invalid symbol -> FAIL, missing budgets
-> FAIL, budget breach -> FAIL.

### 3.4 Expansion

Bounded content retrieval for a Symbol or Section:

```json
{
  "run_id": "uuid",
  "symbol_or_section_id": "@Symbol or section_id",
  "slice": "lines[0:100]",
  "content_hash": "sha256_hash",
  "payload_ref": "path_or_hash"
}
```

**Canonical slice forms:** `lines[a:b]`, `chars[a:b]`, `head(n)`, `tail(n)`.

**Fail-closed:** invalid slice syntax -> FAIL, exceeds bounds -> FAIL, hash
mismatch -> FAIL, `slice=ALL` -> FAIL (unbounded expansion forbidden).

### 3.5 Receipt

Immutable record of an execution step:

```json
{
  "run_id": "uuid",
  "step_id": "step-uuid",
  "expanded": [{"symbol_or_section_id": "@Symbol", "slice": "lines[0:100]", "content_hash": "hash"}],
  "actions": [{"type": "READ|WRITE|EXECUTE", "target": "@Symbol", "status": "SUCCESS|FAILURE", "result": {}}],
  "outputs": {"output1": "value"},
  "status": "SUCCESS|FAILURE|PARTIAL"
}
```

**Fail-closed:** missing run_id/step_id -> FAIL, incomplete expanded records
-> FAIL, missing required_outputs -> FAIL, must be append-only (immutable
after creation).

---

## 4. How It Works (from the docs)

### 4.1 The Catalytic Loop (ROADMAP v2.0, Phase C)

The roadmap explicitly calls this the core of the system:

> This is THE core catalytic behavior. Without this, nothing is actually
> catalytic.

The auto-controlled context loop:

1. **Context Budget & Working Set** (C.1): Define clean space budget
   (max tokens for working set). Track working_set vs pointer_set. Hard fail
   if working_set exceeds budget.

2. **E-Score Eviction** (C.2): On budget exceeded, compute E-score of each
   working_set item vs current query. Evict lowest-E items to pointer_set
   until under budget. Log eviction events to session_events (hash-chained).

3. **E-Score Hydration** (C.3): On each query, compute E-score of query vs
   all pointer_set items. Hydrate high-E items (above threshold) into
   working_set. Hydration is bounded -- max N items, respects budget.

4. **Turn Compression** (C.4): After response, old turns (beyond window)
   compress to hash pointers. Full turn content stored in catalytic space
   (session_events). Only hash pointer + summary remains in working_set.

5. **Catalytic Chat Loop** (C.5): Wire together: query -> hydrate -> assemble
   -> LLM -> compress -> evict. Session capsule logs every step for
   deterministic replay.

### 4.2 The E-Score (from ROADMAP + FORMULA research)

The mathematical heart of the system. From the roadmap:

> E = |<query_vec | item_vec>|^2 (Born rule)

Validated in research question Q44 with r=0.999 empirical correlation.

### 4.3 Hierarchical Memory (ROADMAP Phase J)

For scaling to 100K+ turns, uses a recursive centroid tree:

```
L3: 10 centroids      (each = average of 10,000 turn vectors)
L2: 100 centroids     (each = average of 1,000 turn vectors)
L1: 1,000 centroids   (each = average of 100 turn vectors)
L0: 100,000 vectors   (actual turn embeddings + content)
```

Same E-score function at every level. A centroid is just a mean vector:

> centroid = mean([child_1_vec, child_2_vec, ..., child_n_vec])
>
> E(query, centroid) low --> contents probably all low E --> SKIP
> E(query, centroid) high --> some contents probably high E --> RECURSE

The algorithm uses **Top-K** selection (not threshold-based pruning), per
SQuAD experimental validation:

> 1. Top-K centroid selection > threshold-based pruning
> 2. Semantic clustering (k-means) is REQUIRED
> 3. Retrieval is the bottleneck, not the LLM

Complexity: O(log n) E-computations per query vs brute force O(n). For 100K
turns: ~250 vs 100,000 E-computations.

### 4.4 The Bundle Protocol (BUNDLE_SPEC.md v5.0.0)

Jobs are packaged into deterministic, self-contained bundles:
1. Package manifest.json with all job metadata
2. Hash all artifacts, include in manifest
3. Compute bundle_hash = SHA-256 of manifest + artifacts

Bundles can be replayed offline with byte-identical outputs:
> BundleRunner operates without repo_root or database access -- all inputs
> resolved from artifacts directory. Run twice -> identical receipts.

### 4.5 Context Priority Tiers (verification_context_management.md)

When assembling context, items are prioritized into tiers:
1. **Mandatory**: System prompt + Latest user message
2. **Recent Dialog**: Assistant/User messages in reverse chronological order
3. **Explicit Expansions**: Symbol expansions from latest user message
4. **Optional Extras**: Additional expansions (only if budget remains)

If mandatory items exceed budget, the system refuses to return partial
context (fail-closed). Truncation uses HEAD truncation only (preserves start,
discards end).

---

## 5. How It Relates to the AGS System

### 5.1 Write Isolation (WRITE_ISOLATION.md)

The fundamental design constraint:

> CAT_CHAT is a **consumer** of main cassette content, not a contributor.
> - **Reads:** Main cassettes at `NAVIGATION/CORTEX/cassettes/*.db`
> - **Writes:** Local sandbox at `THOUGHT/LAB/CAT_CHAT/_generated/cat_chat.db`

CAT_CHAT reads from all 9 main AGS cassettes:

| Cassette | Content |
|----------|---------|
| canon.db | LAW/CANON documents |
| governance.db | CONTEXT decisions |
| capability.db | CAPABILITY code |
| thought.db | THOUGHT research |
| navigation.db | NAVIGATION maps |
| direction.db | DIRECTION roadmaps |
| memory.db | MEMORY archives |
| inbox.db | INBOX staging |
| resident.db | AI memories |

It writes only to `_generated/cat_chat.db` -- no cassette pollution.

### 5.2 MCP Tool Access

The `ChatToolExecutor` enforces a strict allowlist of read-only tools:

**Allowed (Read-Only):**
- `cassette_network_query`, `cortex_query`, `semantic_search`
- `semantic_stats`, `context_search`, `context_review`
- `canon_read`, `codebook_lookup`, `research_cache`
- `agent_inbox_list`, `message_board_list`

**Explicitly Excluded:**
- `memory_save`, `memory_promote`, `session_start` (would write to resident.db)

### 5.3 What Stays Local vs What Comes from Main Cassettes

**Local** (instance-specific, never graduates):
- Sessions, session events, working set, pointer set
- Expansion cache (runtime cache, expires)
- Cassette jobs, steps, receipts, budgets

**From main cassettes** (shared across the AGS ecosystem):
- Canon documents, governance decisions, capability definitions
- Research content, navigation maps, planning context
- Memory archives, inbox staging, agent memories

### 5.4 The 7 Catalytic Invariants (ROADMAP, Phase B.4)

| # | Name | Rule |
|---|------|------|
| INV-CATALYTIC-01 | Restoration | File states before/after must be identical |
| INV-CATALYTIC-02 | Verification | Proof size = O(1) per domain |
| INV-CATALYTIC-03 | Reversibility | restore(snapshot) = original (byte-identical) |
| INV-CATALYTIC-04 | Clean Space Bound | Context uses pointers, not full content |
| INV-CATALYTIC-05 | Fail-Closed | Restoration failure = hard exit |
| INV-CATALYTIC-06 | Determinism | Identical inputs = identical Merkle root |
| INV-CATALYTIC-07 | Auto-Context | Working set managed by system, not manual references |

### 5.5 Graduation Path (GRADUATION_PATH.md)

Three future options for when CAT_CHAT graduates from LAB:

**Option A: Session Data to resident.db**
- Merge `sessions` and `session_events` tables into resident.db
- Requires schema alignment and agent ID linking

**Option B: New CAT_CHAT Cassette**
- Create a new cassette registered in cassettes.json
- Requires chunks/files/ FTS tables compatible with GenericCassette

**Option C: Keep in LAB Permanently**
- No migration needed, state stays in `_generated/`
- Can be reset/cleared without affecting main system

Graduation prerequisites:
1. Phase A complete (session persistence) -- DONE
2. Phase B complete (cassette integration) -- DONE
3. Phase C complete (auto-context loop) -- Core DONE (C.6.3 pending)
4. All 7 invariants verified -- documented
5. Compression benchmarks proven -- DONE
6. Golden demo works from fresh clone -- DONE

### 5.6 Design Alignment

Both AGS and CAT_CHAT share the same design philosophy:

| Principle | In AGS | In CAT_CHAT |
|-----------|--------|-------------|
| Determinism | INV-005: same inputs, same outputs | All sections SHA-256 indexed, deterministic E-score |
| Fail-Closed | INV-018: hard gates, no "warn but pass" | BudgetExceededError, SliceError on ALL, append-only receipts |
| Traceability | ADR-021: mandatory session_id | session_capsule: hash-chained event log |
| Verification | INV-016: must execute verification | bundle verify: hash integrity checks |
| Output Roots | INV-006: _runs/, _generated/, _packs/ | paths.py: everything under _generated/ |
| Bounded Resources | Budget enforcement in contracts | Max symbols=10, max sections=5, bounded slices only |

---

## 6. Research Validation (from ROADMAP)

The roadmap documents several experimentally validated claims:

### 6.1 E-Score Validation (Q44)

| Metric | Result |
|--------|--------|
| Born rule correlation | r = 0.999 |
| Threshold | 0.5 (empirically validated) |
| Confirmed | E = |<psi|phi>|^2 |

### 6.2 SQuAD Hierarchy Benchmark

Tested on SQuAD reading comprehension (10K passages, 1K questions):

| Metric | Brute Force | Hierarchy | Result |
|--------|-------------|-----------|--------|
| Recall@10 | 88% | 85% | 97% of brute force |
| E-computations | 10,000 | 1,785 | 5.6x speedup |
| LLM accuracy | 100% | 100% | *on gold context |

### 6.3 Iso-Temporal Protocol

Validated on 99 arxiv papers (3487 sections):

| Method | Recall@10 | vs Pure E |
|--------|-----------|-----------|
| Pure E-score | 33.0% | baseline |
| Context (lambda=0.2) | 36.5% | +10.6% |
| Frame+E (lambda=0.5) | 38.5% | +16.7% |

Also validated on synthetic causal data (up to +37.3% recall).

### 6.4 Design Decisions from Research

The roadmap explicitly documents research-backed design changes:

> 1. Top-K centroid selection > threshold-based pruning
>    - Threshold-based pruning (E >= 0.5) was too aggressive
>    - Top-K selection (explore K=10 highest-E centroids) works better
>
> 2. Semantic clustering is REQUIRED
>    - Random grouping creates meaningless centroids
>    - k-means clustering before building hierarchy is essential

---

## 7. Status Summary

### Completed

- Phases A-J: fully implemented and tested
- 739 tests passing, 0 failures (v1.3.1)
- 4 formal specs published (bundle v5.0.0, receipt/trust/execution v1.0.0)
- Golden demo working from fresh clone
- Compression benchmarks operational
- Research validated: E-score, hierarchy, iso-temporal protocol

### Remaining Gap

- **C.6.3**: Track E-score vs response quality correlation
  - Marked "Future" in the roadmap
  - Core catalytic loop works, but quality correlation is not yet tracked

### Open Question

Per ROADMAP v2.0, the following research findings are noted as potential
improvements but not yet implemented:

| Finding | Source | Implication |
|---------|--------|-------------|
| Df = 22 effective dimensions | Q43 | Cluster in PCA-reduced space, not full 384D |
| Space is curved (holonomy -0.10 rad) | Q43 | Consider spherical k-means |
| Phase transition at alpha=0.9 | Q12 | Sharp boundary between meaningful/meaningless |
| Angular momentum conserved (CV=6e-7) | Q38 | Geodesic distance may outperform cosine^2 |
