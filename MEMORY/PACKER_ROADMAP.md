---
uuid: 00000000-0000-0000-0000-000000000000
title: 01-06-2026-12-58 Packer Roadmap
section: guide
bucket: agent-governance-system/MEMORY
author: System
priority: Medium
created: 2026-01-06 13:09
modified: 2026-01-06 13:09
status: Active
summary: Document summary
tags: []
hashtags: []
---
<!-- CONTENT_HASH: 989e618d2405bee643d4a04f67bf0ccf30ee24d49cf9bf6e274047b0a3ec231d -->

# LLM Packer Roadmap

## Target Pack Structure

```
pack/
 ├── FULL/                    # Combined single-file outputs (.md only)
 ├── SPLIT/                   # Chunked sections (INDEX, CANON, ROOT, MAPS, etc.)
 ├── LITE/                    # Compressed SPLIT with ELO tiers
 └── archive/
     ├── pack.zip             # Contains ONLY: meta/ + repo/
     ├── FULL.txt             # txt copies NEXT TO zip (not inside)
     ├── FULL-TREEMAP.txt
     ├── SPLIT-INDEX.txt
     ├── SPLIT-CANON.txt
     ├── LITE-INDEX.txt
     ├── <SCOPE>-FULL.txt     # txt copies NEXT TO zip (not inside); MUST be scope-prefixed (AGS-/CAT-/LAB-)
     ├── <SCOPE>-FULL-TREEMAP.txt
     ├── <SCOPE>-SPLIT-INDEX.txt
     ├── <SCOPE>-SPLIT-CANON.txt
     ├── <SCOPE>-LITE-INDEX.txt
     └── ...                  # etc for all md files
```

**Scopes (mutually exclusive):**
- AGS: excludes CAT and LAB
- CAT: excludes AGS and LAB
- LAB: excludes AGS and CAT

---

## LLM Packer Role in AGS

The **LLM Packer** is the **context compression engine** for AGS. It creates compressed "memory packs" that give LLMs a bounded view of the repository without loading everything into context.

### How It Fits with CAS and Cassette Network

```
┌─────────────────────────────────────────┐
│   LLM Packer (Compression Strategy)     │
│   "What should the LLM see?"            │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│   CAS (Storage Layer)                   │
│   "How should we store it?"             │
│   - Deduplication (same hash = one blob)│
│   - Immutable (hash = content)          │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│   Cassette Network (Discovery)          │
│   "How do we find what we need?"        │
│   - Semantic search (vectors)           │
│   - Symbol resolution (@Symbols)        │
└─────────────────────────────────────────┘
```

**Key Insight:** LLM Packer decides *what* to include (compression logic), CAS stores *how* (deduplicated blobs), and Cassette Network enables *discovery* (semantic search).

**Future Integration (Lane Z.2):** LITE packs will become manifests (pointers to CAS hashes) instead of storing full file bodies.

---

## Phase 0: 6-Bucket Migration (P0)

**Status:** ❌ **Not Started**  
**Blocker:** LLM Packer still references old paths (CANON/, CONTEXT/, etc.)

### Tasks
- [ ] Update `Engine/packer/core.py` to use new bucket paths:
  - `CANON/` → `LAW/CANON/`
  - `CONTEXT/` → `LAW/CONTEXT/` (decisions, preferences, rejected, open)
  - `CONTRACTS/` → `LAW/CONTRACTS/`
  - `SKILLS/` → `CAPABILITY/SKILLS/`
  - `TOOLS/` → `CAPABILITY/TOOLS/`
  - `MCP/` → `CAPABILITY/MCP/`
  - `PRIMITIVES/` → `CAPABILITY/PRIMITIVES/`
  - `CORTEX/` → `NAVIGATION/CORTEX/`
  - `MAPS/` → `NAVIGATION/MAPS/`
- [ ] Update `Engine/packer/split.py` to scan new bucket roots
- [ ] Update `Engine/packer/lite.py` to prioritize new bucket structure (HIGH ELO: LAW/CANON/, NAVIGATION/MAPS/)
- [ ] Update all scope configs (AGS, CAT, LAB) to reference new paths
- [ ] Update tests in `CONTRACTS/` to verify new bucket paths
- [ ] Update documentation (README, AGENTS.md) to reference new structure

### Exit Criteria
- Packer successfully generates packs using new bucket paths
- All tests pass with new structure
- No references to old paths (CANON/, CONTEXT/, etc.) in packer code

---

## Phase 1: Fix Output + Modularize Together

Do NOT fix code then modularize. Modularize AS you fix so we only do it once.

### Create Package Structure First

- [x] Create `Engine/packer/` directory
- [x] Create `Engine/packer/__init__.py` exporting: `make_pack`, `verify_manifest`, `PackScope`, `PROJECT_ROOT`
- [x] Create `Engine/packer/__main__.py` for canonical `python -m` entry point

### Modularize + Fix Duplication (core.py)

- [x] Create `Engine/packer/core.py` with:
  - `hash_file`, `read_text`, `estimate_tokens`
  - `build_state_manifest`, `manifest_digest`, `load_baseline`
  - `PackScope` dataclass
  - `render_tree` for deterministic visual file tree generation
  - `write_pack_file_tree_and_index` generating `meta/FILE_TREE.txt` and `FULL/*TREEMAP*`
- [x] Remove `FULL_COMBINED/` creation — outputs go to `FULL/` only
- [x] Remove `COMBINED/` folder — no longer exists
- [x] Remove mirroring logic (no duplicate SPLIT/, SPLIT_LITE/)
- [x] Restore treemap outputs: `meta/FILE_TREE.txt`, `FULL/{SCOPE}-TREEMAP-{stamp}.md/txt`

### Modularize + Fix Split (split.py)

- [x] Create `Engine/packer/split.py` with:
  - `write_split_pack`, `write_split_pack_ags`
  - `write_split_pack_catalytic_dpt`, `write_split_pack_catalytic_dpt_lab`
- [x] Output directly to `SPLIT/` (not `COMBINED/SPLIT/`)
- [x] Remove all "combined" references from generated markdown content

### Modularize + Fix Lite (lite.py) — Isolated

- [x] Create `Engine/packer/lite.py` with:
  - `write_split_pack_lite`, `write_lite_indexes`
- [x] Output directly to `LITE/` (not `COMBINED/SPLIT_LITE/` or `SPLIT_LITE/`)
- [x] Wrap imports in try/except so LITE failures don't break core packer

### Modularize + Fix Archive (archive.py)

- [x] Create `Engine/packer/archive.py` with:
  - `write_pack_internal_archives`, `_write_zip`, `_iter_files_under`
- [x] Create SINGLE `archive/pack.zip` containing ONLY:
  - `meta/` folder contents
  - `repo/` folder contents
- [x] Generate txt copies of all mds NEXT TO zip (not inside), flat files with clear names
    - [x] All archive sibling txt filenames MUST be prefixed with scope folder: AGS-, CAT-, LAB-
- [x] Remove separate `meta/` and `repo/` folders from pack root
- [x] Do NOT auto-prune archives

### Cleanup

- [x] Delete old `Engine/packer.py` after migration complete
- [x] Update `Engine/pack.ps1` to use canonical `python -m MEMORY.LLM_PACKER.Engine.packer` entry point
- [x] Update `Engine/verify_phase1.py` to verify treemap presence and COMBINED-free output
- [x] Update Launchers (`1-AGS-PACK.cmd`, `2-CAT-PACK.cmd`, `2-LAB-PACK.cmd`)
- [x] Refactor `lab` scope key and naming logic

---

## Phase 2: Update Governance References

Tests are in CONTRACTS and deeply embedded — we update them to match new output structure.

- [x] Create `scan_old_refs.py` script to find all files referencing FULL_COMBINED, COMBINED/SPLIT, SPLIT_LITE (and any other old paths)
- [x] Run scan script and update every file found that references old paths or files.
- [x] Update `SKILLS/llm-packer-smoke/` smoke tests for new structure (FULL/, SPLIT/, LITE/, archive/pack.zip)
- [x] Update `SKILLS/pack-validate/` to validate new structure
- [x] Ensure tests pass with new output structure

---

## Phase 3: Consolidate Documentation

No redundant files. Put everything in its right place.

### AGENTS.md or README.md (pick one for both humans and agents)

- [x] Decide: single file for humans + agents to minimize tokens
- [x] Include:
  - Pack structure spec (from AGI Compression.md)
  - Delta pack flow, baseline state, anchor files (from DELTA_PACKS.md)
  - Scope isolation rules
  - Usage commands (merge logic from SHIPPING.md)
  - Reference DETERMINISM.md (don't inline it)
- [x] Move changelog in README.md to CHANGELOG.md

### Keep Separate

- [x] DETERMINISM.md — technical spec, referenced from AGENTS/README

### Create New

- [x] `run_tests.cmd` — isolated test runner for packer work

### Delete After Content Moved

- [x] Delete `MEMORY/DELTA_PACKS.md`
- [x] Delete `MEMORY/AGI Compression.md`

---

## Phase 4: LITE ELO Implementation (Future)

Not implementing now. Research needed.

### ELO Tier System

- [ ] **HIGH ELO (inline with full integrity):** CANON/*, AGENTS.md, MAPS/*, core contracts — include completely
- [ ] **MEDIUM ELO (summarize):** SKILLS/*/SKILL.md, CONTEXT/decisions/* — show function signatures + one-line purpose, no code
- [ ] **LOW ELO (omit with pointer):** fixtures, logs, research, generated files — explain why removed, what to ask for

### LITE Output Requirements

- [ ] Include all HIGH ELO content with full integrity (no compression)
- [ ] Summarize MEDIUM ELO: function names, purpose, no code body
- [ ] For LOW ELO: list removed files, explain why, tell user how to request more info
- [ ] Check git history for when LITE was functioning correctly

---

## Phase 5: Modular Architecture Contract

- The packer is a package: `MEMORY/LLM_PACKER/Engine/packer/`
- Core utilities are scope-agnostic (hashing, manifests, deterministic IO).
- Scope-specific behavior is modularized so each scope can diverge safely:
  - Separate system/config modules per scope (AGS, CAT, LAB) for:
    - `source_root`
    - split generation mapping
    - lite strategy
    - meta emission differences
- **Rule:** Adding a new scope must not require editing existing scope logic (add new module + register it).

---

## Phase 6: CAS Integration (Future)

**Goal:** Integrate LLM Packer with Content-Addressed Storage (CAS) to enable deduplication and immutable artifact storage.

**Depends on:** Lane Z.2 (F3 / Content-Addressable Storage) must be complete.

### Tasks
- [ ] **Refactor LITE packs to use CAS references**
  - Instead of storing full file bodies, store CAS hashes
  - LITE pack manifest becomes: `{"file": "LAW/CANON/INTEGRITY.md", "hash": "sha256:abc123..."}`
  - File bodies stored in `.cas/` directory (deduplicated)
- [ ] **Update packer to write to CAS**
  - `write_split_pack()` → writes file bodies to CAS, returns hashes
  - `write_lite_pack()` → stores only manifest (pointers to CAS hashes)
- [ ] **Add CAS verification**
  - `verify_manifest()` → checks CAS hashes match file bodies
  - Fail-closed: missing CAS blob = verification failure
- [ ] **Implement garbage collection**
  - Track which CAS blobs are referenced by active packs
  - Prune unreferenced blobs (with safety margin)
- [ ] **Benchmark deduplication savings**
  - Measure storage reduction (same file across multiple packs = one CAS blob)
  - Measure pack generation speed (no need to re-hash identical files)

### Exit Criteria
- LITE packs are 80%+ smaller (manifests only, not full bodies)
- CAS deduplication works (same file = one blob)
- Verification passes (CAS hashes match file bodies)
- Garbage collection is safe (no accidental deletion of active blobs)

---

## Phase 7: Research (Future)

- [ ] **Context Indexing:** Include RAG or index like Cortex?
- [ ] **Competitive Analysis:** What are others doing that works?
- [ ] **Improvement:** How can we make it better?
- [ ] **Research Question:** Is this just RAG?

---

## Known Bugs (Tracked Separately)

- [x] Zip trying to run multiple times concurrently (Solved by strict scoped launchers)
- [x] Zip locking issues on Windows (Solved by write-to-tmp strategy)
- [x] Split for CAT needs verification (Verified)

---

## Do NOT Touch

- Exported packs already in `_packs/` (existing output)
- Already-created archives

## DO Update (Not Protected)

- Main governance tests in CONTRACTS — update to match new structure
- CANON files referencing packer
- Skills using packer output
