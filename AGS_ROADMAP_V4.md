---
title: AGS Roadmap V4 (Remaining Work Only)
version: 4.0.0
last_updated: 2026-01-30
scope: Unfinished phases only - Crypto Safe, Swarm, Omega
style: agent-readable, task-oriented, minimal ambiguity
status: Active
supersedes: AGS_ROADMAP_MASTER.md (deprecated)
notes:
  - Every task must produce: tests + receipts + report.
  - Write scope must be explicitly allowlisted per ticket.
  - LAB is the safe zone; CANON requires maximal constraint.
---

# AGS Roadmap V4 - Remaining Phases

This roadmap contains ONLY the unfinished work. All completed phases (1-8) have been archived.

## Archived Phases (Completed)

| Phase | Name | Status | Test Count |
|-------|------|--------|------------|
| 1.1-1.7 | Integrity Gates & Repo Safety | COMPLETE | - |
| 2.1-2.3 | CAS & Packer Foundation | COMPLETE | - |
| 2.4.1-2.4.3 | Write Enforcement & Git Hygiene | COMPLETE | - |
| 3 | CAT Chat (Deterministic Chat) | CORE COMPLETE | 739 tests |
| 4.1-4.6 | Catalytic Architecture | COMPLETE | 83 tests |
| 5 | Vector/Symbol Integration | COMPLETE | 529 tests |
| 6 | Cassette Network (Semantic Manifold) | COMPLETE | 39 tests |
| 7 | Vector ELO (Systemic Intuition) | CORE COMPLETE | - |
| 8.0-8.5 | Resident AI (Vector Memory) | COMPLETE | - |

**Archive:** `MEMORY/ARCHIVE/roadmaps/`

---

# Global Definition of Done

Every task must produce:
- [ ] All relevant tests pass (task is incomplete until green)
- [ ] Receipts emitted (inputs, outputs, hashes, commands run, exit status)
- [ ] Human-readable report emitted (what changed, why, how verified)
- [ ] Scope respected (explicit allowlist for writes)

---

# Phase 1: Crypto Safe (Template Sealing & Release)

**Purpose:** Cryptographically seal the template for license enforcement and provenance.
**Status:** NOT STARTED
**Priority:** P1

## 1.1 Template Sealing Primitive (CRYPTO_SAFE.2)

- [ ] 1.1.1 Implement `template_seal(template_dir, output_path, meta) -> receipt`
  - Hash all template files (code, governance rules, architecture)
  - Sign manifest with your key (proves YOU released this)
  - Emit tamper-evident seal file
- [ ] 1.1.2 Implement `template_verify(sealed_dir, signature) -> verdict`
  - Verify hashes match original
  - Verify signature is valid
  - Detect ANY tampering

## 1.2 Release Manifest Schema (CRYPTO_SAFE.3)

- [ ] 1.2.1 Define release manifest schema
  - List of all template files with hashes
  - Version, timestamp, license reference
  - Your signature
- [ ] 1.2.2 Add signature support (offline signing)
  - GPG or age-based signing
  - Public key published for verification
  - "This is what I released" - irrefutable

## 1.3 Release Export Integration (CRYPTO_SAFE.4)

**Prerequisites:**
- [ ] **DECISION: Define template boundary** - Which files/features are framework vs instance-specific?
  - Review each directory and decide what's public-facing
  - Document first-run initialization process for new users
  - Test that template works standalone (without your data)
  - This is a MANUAL decision, not automated

- [ ] 1.3.1 Implement `export_template.py` script
  - Exclude all instance data (per inventory + manual decisions)
  - Include all framework code
  - Add `.gitkeep` files for empty directories
  - Seal the result
- [ ] 1.3.2 Emit `RELEASE_MANIFEST.json` + signature into export
- [ ] 1.3.3 Add `.gitattributes` export-ignore patterns for `git archive`
- [ ] 1.3.4 Write first-run documentation (how new users initialize their AGS instance)

## 1.4 Seal Verification Tool (CRYPTO_SAFE.5)

- [ ] 1.4.1 Add `verify_release(release_dir)` that checks:
  - All template files match manifest hashes
  - Signature is valid
  - No instance data leaked into release
  - Deterministic verification (same input -> same result)

## 1.5 Tests & Docs (CRYPTO_SAFE.6-7)

- [ ] 1.5.1 Fixtures: tampered file -> FAIL, invalid signature -> FAIL, instance data leak -> FAIL
- [ ] 1.5.2 Add `NAVIGATION/PROOFS/CRYPTO_SAFE/` verification guide

**Exit Criteria:**
- [ ] Template releases contain no instance data
- [ ] Seals are tamper-evident (any modification detectable)
- [ ] "You broke my seal" is cryptographically provable

---

# Phase 2: Swarm Architecture

**Purpose:** Multi-agent coordination with delegated task execution.
**Status:** NOT STARTED
**Priority:** P2 (experimental until proven)

## 2.1 Swarm Primitives (Z.6)

- [ ] 2.1.1 Test MCP tool calling with 0.5B models (Z.6.1)
- [ ] 2.1.2 Task queue primitives (dispatch/ack/complete) (Z.6.2)
- [ ] 2.1.3 Chain of command (escalate/directive/resolve) (Z.6.3)
- [ ] 2.1.4 Governor pattern for ant workers (Z.6.4)

## 2.2 Delegation Protocol (D.1)

- [ ] 2.2.1 Define JSON directive schema for delegated subtasks:
  - task_id, model_class (tiny/medium/large)
  - allowed_paths, read_paths
  - deliverable_types, required_verifications
- [ ] 2.2.2 Define Worker Receipt schema:
  - touched_files (sorted), produced_artifacts (CAS refs)
  - patch_ref (optional), assumptions, errors (sorted), verdict
- [ ] 2.2.3 Require patch-first outputs for tiny models (no direct writes unless explicitly allowlisted)
- [ ] 2.2.4 Define Verifier requirements:
  - Validate allowlists
  - Apply patch deterministically
  - Run tests + greps
  - Emit receipts and fail-closed on any mismatch

## 2.3 Delegation Harness (D.2)

- [ ] 2.3.1 One "golden delegation" job:
  - Tiny worker produces patch + receipt
  - Governor verifies + applies
  - Tests pass
  - Receipts deterministic across re-runs with fixed inputs
- [ ] 2.3.2 Negative tests:
  - Out-of-scope file touched -> FAIL
  - Missing receipt fields -> FAIL
  - Non-deterministic ordering -> FAIL

---

# Phase 3: System Evolution (Omega)

**Purpose:** Long-horizon improvements and future research integration.
**Status:** ONGOING
**Priority:** P3 (post-substrate)

## 3.1 Performance Foundation

- [ ] 3.1.1 Incremental indexing
- [ ] 3.1.2 Query result caching
- [ ] 3.1.3 Compression metrics dashboard

## 3.2 Scale & Governance

- [ ] 3.2.1 Multi-cassette federation
- [ ] 3.2.2 Temporal queries (time travel)
- [ ] 3.2.3 Receipt compression

## 3.3 Intelligence & UX

- [ ] 3.3.1 Automatic symbol extraction
- [ ] 3.3.2 Smart slice prediction
- [ ] 3.3.3 Provenance graph visualization
- [ ] 3.3.4 Zero-knowledge proofs research

## 3.4 CAT Chat Graduation (from Phase 3 Future Work)

- [ ] 3.4.1 E-score vs response quality correlation tracking (C.6.3)
- [ ] 3.4.2 PCA-reduced space clustering (Df=22) for advanced context management
- [ ] 3.4.3 Production graduation: Move CAT Chat from LAB to main system

## 3.5 ESAP - Eigenvalue Spectrum Alignment Protocol (from Phase 6 Future Work)

**Research Status:** VALIDATED (r = 0.99+ eigenvalue correlation across models)
**Location:** `THOUGHT/LAB/VECTOR_ELO/eigen-alignment/`

Cross-model semantic alignment via eigenvalue spectrum invariance.

- [ ] 3.5.1 ESAP.1 Implement full protocol per OPUS pack spec
  - Protocol message types: ANCHOR_SET, SPECTRUM_SIGNATURE, ALIGNMENT_MAP
  - CLI: `anchors build`, `signature compute`, `map fit`, `map apply`
- [ ] 3.5.2 ESAP.2 Benchmark with 8/16/32/64 anchor sets
- [ ] 3.5.3 ESAP.3 Test neighborhood overlap@k on held-out set
- [ ] 3.5.4 ESAP.4 Compare with vec2vec (arXiv:2505.12540) neural approach
- [ ] 3.5.5 ESAP.5 Integrate as cassette handshake artifact (cross-model portability)
- [ ] 3.5.6 ESAP.6 8e Conservation Law Integration (pending research validation)
  - Q48-50 discovered: `Df x alpha = 8e ~ 21.746` (CV<2% across 24 models)
  - Track as telemetry first, gate later if research holds

## 3.6 ELO Monitoring (from Phase 7 Future Work)

- [ ] 3.6.1 E.6.2 Export ELO metrics to Prometheus/Grafana
- [ ] 3.6.2 E.6.3 Add ELO alerts (low-ELO content accessed frequently = potential echo chamber)

## 3.7 Vector Execution (from Phase 8 Future Work)

**Status:** Design-only specification. Zero implementation work has begun.
**Priority:** P2 (medium-low), long-horizon (year 2+ scope)
**Distinct from 8.0-8.5:** This is about executing *code* in vector space (e.g., running fibonacci via vector ISA), NOT the vector-based memory/reasoning already implemented.

Research foundation exists: HDC/VSA papers indexed (5), vec2text papers (5), CodeBERT citations documented.

- [ ] 3.7.1 R.6.1 Code vector representation research + implementation
- [ ] 3.7.2 R.6.2 Vector ISA design + interpreter
- [ ] 3.7.3 R.6.3 Hybrid execution runtime + fallback
- [ ] 3.7.4 R.6.4 SPECTRUM-V verification protocol
- [ ] 3.7.5 R.6.5 Production integration rollout phases

## 3.8 Cassette Network Global Protocol (from Phase 6 Future Work)

**Vision:** Cassette Network Protocol becomes internet-scale standard for semantic knowledge sharing - "Git for meaning."

| Phase | Scope | Features |
|-------|-------|----------|
| Current | Local | SQLite cassettes |
| Phase 7 | Internet | TCP/IP, cassette URIs (`snp://example.com/cassette-id`) |
| Phase 8 | P2P | DHT-based discovery, no central registry |
| Phase 9 | Global | Multi-language SDKs, public cassettes, DAO governance |

- [ ] 3.8.1 Design internet-scale cassette protocol
- [ ] 3.8.2 Implement cassette URIs and remote sync
- [ ] 3.8.3 P2P discovery via DHT
- [ ] 3.8.4 Multi-language SDK (Python, JS, Go)
- [ ] 3.8.5 DAO governance specification

---

# Priority Summary

| Priority | Phase | Status | Dependencies |
|----------|-------|--------|--------------|
| P1 | 1. Crypto Safe | Not Started | None (can start now) |
| P2 | 2. Swarm | Not Started | MCP tool calling research |
| P3 | 3. Omega | Ongoing | Phases 1-8 complete |

---

# Success Metrics

| Metric | Target |
|--------|--------|
| Template releases | Zero instance data leakage |
| Seal verification | 100% tampering detection |
| Swarm delegation | Deterministic receipts |
| ELO convergence | Variance <10% after 100 sessions |
| LITE pack accuracy | 90%+ accessed files are high-ELO |

---

# Changelog

| Version | Date | Changes |
|---------|------|---------|
| 4.0.0 | 2026-01-30 | New roadmap with only remaining phases; supersedes AGS_ROADMAP_MASTER.md |

---

*Roadmap v4.0.0 - 2026-01-30*
*Phases 1-8 archived. Remaining: Crypto Safe, Swarm, Omega.*
