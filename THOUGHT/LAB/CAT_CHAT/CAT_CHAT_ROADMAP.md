# CAT_CHAT Roadmap (Consolidated)

**Last Updated:** 2025-12-31  
**Status:** Phase 6 complete, Phase 7 pending

---

## Overview

Build a chat substrate where models write compact, structured messages that reference canonical material via **symbols**, and workers expand only **bounded slices** as needed.

---

## Completed Phases

### Phase 0-3: Foundation (COMPLETE)
- ✅ Contract vocabulary frozen
- ✅ SQLite substrate + deterministic indexing
- ✅ Symbol registry + bounded resolver
- ✅ Message cassette (LLM-in-substrate communication)

### Phase 6: Attestation & Trust (COMPLETE)
- ✅ Phase 6.2: Receipt Attestation (Ed25519 signing)
- ✅ Phase 6.3: Receipt Chain Anchoring
- ✅ Phase 6.5: Signed Merkle Attestation
- ✅ Phase 6.6: Validator Identity Pinning + Trust Policy
- ✅ Phase 6.7: Sanity Check Fixes
- ✅ Phase 6.10: Receipt Chain Ordering Hardening

**Test Status:** 118 passed, 13 skipped

---

## Pending Work

### Phase 4: Discovery (FTS + Vectors)
- [ ] Add FTS index over sections
- [ ] Add embeddings table for sections
- [ ] Implement hybrid search (FTS + vector)
- [ ] Store retrieval receipts

### Phase 5: Translation Protocol
- [ ] Define Bundle schema
- [ ] Implement bundler (uses discovery)
- [ ] Add bundle verifier
- [ ] Add memoization across steps

### Phase 6.8: Execution Policy Gate
- [ ] Execution policy schema
- [ ] Policy module (load, validate, enforce)
- [ ] CLI policy integration
- [ ] Executor policy enforcement

### Phase 7: Production Integration
- [ ] MCP server integration
- [ ] Terminal sharing for swarm coordination
- [ ] Multi-model router stabilization
- [ ] Session persistence

---

## Hard Invariants

- ✅ No bulk context stuffing (use symbols/section_ids)
- ✅ No unbounded expansion (budgets enforced)
- ✅ Receipts mandatory (every step recorded)
- ✅ Deterministic addressing (stable section resolution)
- ✅ Discovery ≠ justification (vectors select, contracts verify)

---

## Next Steps (Priority Order)

1. **Phase 4 (Discovery)** - Enable semantic search
2. **Phase 5 (Translation)** - Minimal executable bundles
3. **Phase 6.8 (Policy)** - Unify verification requirements
4. **Phase 7 (Production)** - MCP integration + swarm coordination
