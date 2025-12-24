# Catalytic Compression Report
**Working title:** Compression as Data-Plane, Spectrum as Model-Plane  
**Date:** 2025-12-23  
**Status:** Research note (candidate: `RESEARCH/`)

## Abstract
This report defines a compression direction that preserves *catalytic* guarantees while reducing token burn and disk/IO costs. The key move is a strict separation:

- **Control plane (LLM-visible):** small, structured, text-first artifacts (maps, hashes, manifests, ledgers, schemas).
- **Data plane (tool-visible):** large byte payloads stored content-addressed (CAS), optionally compressed for storage and transport.

LLMs do not "read compressed files." They read a **spectral layer** (structure + hashes) and request **deterministic expansion** by hash when needed. Compression becomes an implementation detail of the CAS, not the semantic representation.

---

## 1. Problem Statement
You want:
1. **Minimum tokens** for iteration and orchestration (tiny bedroom compute, not TPU fantasies).
2. **Continuity** across runs (weight shifting / persistent improvement), without relying on expensive frontier model sessions.
3. **Catalytic invariants**: aggressive scratch use with proofs of restoration and truthfulness.
4. **LLM usability**: models must be able to navigate, read, and act on the system reliably.

The failure mode to avoid:
- Replacing readable structure with opaque compression that LLMs cannot interpret without a tool, leading to confusion, hidden state, and unverifiable claims.

---

## 2. Core Thesis
### 2.1 Compression is not the interface
Binary compression (zstd/gzip/brotli) reduces disk size but does not reduce LLM tokens unless you avoid injecting that content into context.

**Therefore:**
- Compress **storage** and **transport**.
- Keep **LLM-facing representation** readable and minimal.

### 2.2 The LLM reads a spectrum, not bodies
The “spectral layer” is the compact, model-readable representation of a domain:
- paths
- hashes
- signatures (AST skeletons, exports, definitions)
- dependency edges
- stable section IDs + deterministic summaries
- proofs (Merkle roots, restoration verification artifacts)

Everything heavy lives behind hashes in the CAS.

### 2.3 Catalytic guarantee = invertible + verifiable
Compression direction is only catalytic if it is:
- **Invertible**: decode(encode(x)) == x (exact bytes).
- **Verifiable**: hash(raw_bytes) matches expected.
- **Deterministic**: same inputs produce same identifiers and proofs.

---

## 3. Architecture
### 3.1 Two-plane model

**Control Plane (LLM-visible)**
- Job specs (schemas), manifests, ledgers
- Merkle roots for domains
- Spectrum maps: `{rel_path -> content_hash}` plus structural metadata
- Diff summaries and proof records
- All text-based; validation-ready (JSON/YAML/MD)

**Data Plane (tool-visible)**
- CAS blobs containing raw bytes, optionally stored compressed
- Intermediate artifacts: AST dumps, indexes, browser outputs, search caches
- Large generated reports, refactor scratch, etc.

### 3.2 Canonical identity: hash raw bytes
**Rule: The content address is derived from *raw bytes*, not compressed bytes.**

Reason:
- Raw hash defines semantic identity independent of codec choice.
- You can change codecs without changing the content identity.
- Verification remains stable.

Implementation pattern:
- `raw_sha = sha256(raw_bytes)`
- store payload as: `CAS/{raw_sha[:2]}/{raw_sha[2:]}` with metadata
- optionally store a compressed file beside it:
  - `payload.zst` with `codec=zstd` and deterministic parameters
  - metadata captures: `raw_sha, codec, stored_size, raw_size, maybe raw_mime`

### 3.3 "Spectral Codec" (directory representation)
Define a deterministic, LLM-readable structure:
- `SPECTRUM.json` (or `.ndjson`) mapping path to hash
- `STRUCTURE.json` containing:
  - exported symbols, function/class signatures, import graph, type hints
- `DOMAIN_ROOT.txt` containing Merkle root (single 32-byte line)

The spectrum is what the LLM packs into context; the CAS is what tools expand.

---

## 4. Tools Needed (Minimal)
### 4.1 CAS with optional compression (data plane)
Required operations:
- `put_bytes(raw_bytes) -> raw_sha`
- `get_bytes(raw_sha) -> raw_bytes` (auto-decompress if stored compressed)
- `put_file(path) -> raw_sha`
- `verify(raw_sha, raw_bytes) -> bool`

Compression is optional and internal:
- Store raw bytes uncompressed for simplicity initially.
- Add compression once the correctness proof is stable.

### 4.2 Merkle root scanner (domain proof)
- `scan(domain_path) -> root_hash`
- `diff(pre, post) -> added/removed/modified` (only when roots differ)

This enables:
- O(1) restoration check (root compare)
- O(n) diagnostic only on mismatch

### 4.3 Expand-on-demand readers (LLM usability)
To keep LLM tokens low:
- `read_hash(raw_sha, range=..., max_bytes=...) -> text`
- `grep_hash(raw_sha, pattern) -> matches`
- `ast_hash(raw_sha) -> signatures`

Key: models request specific pieces rather than ingesting everything.

---

## 5. Tiny Model Role: Compression Policy, Not Compression Generator
If you want microscopic models:
- Do **not** ask them to generate bitstreams (high risk of silent lossiness).
- Ask them to choose *representations and boundaries*.

**Compression Policy Model outputs:**
- which transform to apply (raw, normalized, AST skeleton, signature-only)
- chunk boundaries
- dictionary selection (optional)
- which items to externalize vs expand

This stays verifiable:
- Every transform is deterministic
- Every output is checked by hash equality

---

## 6. Offloading Compute Beyond Disk
Compression and externalization generalize to other executors as long as outputs become hashes in the CAS.

### 6.1 Browser as catalytic compute substrate
Browser/JS is excellent for:
- DOM parsing/traversal
- HTML rendering checks
- image ops (decode/resize/compress)
- large JSON transforms
- AST parsing with JS libs

Pattern:
1) LLM sends (input_hash, js_task_id, args)
2) Browser loads bytes from CAS by hash
3) Browser computes result bytes
4) Browser writes result back to CAS, returns result_hash
5) LLM sees only hashes + small summaries

### 6.2 Other substrates
- local CLI tools (ripgrep, tree-sitter, clang, python linters)
- sqlite or duckdb for structured queries
- wasm sandboxes for deterministic transforms
- git plumbing (diff/patch) as a reversible execution primitive

All of these become "catalytic" if:
- scratch outputs are in a declared domain
- restoration is proven
- durable outputs are explicitly declared and validated

---

## 7. Proof of Concept (Tiny R&D)
### Goal
Demonstrate “LLM sees a spectrum, tools expand by hash” with verifiable reconstruction.

### PoC A: Holographic roundtrip (10 minutes)
1) Create a small directory with 20–200 files.
2) Encode to spectrum + store all file bodies in CAS.
3) Delete the directory.
4) Decode from spectrum + CAS into a new directory.
5) Verify:
   - spectrum roots match
   - per-file raw hashes match

### PoC B: Restoration proof (catalytic runtime)
1) Snapshot Merkle root of a scratch domain.
2) Run a “thrash” command (create many temp files, indexes, whatever).
3) Force cleanup/restore.
4) Verify Merkle root is identical.
5) If mismatch, show diff.

### PoC C: Token efficiency demo
1) Pack only `STRUCTURE + SPECTRUM + ROOTS`.
2) Ask LLM to find/modify one function.
3) LLM requests `read_hash` for only the needed file regions.
4) Measure tokens vs naive “paste the file” approach.

---

## 8. Metrics (What makes this real, not hype)
Track these from day one:
- `root_scan_time_ms` vs domain size
- CAS store size vs raw size (compression ratio)
- “LLM tokens to complete task” baseline vs spectral approach
- diff false positives/negatives (should be zero, byte-true)
- rebuild determinism: same inputs => same roots and hashes

---

## 9. Risks and Failure Modes
1) **Lossy transforms disguised as compression**
   - Fix: enforce byte-level reconstruction checks.
2) **Hashing compressed bytes instead of raw**
   - Fix: raw hash is canonical identity.
3) **Non-deterministic tool outputs**
   - Fix: version pinning + include tool versions in ledgers.
4) **Opaque representations**
   - Fix: keep spectrum + structure human readable.
5) **Token leak via accidental “expand everything”**
   - Fix: default to hash pointers; require explicit expansion.

---

## 10. Decision Summary
**Direction change: YES, but subtle.**
- Keep **JSON/structured control-plane** for orchestration and proofs.
- Add **CAS compression** strictly as a data-plane optimization.
- Make LLMs consume **spectral artifacts**, not file bodies.
- Use tiny models for **policy selection**, not bitstream generation.

This preserves catalytic guarantees and reduces tokens without creating unreadable black boxes.

---

## Appendix A: Minimal File Artifacts
Suggested `RESEARCH/` artifacts for this direction:
- `SPECTRUM.json` (path -> raw_sha)
- `STRUCTURE.json` (signatures, imports, symbols)
- `DOMAIN_ROOTS.json` (domain -> merkle_root)
- `RUN_LEDGER.json` (proofs, versions, timings)
- CAS directory (opaque, tool-managed)

---

## Appendix B: Checklist
- [ ] CAS hashes raw bytes, not compressed bytes
- [ ] Deterministic codec settings (if compressing)
- [ ] Expand-by-hash tools exist
- [ ] Merkle root proof exists
- [ ] Runtime writes an auditable ledger
- [ ] PoC demonstrates roundtrip reconstruction
- [ ] Token efficiency measurement is captured
