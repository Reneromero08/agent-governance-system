# Catalytic Computing for Agentic Governance Systems (AGS)
## Findings Report (R&D Summary)

**Date:** 2025-12-23  
**Status:** Research synthesis + actionable design specification (bedroom-scale)

---

## Abstract

This report summarizes an emerging architecture for **catalytic computing** applied to agentic systems. The central finding is that an agent’s context window should not be treated as its workspace. Instead, **context is the control plane**, while the true workspace is an **external catalytic substrate** (filesystem, databases, browser runtime, toolchain, and parallel worker swarms). The catalytic constraint is a hard invariant: external substrates may be mutated aggressively during computation but must be **restored** (or proven unchanged) after the run, except for explicitly governed outputs. This shifts trust from “believe the model” to “verify the proof,” enabling large-scale operations (refactors, indexing, pipelines, multi-step transformations) with minimal token cost and maximal auditability.

A second finding is that “continuity” can be implemented locally without large models by treating **weight updates as first-class artifacts**. A microscopic model can perform strict JSON orchestration and improve across sessions through test-time learning (Titans-like behavior), while all heavy work is offloaded to deterministic tools and swarms. The smallest decisive proof-of-concept is a catalytic lab (CATLAB) where a tiny model learns to emit schema-valid JobSpecs, with every delta hashed, ledgered, and rollbackable.

---

## 1. Background and Motivation

### 1.1 The bottleneck we are attacking
Large language models are bounded by context length and token cost. Most agent failures on real codebases arise from:
- incomplete visibility (sampling and chunking)
- lossy summarization
- brittle retrieval
- unverified side effects (tools mutate state without audit)

This creates “surgery through a keyhole” failure modes, especially for multi-step operations spanning large repos.

### 1.2 Catalytic computing, translated
Catalytic space complexity establishes a counterintuitive capability: a system with small clean memory and a large “full” memory can still compute nontrivial results if the large memory is treated as a catalyst, participating but restored. The engineering translation is:

- **Clean space:** small, expensive, trusted (LLM context, short logs, minimal irreversible outputs)
- **Catalytic space:** huge, cheap, mutable during run (disk, DB, browser RAM, toolchain caches, swarm scratch)
- **Invariant:** catalytic space is restored to its pre-run state, or the run fails hard, or the system reverts to a prior state.

The “holy shit” moment is not “auditing file writes.” It is **breaking the context barrier**: holding a small digest in context while manipulating gigabytes outside context, with a proof boundary.

---

## 2. Core Hypothesis

### 2.1 Hypothesis A: Context is a control surface
The model’s context window functions like CPU registers or L1 cache. It should not store the world. It should store:
- references (hashes)
- run plans (JobSpecs)
- invariants and contracts
- proofs and summaries of external computation

### 2.2 Hypothesis B: Proof replaces permission
Instead of restricting agent power with brittle sandboxing, we grant broad capability and enforce safety through verifiable invariants:
- restoration proofs (roots match)
- schema proofs (outputs validate)
- reproducibility (inputs and artifacts addressed by hash)
- rollback (time-travel via ledger)

### 2.3 Hypothesis C: Continuity can be real weight shifting
Continuity can be implemented without persistent chat history by maintaining:
- a tiny model whose weights actually update (test-time learning)
- deltas that are hashed, recorded, and reversible
- external memory stored in content-addressed stores (CAS), not in the model’s fragile recall

This yields durable evolution inside the system that is measurable and controllable.

---

## 3. System Model

### 3.1 Three planes

#### Plane 1: Control Plane (deterministic kernel)
A small set of deterministic primitives:
- content addressing (CAS)
- spectral mapping (domain → spectrum)
- root hashing (Merkle-like digest)
- ledgers (immutable run receipts)
- validators (schema and invariants)
- commit/rollback rules

This plane should contain no “reasoning.” It is physics.

#### Plane 2: Compute Plane (offloaded substrates)
The heavy lifting occurs in external substrates:
- local small-model swarm (1B, 3B, 7B)
- CLI toolchain (formatters, linters, compilers, codemods, parsers)
- DB engines (SQLite, DuckDB) for joins, indexing, analytics
- browser runtime (JS) for DOM traversal, rendering, parsing pipelines
- optional remote models (only when necessary)

#### Plane 3: Trust Plane (proof boundary)
Everything that matters is verified:
- catalytic restoration proof (roots match)
- output truthfulness (outputs exist and validate)
- reproducibility (hashes, deterministic order)
- rollback to any prior state

---

## 4. Mathematical Notation (Transform Framing)

This system can be described in transform language without metaphysics.

### 4.1 Content transform
- **Externalize:**  

  `h = T(f)`  
  Map bytes `f` to a stable hash `h`.

- **Internalize:**  

  `f = S(h)`  
  Recover bytes `f` given hash `h` from a store.

### 4.2 Domain transform (spectral representation)
- **Encode a directory (domain):**  

  `T(D) -> C`  
  Where `C` is a spectrum: `{relative_path: content_hash}`.

- **Decode:**  

  `T⁻¹(C) -> D`  
  Re-materialize a domain from spectrum and store.

### 4.3 Root representation (the “32-byte mind”)
- `root(C) = H(sorted(path:hash entries))`

If two roots match, the domain state matches. Roots provide an O(1) equality check. Diffs are computed only when roots differ.

### 4.4 “Unitarity” as a conservation law
A catalytic run is “unitary” if:
- `pre_root == post_root` for all catalytic domains.

This is the computational conservation of information within the controlled boundary.

---

## 5. Catalytic Core Primitives

### 5.1 Content-Addressed Store (CAS)
Purpose: move data out of context into disk, address it by hash, and treat it as immutable.

Minimal interface:
- `put(bytes) -> hash`
- `get(hash) -> bytes`
- optional `put_file(path) -> hash`, `get_to_file(hash, path)`

Implementation constraints:
- idempotent writes (same content, same hash)
- sharded directories (git-style) to avoid filesystem limits
- atomic write semantics (write temp, rename)

CAS enables:
- externalized intermediate artifacts
- cheap deduplication
- precise provenance (hash is identity)

### 5.2 Spectral codec (domain → spectrum)
A “spectrum” is simply `{path: hash}` for a domain.

This provides:
- compact representation of a large directory
- ability to reason in coordinates (hashes) rather than raw bytes
- decoupling content from location

### 5.3 Root hashing and Merkle evolution
Start simple:
- root = hash of sorted leaf entries

Evolve to hierarchical Merkle:
- supports partial proofs and log-scale diffs
- avoids full rescans for large domains (future upgrade)
- enables “millions of files, one root” ergonomics

### 5.4 Ledger (run receipts)
Every catalytic session writes an immutable ledger entry including:
- run_id
- tool versions (hashes)
- input hashes
- config hash
- pre_roots per domain
- post_roots per domain
- computed diffs if roots differ
- governed outputs (paths + hashes)
- exit codes
- validation results

The ledger is the trust substrate.

---

## 6. What Makes “Badass Catalytic” (Operational Capabilities)

### 6.1 Catalytic pipelines
A catalytic run must support multiple steps, not just one command.

Goal:
- allow N-step pipelines that thrash scratch freely
- enforce one proof boundary at the end
- durable outputs are only committed at step N

This enables:
- refactor pipelines
- index build then validate then pack build
- multi-tool transformation plans

### 6.2 One invariant: restoration or hard fail
“Badass” is power per unit risk. The invariant is simple:
- if restoration proof fails, the run is invalid
- if outputs are missing, the run is invalid
- ledger must not lie

### 6.3 Fixing known validator holes
Two correctness risks were identified and must be addressed:

1) **Forbidden overlap hole**  
Domain validation must catch both domain-inside-forbidden and forbidden-inside-domain containment.

2) **Output existence checks**  
Validation must verify durable outputs exist and are within allowed roots.

These fixes are prerequisites for trust.

---

## 7. Offloading Compute Substrates (Beyond the Browser)

The browser offload is one instance of a broader pattern: any substrate can become catalytic if it can receive input by hash, compute cheaply, and return output hashes plus validator reports.

### 7.1 Browser as compute substrate
Browser runtime excels at DOM parsing, traversal, rendering, and client-side transformations.

Pattern:
- model emits JobSpec including `browser_exec(js, input_hash) -> output_hash`
- browser retrieves input by hash, computes, writes to CAS, returns hash
- kernel validates outputs and ledgers the result

### 7.2 Other substrates that fit the same interface
- DB engines: stable query execution for indexing, joins, aggregation
- parsers: Tree-sitter, AST extractors
- compiler/typecheckers: semantic truth generation and dependency checks
- CLI tools: ripgrep, formatters, linters, codemods
- media pipelines: ffmpeg, imagemagick
- headless automation: playwright pipelines

Unified requirement: outputs are hash-addressed and contract-validated.

---

## 8. Swarm Integration Without Confusion

Swarm is compute. Catalytic is the memory model plus proof boundary. They must be separated.

### 8.1 Multi-tier orchestration chain
A practical chain to manage token cost and reliability:
- Opus (planner): governance, schemas, contracts
- Codex CLIs (presidents): scoped implementation tasks
- Governor (Qwen 7B): strict dispatch, schema enforcement
- Workers (1B/3B/7B): mechanical labor in constrained templates

Small models are reliable when asked to fill structured templates under validation.

### 8.2 Strict JSON JobSpec as the universal interface
Workers output JSON validated locally before acceptance. This turns cheap models into reliable “hands.”

---

## 9. Real Continuity via “Titans-like” Weight Shifting

### 9.1 What we mean by “real weight shifting”
Not prompt memory. Not summaries. Actual parameter updates that persist and change behavior.

Feasible local approach:
- freeze most components
- update only a tiny module (adapter, memory module, controller)
- treat weight deltas as versioned artifacts

### 9.2 Why strict JSON orchestration is the perfect micro-task
Success is binary:
- schema-valid JSON or fail
- deterministic error vectors for learning signals
- measurable improvement across sessions

### 9.3 Parameter scale for the micro-model
For strict JobSpec emission, the “model” can be extremely small if constraints are strong:
- viable minimum: ~10k to 50k params (grammar machine plus small controller)
- comfortable: ~100k to 300k params
- overkill: >1M params for this narrow role

The model is not a general LLM. It is a constrained orchestrator.

### 9.4 Catalytic learning loop
Each run produces:
- `job_spec_hash`
- `validation_report_hash`
- `delta_hash` (weight update)
- `new_model_hash`

This yields reproducibility, rollback, and accountability.

---

## 10. Parallel Testing and Safe Learning

### 10.1 The iron law
Only one authority commits weight updates. Others propose deltas.

### 10.2 Fan-out / reduce / commit
1) Workers evaluate tasks in parallel and produce proposals.
2) Governor reduces proposals deterministically.
3) Governor applies one committed update and emits a new model hash.
4) Workers evaluate; rollback if regression.

---

## 11. Caching and “Never Pay Twice”

Cache key:
`(job_hash, input_hash, toolchain_hash, model_hash) -> output_hash`

If cached, repeated work becomes instant and token costs collapse.

---

## 12. Minimum Proof-of-Concept: CATLAB (Tiny R&D)

### 12.1 Components
- CAS (put/get by hash)
- spectral codec (domain → spectrum)
- root digest (pre/post verification)
- ledger writer (immutable run receipts)
- JSON schema validator with error vectors
- micro-model orchestrator with online updates
- fixture dataset (100 to 500 samples)

### 12.2 Success criteria
- schema pass rate improves over time via real weight updates
- every delta is ledgered and hash-addressed
- rollback reproduces prior behavior
- parallel evaluations preserve determinism

This is the smallest “undeniable” catalytic continuity demonstration.

---

## 13. Extensions (On-Deck)

- semantic validators (“sheaf-like gluing”) using compilers and dependency graphs
- true hierarchical Merkle proofs for scale
- full catalytic pipelines in AGS

---

## Conclusion

We now have a clear two-speed architecture:
- a deterministic catalytic kernel that defines truth
- offloaded substrates (swarm, browser, DB, tools) that provide cheap compute
- continuity through real weight shifting in a tiny orchestrator, fully ledgered

Next step is CATLAB: prove the core thesis before scaling.

---

## Footnotes

1. Catalytic space: computation leveraging large “full” memory under a restoration constraint (catalyst metaphor).  
2. Titans-like test-time learning: updating a small memory/controller module during operation so behavior persists across sessions.  
3. “Unitarity” is used here as an engineering metaphor for information conservation: pre-state equals post-state for catalytic domains.
