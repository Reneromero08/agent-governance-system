---
title: "CATALYTIC_STACK_COMPRESSED_MERGED_GPT_5.2"
section: "research"
author: "System"
priority: "Medium"
created: "2025-12-26 06:39"
modified: "2025-12-26 06:39"
status: "Active"
summary: "Legacy research document migrated to canon format"
tags: ["research", "legacy"]
---
<!-- CONTENT_HASH: 9e357ccf88b262e489136ae7999a5fdd6a794698ec70e2125f3718e05c3754ba -->

# Catalytic Computing Stack (Compressed + Merged)

## 0\. Executive summary (the whole system in ~30 lines)

**Thesis:** *Context is the control plane, disk is the workspace.*  
Agents can thrash large state inside declared **catalytic domains**, but must restore those domains **byte-identical** at end-of-run, while leaving durable artifacts only in **durable output roots**. Trust shifts from “believe the model” to “verify the proof.”

**Three planes**

1. **Control plane (LLM-visible, small):** contracts, JobSpecs, schemas, ledgers, hashes, roots, spectra, summaries.
2. **Data/compute plane (tool-visible, large):** CAS blobs, scratch, indexes, DB, browser runtime, CLI toolchain, swarm workers.
3. **Trust plane (proof boundary):** restoration proofs + schema validation + output existence/containment.

**Non-negotiables**

* No silent writes outside catalytic domains.
* No durable artifacts outside durable output roots.
* Restoration proof must validate (or the run is invalid).
* Outputs must *exist* and must be *contained* in allowed durable roots.

---

## 1\. CMP-01: Catalytic Mutation Protocol (condensed spec)

### 1.1 One-line rule

An agent may mutate only inside declared catalytic domains and must restore them byte-identical at end-of-run, while writing durable outputs only to allowed output roots.

### 1.2 Durable output roots (hard allowlist)

* `CONTRACTS/\_runs/`
* `CORTEX/\_generated/`
* `MEMORY/LLM\_PACKER/\_packs/`

### 1.3 Catalytic domains (recommended defaults)

* `CONTRACTS/\_runs/\_tmp/`
* `CORTEX/\_generated/\_tmp/`
* `MEMORY/LLM\_PACKER/\_packs/\_tmp/`
* `TOOLS/\_tmp/` (optional)
* `MCP/\_tmp/` (optional)

### 1.4 Forbidden roots (examples)

* `BUILD/`
* `CANON/`
* `AGENTS.md` (and other root authorities)

### 1.5 Lifecycle (must be explicit)

0. **Declare**: run\_id, domains, durable roots, intent, determinism, tools\_used
1. **Snapshot**: pre-proof of catalytic domains
2. **Mutate**: only within catalytic domains
3. **Commit outputs**: only within durable output roots
4. **Restore**: catalytic domains -> exact pre-state
5. **Prove**: validate proof; hard fail + quarantine if mismatch

### 1.6 Proof formats

* **A. Manifest diff (practical default):** pre/post `{path -> sha256}` manifests + diff (must be empty)
* **B. Merkle root:** store pre/post roots; diff leaves only on mismatch (scales better)
* **C. Git overlay:** only if runtime is git-native and you can enforce empty diffs

### 1.7 Run ledger (minimum schema)

Store under `CONTRACTS/\_runs/<run\_id>/`

Required:

* `RUN\_INFO.json` (run\_id, timestamp, intent, determinism, toolchain, domains, durable roots)
* `PRE\_MANIFEST.json`
* `POST\_MANIFEST.json`
* `RESTORE\_DIFF.json` (must be empty)
* `OUTPUTS.json` (durable outputs list, ideally with sha256)
* `STATUS.json` (exit\_code + restoration\_verified)

### 1.8 Enforcement hooks (3 layers)

1. **Preflight**: validate domains/roots/forbidden overlaps + output paths allowed
2. **Runtime guard** (strongly recommended next): record writes during execution; fail on forbidden writes
3. **CI validation**: ledger must validate; restoration must hold; outputs must exist and be contained

---

## 2\. Compression direction: spectrum over bodies (control plane vs data plane)

### 2.1 Core rule

Compression is not the interface. Keep LLM-facing representation **readable and minimal**. Put bytes behind hashes.

### 2.2 Canonical identity

Hash **raw bytes**, not compressed bytes:

* `raw\_sha = sha256(raw\_bytes)`
* CAS storage may compress internally, but **identity stays raw\_sha**

### 2.3 “Spectral layer” (LLM-visible)

For any domain, store small artifacts:

* `SPECTRUM.json` : `{rel\_path: raw\_sha}`
* `STRUCTURE.json` : exports, signatures, import graph, IDs
* `DOMAIN\_ROOT.txt` : one root digest line

LLM navigates via spectrum + structure, then requests expansion by hash (range-limited reads) only when needed.

## Why "Spectrum"? (One-sentence metaphors)

* **Fourier**: Compact representation of structure, not raw samples.
* **Fractal**: Small seed (hash) → infinite detail (content).
* **Sheaf**: Local sections must "glue" correctly (Merkle roots match).

---

## 3\. Core primitives (the deterministic kernel)

### 3.1 CAS (content-addressed store)

Minimal:

* `put(bytes)->sha`
* `get(sha)->bytes`
* optional file helpers; atomic writes; sharded dirs

### 3.2 Spectral codec (domain ↔ spectrum)

* Encode: directory -> `{path:sha}`
* Decode: `{path:sha}` + CAS -> directory (exact bytes)

### 3.3 Root digest

Start: `root = sha256(sorted("path:sha" lines))`  
Upgrade later: hierarchical Merkle for incremental verification.

### 3.4 Ledger (run receipts)

Append-only run record containing:

* inputs (hashes/roots)
* outputs (paths + hashes)
* tool versions
* validation results
* restore proof metadata

### 3.5 Validators

* schema validators (JobSpec, ledger schema, output schema)
* CMP-01 validator (domains/roots/forbidden overlaps, outputs containment + existence)

---

## 4\. Roadmap (two-speed plan)

### Phase 0: Freeze the contract

* One canonical JobSpec JSON schema
* One canonical deterministic validation error vector
* One canonical ledger schema

### Phase 1: CATLAB (tiny proof)

* CAS + spectrum + root + ledger + validator
* micro-orchestrator that learns to emit schema-valid JobSpecs (real local continuity)
* fixture dataset (100–500 prompts -> JobSpecs), incl. adversarial invalid cases

Exit:

* measurable pass-rate improvement across sessions via persisted deltas
* reproducibility and rollback of deltas

### Phase 2: Swarm parallelism

* parallel evaluation + deterministic reducer
* one committed update per round; rollback on regression

### Phase 3: Substrate offload adapters

* browser\_exec(js, input\_hash)->output\_hash
* db\_exec(query, input\_hashes)->output\_hash
* cli\_exec(tool, args, input\_hashes)->output\_hash
  All gated by validators + ledgered.

### Phase 4: Runtime hardening (trust boundary correctness)

* close forbidden-overlap hole (both containment directions)
* enforce output existence checks (not just “allowed path”)
* ensure all domains and outputs are within declared roots

### Phase 5: Catalytic pipelines

* N-step pipelines; one proof boundary at end; durable outputs committed at finalization only

### Phase 6: Integrate into AGS

* standardize schemas across skills
* wrap high-risk operations (refactors, pack builds, index builds) in catalytic runtime

---

## 5\. Current implementation status (F2 prototype snapshot)

### What exists (working MVP)

* `TOOLS/catalytic\_runtime.py`: wraps a command in CMP-01 phases (snapshot, exec, verify, ledger)
* `TOOLS/catalytic\_validator.py`: validates ledgers in CI
* Proven: cortex build can run in catalytic mode with restoration verification

### Known limitations to address next

* Snapshot cost: O(n) re-hash every run (upgrade to Merkle roots or incremental manifests)
* Atomicity: outputs may be written before restoration proof succeeds (upgrade to staging + finalize)
* Memory blowup risk: in-memory manifest dicts for huge domains (upgrade to streaming / Merkle)
* No write tracing during execution (add write log or guard layer)
* Allowed roots hard-coded (parameterize via canon/config)

---

## 6\. Required correctness patches (must-do)

1. **Forbidden overlap hole**  
   Domain validation must catch:

* domain inside forbidden
* forbidden inside domain
* and any partial overlap/containment edge cases.

2. **Output existence + containment**  
   Validation must:

* ensure every declared durable output exists after run
* ensure each output is contained in allowed durable roots (path traversal safe)
* reject outputs that point outside project root or into forbidden roots.

---

## 7\. Suggested “single canonical doc” split (if you prefer multiple files)

If you keep this merged doc, it can still be split cleanly later into:

1. `CONTRACTS/CMP-01.md` (normative spec)
2. `RESEARCH/catalytic-computing.md` (theory + translation boundaries)
3. `RESEARCH/catalytic-compression.md` (spectrum/CAS direction)
4. `ROADMAP/catalytic-roadmap.md` (phases)
5. `REPORTS/catalytic-implementation.md` (what’s built + gaps)

The key is: **one place for norms** (CMP-01), everything else non-normative.

---

