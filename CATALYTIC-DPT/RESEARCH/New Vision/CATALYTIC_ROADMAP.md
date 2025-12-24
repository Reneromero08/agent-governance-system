# Catalytic Computing for AGS
## Roadmap (Two-Speed Plan: Tiny Proof First, Then Scale)

**Date:** 2025-12-23  
**Principle:** Minimum effort, maximum revolution. Prove the core thesis before expanding scope.

---

## North Star

Build a local-first catalytic agent system where:
- external substrates (disk, DB, browser, tools, swarm) are the true workspace
- context holds hashes, plans, and proofs
- every run is auditable and reversible
- continuity exists as real weight shifting in a tiny orchestrator
- parallelism is safe via single-commit governance

---

## Phase 0: Freeze the Contract (1 afternoon)

**Deliverables**
- Define one canonical **JobSpec JSON schema** for the PoC.
- Define one canonical **validation error vector format** (deterministic).
- Define one canonical **ledger schema** (run receipt format).

**Exit criteria**
- A test can declare PASS/FAIL with no ambiguity.
- Ledger contains enough hashes to replay or rollback.

---

## Phase 1: CATLAB (Tiny R&D Proof) (1 to 3 days)

**Goal**
Prove catalytic continuity with the smallest possible system. No repo refactors, no packs, no broad integration.

### 1.1 Implement Catalytic Kernel (deterministic)
- `catalytic_store.py` (CAS put/get, sharded, atomic writes)
- `spectral_codec.py` (domain → spectrum `{path:hash}`)
- `root.py` (root digest over spectrum)
- `ledger.py` (append-only receipts)
- `validator.py` (JSON schema + error vector)

### 1.2 Implement Micro-Orchestrator with Real Weight Updates
Pick the smallest model mechanism that can learn from validators:
- start with a tiny controller (MLP/GRU or even linear classifier)
- update only a small module (adapter/memory/controller)
- persist as `model_hash` + `delta_hash` artifacts

### 1.3 Fixture Dataset
- 100 to 500 examples of prompts → valid JobSpecs
- include adversarial invalid cases to train correction

**Metrics**
- schema pass rate
- average retries per task
- regression rate (after updates)
- reproducibility (same seed, same results)

**Exit criteria**
- Pass rate improves measurably across sessions due to weight updates.
- Every delta is recorded, hashed, and rollbackable.
- Root digests and ledgers match expectations.
- No dependence on Opus for the core loop.

---

## Phase 2: Swarm Parallelism (Safe Learning) (1 to 2 days)

**Goal**
Accelerate evaluation without corrupting continuity.

**Deliverables**
- worker runner that evaluates tasks in parallel and emits:
  - outputs
  - validation reports
  - proposed deltas or gradients
  - metrics
- governor reducer that:
  - sorts proposals deterministically
  - applies only one committed update
  - emits one new model hash
- rollback policy:
  - if evaluation shows regression, revert to prior model hash

**Exit criteria**
- 10x evaluation throughput without race conditions.
- Deterministic outcomes regardless of worker completion order.

---

## Phase 3: Substrate Offload Adapters (2 to 5 days, incremental)

**Goal**
Generalize offloading beyond “models” to arbitrary compute substrates.

### 3.1 Browser Compute Adapter
- `browser_exec(js_code, input_hash) -> output_hash`
- browser pulls inputs from CAS, writes outputs to CAS
- validator gates acceptance

### 3.2 DB Compute Adapter
- `db_exec(query, input_hashes) -> output_hash`
- use SQLite or DuckDB for joins, indexing, analytics

### 3.3 Toolchain Adapter
- `cli_exec(tool, args, input_hashes) -> output_hash`
- stable wrappers around ripgrep, formatters, parsers, compilers

**Exit criteria**
- At least two substrates working end-to-end with hash I/O and validators.
- Ledger captures substrate versions and outputs for replay.

---

## Phase 4: Catalytic Runtime Hardening (1 to 3 days)

**Goal**
Upgrade trust boundary correctness.

**Deliverables**
- fix forbidden overlap validation (both containment directions)
- enforce output existence checks
- domain root checks for all catalytic domains
- failure modes produce hard fail and clear diffs

**Exit criteria**
- Ledger cannot claim success if outputs do not exist.
- Domain boundary holes are closed.

---

## Phase 5: Catalytic Pipelines (2 to 4 days)

**Goal**
Multi-step runs with a single restoration proof boundary.

**Deliverables**
- CLI or API that supports repeated `--step "<intent>::<cmd>"`
- per-step exit codes and per-step logs
- one restoration proof at end
- durable outputs only committed at finalization

**Exit criteria**
- Demonstrate a 3-step pipeline:
  - build scratch index
  - run transform
  - validate + emit governed output
- Single proof boundary covers all steps.

---

## Phase 6: Integrate With AGS Proper (after PoC is proven)

**Goal**
Adopt CATLAB primitives into the main system gradually.

**Deliverables**
- standardize JobSpec schema across skills
- convert swarm-governor outputs to schema-validated JobSpecs
- wrap high-risk operations in catalytic runtime and pipelines

**Exit criteria**
- One meaningful AGS operation runs catalytically with restoration proof and ledger receipts.

---

## Phase 7: Optional Deep Math Upgrades (only if needed)

- hierarchical Merkle proofs (incremental verification)
- semantic validators (“sheaf-like gluing”) using typecheckers and dependency graphs

---

## Non-Goals (for now)
- full repo pack bodies
- big model finetuning
- sweeping refactors
- canon expansion

PoC first. Proof first. Then scale.

---

## Immediate Next Action
Implement Phase 0 contracts:
- JobSpec schema
- validation error vector format
- ledger schema
