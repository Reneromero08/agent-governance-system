# CATALYTIC-DPT Roadmap
**Two-Speed Plan: PoC First, Then Scale**

Date: 2025-12-23
Principle: Minimum effort, maximum revolution. Prove the thesis before expanding scope.

---

## North Star

Build a **catalytic agent system** where:
- External substrates (disk, DB, browser, tools, swarm) are the true workspace
- Context holds hashes, plans, and proofs
- Every run is auditable and reversible via restoration proofs
- Continuity exists as real weight shifting in a tiny orchestrator (200M model)
- Parallelism is safe via single-commit governance

---

## Phase 0: Freeze the Contract (1-2 hours)

**Goal**: Define canonical schemas before implementation

### Deliverables
- [x] JobSpec JSON schema (task specification format)
- [x] Validation error vector format (deterministic error reporting)
- [x] Run ledger schema (audit trail format)

### Tasks
1. Write `SCHEMAS/jobspec.schema.json`
   - Support Phase 0 and Phase 1 task types
   - Include catalytic_domains, durable_outputs, swarm_parallel
   - Include metadata (priority, timeout, governance_check)

2. Write `SCHEMAS/validation_error.schema.json`
   - Deterministic error codes (UPPERCASE_SNAKE_CASE)
   - Support for error paths and nested errors
   - Distinction between errors and warnings
   - Timestamp and validator version

3. Write `SCHEMAS/ledger.schema.json`
   - Reference jobspec.schema.json
   - Include pre/post manifests, restore_diff, outputs
   - Include decision_log field for timestamped decisions
   - Include restoration_verified boolean

4. Document schemas in `SCHEMAS/README.md`

### Exit Criteria
- All three schemas are valid JSON Schema Draft 7
- Each schema validates itself
- Documentation explains each field clearly
- A 200M parameter model can understand the schemas from docs

---

## Phase 1: CATLAB (Tiny R&D Proof) (1-3 days)

**Goal**: Prove catalytic continuity with smallest possible system

### 1.1 Implement Catalytic Kernel (Deterministic)

Build core primitives in `PRIMITIVES/`:

#### catalytic_store.py
- `put(content: bytes) -> hash` (SHA-256)
- `get(hash: str) -> bytes`
- Sharded storage under `CATALYTIC-DPT/TESTBENCH/_store/`
- Atomic writes (temp + rename)
- ~100 LOC

#### merkle.py
- `MerkleTree.__init__(items: List[Tuple[str, str]])`
- `root_hash() -> str`
- `proof(leaf_hash: str) -> List[str]` (O(log n) verification)
- `verify(root, leaf, proof) -> bool`
- ~150 LOC

#### spectral_codec.py
- Transform domain (file system state) → spectrum (compact hash map)
- `domain_to_spectrum(path: Path) -> Dict[str, str]` (path → hash)
- `spectrum_to_root(spectrum: Dict) -> str`
- Analogous to Fourier transform: spatial → frequency domain
- ~100 LOC

#### ledger.py
- Append-only receipt storage
- `append(entry: Dict) -> None` (to JSONL)
- `read_all() -> List[Dict]`
- Deterministic ordering
- ~50 LOC

#### validator.py
- JSON schema validation (uses `jsonschema` library)
- Error vector generation (deterministic error codes)
- `validate_jobspec(spec: Dict) -> ValidationResult`
- `validate_ledger(ledger: Dict) -> ValidationResult`
- ~200 LOC

### 1.2 Micro-Orchestrator with Real Weight Updates

Pick the smallest model mechanism that can learn from validators:
- Start with tiny controller: MLP (2 hidden layers, 128 units) or GRU (32 units)
- Update only a small module (adapter/memory/controller weights)
- Persist as `model_hash` + `delta_hash` artifacts
- Example: Given a JobSpec, predict "will this pass validation" → train on actual outcomes

#### micro_orchestrator.py (~300 LOC)
```
1. Load current model weights (or init if first time)
2. Receive JobSpec
3. Predict: valid_probability, recommended_retries
4. Execute job (via catalytic_store, merkle, ledger)
5. Observe: actual validation result
6. Compute: gradient w.r.t. prediction error
7. Update: weight delta (alpha * gradient)
8. Persist: new model hash (SHA-256 of weights)
9. Log: (input, prediction, actual, delta_hash) to ledger
```

### 1.3 Fixture Dataset

- 100-500 examples: prompts → valid JobSpecs
- Adversarial invalid cases: missing fields, wrong types, out-of-bounds
- Store in `FIXTURES/phase1/`

### Metrics

- Schema pass rate (% of specs that validate)
- Average retries per task (improvement over time)
- Regression rate (after weight updates, does accuracy drop?)
- Reproducibility (same seed, same results always)

### Exit Criteria
- [ ] Pass rate improves measurably across sessions due to weight updates
- [ ] Every delta is recorded, hashed, and rollbackable
- [ ] Root digests and ledgers match expectations (run twice, get same roots)
- [ ] No dependence on Opus/Sonnet for the core loop (Codex 200M can execute)
- [ ] All fixtures pass
- [ ] No regression detected

---

## Phase 2: Swarm Parallelism (Safe Learning) (1-2 days)

**Goal**: Accelerate evaluation without corrupting continuity

### Deliverables
- Worker runner that evaluates tasks in parallel
- Governor reducer that applies one committed update deterministically
- Rollback policy if regression detected

### Exit Criteria
- 10x evaluation throughput
- Deterministic outcomes regardless of worker completion order
- No race conditions

---

## Phase 3: Substrate Offload Adapters (2-5 days, incremental)

**Goal**: Generalize offloading beyond models to arbitrary compute

### 3.1 Browser Compute Adapter
- `browser_exec(js_code, input_hash) -> output_hash`
- Browser pulls from CAS, writes to CAS

### 3.2 DB Compute Adapter
- `db_exec(query, input_hashes) -> output_hash`
- SQLite/DuckDB for joins, analytics

### 3.3 Toolchain Adapter
- `cli_exec(tool, args, input_hashes) -> output_hash`
- Ripgrep, formatters, compilers

### Exit Criteria
- At least two substrates working with hash I/O and validators
- Ledger captures substrate versions and outputs for replay

---

## Phase 4: Catalytic Runtime Hardening (1-3 days)

**Goal**: Upgrade trust boundary correctness

### Deliverables
- Fix forbidden overlap validation (both directions)
- Enforce output existence checks
- Domain root checks for all domains
- Failure modes produce hard fail + diffs

### Exit Criteria
- Ledger cannot claim success if outputs don't exist
- Domain boundaries are airtight

---

## Phase 5: Catalytic Pipelines (2-4 days)

**Goal**: Multi-step runs with single restoration proof boundary

### Deliverables
- CLI: `--step "intent::cmd"` support
- Per-step logs and exit codes
- Single restoration proof at end

### Exit Criteria
- Demonstrate 3-step pipeline:
  1. Build scratch index
  2. Run transform
  3. Validate + emit governed output
- Single proof boundary covers all steps

---

## Phase 6: Integrate With AGS Proper (After PoC Proven)

**Goal**: Adopt CATLAB primitives into main system

### Deliverables
- Standardize JobSpec schema across AGS skills
- Convert swarm-governor outputs to schema-validated JobSpecs
- Wrap high-risk AGS operations in catalytic runtime

### Exit Criteria
- One meaningful AGS operation runs catalytically with restoration proof

---

## Phase 7: Optional Deep Math Upgrades (Only If Needed)

- Hierarchical Merkle proofs (incremental verification)
- Semantic validators using typecheckers and dependency graphs

---

## Non-Goals (For Now)

- Full repo pack bodies
- Big model finetuning
- Sweeping refactors
- Canon expansion

**PoC first. Proof first. Then scale.**

---

## Build Order (Dependency Graph)

```
Phase 0 (Schemas)
    ↓
Phase 1 (CATLAB Primitives)
    ├─→ catalytic_store
    ├─→ merkle
    ├─→ spectral_codec
    ├─→ ledger
    ├─→ validator
    └─→ micro_orchestrator
    ↓
Phase 2 (Swarm Integration)
    ↓
Phase 3 (Substrate Adapters)
    ↓
Phase 4 (Runtime Hardening)
    ↓
Phase 5 (Pipelines)
    ↓
Phase 6 (AGS Integration)
```

---

## Next Action

**Implement Phase 0 contracts:**
1. Create `SCHEMAS/jobspec.schema.json` with examples
2. Create `SCHEMAS/validation_error.schema.json`
3. Create `SCHEMAS/ledger.schema.json`
4. Write `SCHEMAS/README.md`
5. Create test fixtures that validate against schemas
6. Exit: Schemas are self-validating and documented

Then: **Move to Phase 1 CATLAB implementation**
