# CATALYTIC-DPT TESTBENCH

**Purpose**: Isolated PoC testing before integration into AGS
**Strategy**: Validate each phase independently; no impact on main system
**Autonomy**: Testbench can be run by 200M model (Codex) following SOP

---

## Testing Pyramid

```
        Phase 7 (Optional Math)
        Phase 6 (AGS Integration)
        Phase 5 (Pipelines)
        Phase 4 (Runtime Hardening)
        Phase 3 (Substrate Adapters)
        Phase 2 (Swarm)
TESTBENCH FOCUS → Phase 1 (CATLAB) ← 200M model validates here
        Phase 0 (Contracts)
```

---

## Phase 0: Schema Validation

### Test Cases

#### valid_jobspec_phase0.json
```json
{
  "job_id": "test-valid-phase0",
  "phase": 0,
  "task_type": "schema_definition",
  "intent": "Test valid JobSpec against schema",
  "inputs": {"test": true},
  "outputs": {
    "durable_paths": ["CATALYTIC-DPT/TESTBENCH/_output"],
    "validation_criteria": {"schema_is_valid": true}
  },
  "catalytic_domains": ["CATALYTIC-DPT/TESTBENCH/_tmp"],
  "determinism": "deterministic",
  "swarm_parallel": false
}
```

#### invalid_jobspec_missing_phase.json
```json
{
  "job_id": "test-invalid-no-phase",
  "task_type": "schema_definition",
  "intent": "Should fail: missing 'phase' field"
}
```

#### invalid_jobspec_bad_determinism.json
```json
{
  "job_id": "test-invalid-bad-determinism",
  "phase": 0,
  "task_type": "schema_definition",
  "intent": "Should fail: invalid determinism value",
  "determinism": "INVALID_VALUE",
  "outputs": {...},
  "catalytic_domains": [...]
}
```

### Success Criteria
- [ ] All valid specs pass `validator.validate_jobspec()`
- [ ] All invalid specs fail with clear error codes
- [ ] Error messages follow validation_error schema
- [ ] Validation is deterministic (same input → same output)

---

## Phase 1: CATLAB Kernel

### 1.1 catalytic_store Tests

#### Test: Store and Retrieve
```python
def test_store_retrieve():
    store = CatalyticStore("CATALYTIC-DPT/TESTBENCH/_store")
    content = b"Hello, catalytic!"
    hash1 = store.put(content)
    retrieved = store.get(hash1)
    assert retrieved == content
    print(f"PASS: store_retrieve ({len(content)} bytes)")
```

#### Test: Deterministic Hashing
```python
def test_deterministic_hashing():
    store = CatalyticStore("CATALYTIC-DPT/TESTBENCH/_store")
    content = b"Determinism"
    hash1 = store.put(content)
    hash2 = store.put(content)
    assert hash1 == hash2
    print(f"PASS: deterministic_hashing (hash={hash1[:16]}...)")
```

#### Test: Large File
```python
def test_large_file():
    store = CatalyticStore("CATALYTIC-DPT/TESTBENCH/_store")
    content = b"X" * (10 * 1024 * 1024)  # 10 MB
    hash1 = store.put(content)
    retrieved = store.get(hash1)
    assert len(retrieved) == 10 * 1024 * 1024
    print(f"PASS: large_file ({len(content) / 1024 / 1024} MB)")
```

#### Test: Batch Operations
```python
def test_batch_operations():
    store = CatalyticStore("CATALYTIC-DPT/TESTBENCH/_store")
    hashes = []
    for i in range(100):
        h = store.put(f"Item {i}".encode())
        hashes.append(h)

    for i, h in enumerate(hashes):
        content = store.get(h)
        assert content == f"Item {i}".encode()
    print(f"PASS: batch_operations ({len(hashes)} items)")
```

### 1.2 merkle Tests

#### Test: Root Hash Stability
```python
def test_root_hash_stability():
    items = [("file1", "abc123"), ("file2", "def456"), ("file3", "ghi789")]
    tree1 = MerkleTree(items)
    tree2 = MerkleTree(items)
    assert tree1.root_hash() == tree2.root_hash()
    print(f"PASS: root_hash_stability (root={tree1.root_hash()[:16]}...)")
```

#### Test: Proof Verification
```python
def test_proof_verification():
    items = [("file1", "abc123"), ("file2", "def456"), ("file3", "ghi789")]
    tree = MerkleTree(items)
    root = tree.root_hash()

    for path, hash_val in items:
        proof = tree.proof(hash_val)
        assert verify(root, hash_val, proof)
    print(f"PASS: proof_verification ({len(items)} items verified)")
```

#### Test: Tamper Detection
```python
def test_tamper_detection():
    items = [("file1", "abc123"), ("file2", "def456")]
    tree = MerkleTree(items)
    root = tree.root_hash()

    # Tamper with proof
    proof = tree.proof("abc123")
    tampered_proof = [p[:31] + "X" for p in proof]  # Flip last char

    assert not verify(root, "abc123", tampered_proof)
    print(f"PASS: tamper_detection (detection works)")
```

### 1.3 spectral_codec Tests

#### Test: Domain to Spectrum
```python
def test_domain_to_spectrum():
    # Create test files
    test_dir = Path("CATALYTIC-DPT/TESTBENCH/_domain")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.txt").write_text("content2")

    spectrum = domain_to_spectrum(test_dir)
    assert "file1.txt" in spectrum
    assert "file2.txt" in spectrum
    print(f"PASS: domain_to_spectrum ({len(spectrum)} files)")
```

#### Test: Spectrum Root Determinism
```python
def test_spectrum_root_determinism():
    test_dir = Path("CATALYTIC-DPT/TESTBENCH/_domain")
    spectrum1 = domain_to_spectrum(test_dir)
    root1 = spectrum_to_root(spectrum1)

    spectrum2 = domain_to_spectrum(test_dir)
    root2 = spectrum_to_root(spectrum2)

    assert root1 == root2
    print(f"PASS: spectrum_root_determinism (root={root1[:16]}...)")
```

### 1.4 ledger Tests

#### Test: Append Entry
```python
def test_append_entry():
    ledger = Ledger("CATALYTIC-DPT/TESTBENCH/test_ledger.jsonl")

    entry1 = {"timestamp": "2025-12-23T00:00:00Z", "event": "test1"}
    ledger.append(entry1)

    entries = ledger.read_all()
    assert len(entries) == 1
    assert entries[0]["event"] == "test1"
    print(f"PASS: append_entry")
```

#### Test: Deterministic Ordering
```python
def test_deterministic_ordering():
    ledger = Ledger("CATALYTIC-DPT/TESTBENCH/test_ledger2.jsonl")

    for i in range(10):
        ledger.append({"seq": i, "value": i * 2})

    entries = ledger.read_all()
    for i, entry in enumerate(entries):
        assert entry["seq"] == i
    print(f"PASS: deterministic_ordering ({len(entries)} entries)")
```

### 1.5 validator Tests

#### Test: Validate Valid JobSpec
```python
def test_validate_valid_jobspec():
    spec = {
        "job_id": "test-valid",
        "phase": 0,
        "task_type": "schema_definition",
        "intent": "Test",
        "inputs": {},
        "outputs": {"durable_paths": [], "validation_criteria": {}},
        "catalytic_domains": []
    }

    validator = Validator()
    result = validator.validate_jobspec(spec)
    assert result.valid == True
    print(f"PASS: validate_valid_jobspec")
```

#### Test: Reject Invalid JobSpec
```python
def test_reject_invalid_jobspec():
    spec = {
        "job_id": "test-invalid",
        # Missing required fields
    }

    validator = Validator()
    result = validator.validate_jobspec(spec)
    assert result.valid == False
    assert len(result.errors) > 0
    print(f"PASS: reject_invalid_jobspec ({len(result.errors)} errors)")
```

#### Test: Error Vector Format
```python
def test_error_vector_format():
    spec = {"job_id": "test"}  # Invalid
    validator = Validator()
    result = validator.validate_jobspec(spec)

    # Check error vector format
    assert hasattr(result, 'valid')
    assert hasattr(result, 'errors')
    assert hasattr(result, 'timestamp')
    assert all(hasattr(e, 'code') for e in result.errors)
    print(f"PASS: error_vector_format")
```

### 1.6 Micro-Orchestrator Tests

#### Test: Weight Update Improves Accuracy
```python
def test_weight_update_improves_accuracy():
    orchestrator = MicroOrchestrator()

    # Round 1: predict on fixture specs
    acc1 = 0
    for spec in load_fixtures():
        pred = orchestrator.predict(spec)
        actual = validate_spec(spec)  # Ground truth
        if pred == actual:
            acc1 += 1
        orchestrator.update(spec, pred, actual)  # Learn

    # Round 2: accuracy should improve
    acc2 = 0
    for spec in load_fixtures():
        pred = orchestrator.predict(spec)
        actual = validate_spec(spec)
        if pred == actual:
            acc2 += 1

    assert acc2 >= acc1  # No regression
    print(f"PASS: weight_update ({acc1} → {acc2} correct)")
```

#### Test: Reproducibility (Same Seed → Same Results)
```python
def test_reproducibility():
    orchestrator1 = MicroOrchestrator(seed=42)
    orchestrator2 = MicroOrchestrator(seed=42)

    spec = load_fixture("adversarial_case_1")
    pred1 = orchestrator1.predict(spec)
    pred2 = orchestrator2.predict(spec)

    assert pred1 == pred2
    print(f"PASS: reproducibility")
```

---

## Phase 2+: Swarm, Adapters, Pipelines

### Test Strategy
- Each phase builds on Phase 1 ✓
- Phase 2 tests: parallel execution, deterministic sorting
- Phase 3 tests: browser, DB, CLI adapters
- Phase 4 tests: boundary violations detected
- Phase 5 tests: multi-step pipelines with single proof boundary

---

## Test Execution (run_poc.py)

```python
#!/usr/bin/env python3
"""
PoC Test Runner for CATALYTIC-DPT

Usage:
  python run_poc.py --phase 0
  python run_poc.py --phase 1
  python run_poc.py --all
  python run_poc.py --report
"""

def main():
    phases = {
        0: test_phase0_schemas,
        1: test_phase1_catlab,
        2: test_phase2_swarm,
    }

    results = {}
    for phase, test_func in phases.items():
        print(f"\n{'='*60}")
        print(f"Phase {phase} Tests")
        print(f"{'='*60}")

        try:
            passed, failed = test_func()
            results[phase] = {"passed": passed, "failed": failed}
            print(f"✓ Phase {phase}: {passed} passed, {failed} failed")
        except Exception as e:
            results[phase] = {"error": str(e)}
            print(f"✗ Phase {phase}: ERROR - {e}")

    # Generate report
    generate_report(results)

if __name__ == "__main__":
    main()
```

---

## Success Criteria for Full PoC

### Phase 0
- [ ] All 3 schemas defined and documented
- [ ] Schemas validate themselves
- [ ] 10+ valid and 10+ invalid test cases pass

### Phase 1
- [ ] All 5 primitives implemented (~1000 LOC total)
- [ ] All 50+ unit tests pass
- [ ] Micro-orchestrator shows measurable improvement
- [ ] Zero regressions detected
- [ ] Full restoration proofs pass
- [ ] Ledgers are auditable and deterministic

### Integration Readiness
- [ ] TESTBENCH report shows all green
- [ ] Zero dependencies on external systems
- [ ] Codex (200M) can execute via SOP
- [ ] Decision logs are complete and clear
- [ ] Ready for Phase 2 (Swarm) without changes

---

## Execution Instructions for Codex

1. Read `CATALYTIC-DPT/CODEX_SOP.json`
2. Read `CATALYTIC-DPT/TESTBENCH.md` (this file)
3. Run `python CATALYTIC-DPT/TESTBENCH/run_poc.py --all`
4. Check `CATALYTIC-DPT/TESTBENCH/report.md`
5. If all green: report success to user agent (Claude)
6. If any red: escalate with logs to user agent

**PoC first. Perfect execution. Then integrate.**
