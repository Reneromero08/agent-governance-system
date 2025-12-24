# CATALYTIC-DPT (Department)

**Status**: R&D → PoC → Integration
**Purpose**: Isolated development of catalytic computing system before integration into AGS proper
**Governance**: Sandboxed from main repo; testable independently; zero impact on AGS until exit criteria met

---

## Directory Structure

```
CATALYTIC-DPT/
├── README.md                    # This file
├── ROADMAP.md                   # Phase 0-7 execution plan
├── TESTBENCH.md                 # PoC testing strategy and fixtures
├── CODEX_SOP.json               # Operating manual for autonomous execution
│
├── PRIMITIVES/                  # Core implementations
│   ├── catalytic_store.py       # Content-addressable storage
│   ├── merkle.py                # Merkle tree for root digests
│   ├── spectral_codec.py        # Domain → spectrum encoding
│   ├── ledger.py                # Append-only receipt storage
│   ├── validator.py             # Schema validation + error vectors
│   └── __init__.py
│
├── SCHEMAS/                     # Phase 0 contracts (JSON schemas)
│   ├── jobspec.schema.json      # Canonical job specification
│   ├── validation_error.schema.json
│   ├── ledger.schema.json
│   └── README.md                # Schema documentation
│
├── TESTBENCH/                   # PoC test environment
│   ├── run_poc.py               # Main test runner
│   ├── smoke_test.json          # Minimal sanity check
│   └── report.md                # Execution report
│
├── FIXTURES/                    # Example data for testing
│   ├── phase0/                  # Phase 0 schema examples
│   │   ├── valid/
│   │   └── invalid/
│   ├── phase1/                  # Phase 1 CATLAB examples
│   │   ├── store_tests/
│   │   ├── merkle_tests/
│   │   └── ledger_tests/
│   └── adversarial/             # Edge cases
│
└── SKILLS/                      # Skills for integration into AGS
    ├── catlab-jobspec-schema/
    ├── catlab-validator/
    └── catlab-executor/
```

---

## Current Phase

**Phase 0: Contracts** (In Progress)

Define three canonical schemas before any implementation:
1. ✓ JobSpec JSON schema
2. ✓ Validation error vector format
3. ✓ Run ledger schema

**Phase 1: CATLAB** (Next)

Tiny R&D proof with real weight updates:
- `catalytic_store.py` (CAS)
- `merkle.py` (root digests)
- `spectral_codec.py` (domain encoding)
- `ledger.py` (receipt storage)
- `validator.py` (schema validation)
- Micro-orchestrator with gradient updates

**Phase 2-7**: See ROADMAP.md

---

## Success Criteria (PoC)

### Phase 0 Exit
- [ ] All three schemas defined and validated
- [ ] Schemas can validate themselves
- [ ] Documentation is clear for 200M parameter model

### Phase 1 Exit
- [ ] `catalytic_store.py` passes all fixtures (100-500 test cases)
- [ ] `merkle.py` produces stable roots for identical inputs
- [ ] `ledger.py` appends deterministically
- [ ] `validator.py` correctly identifies schema violations
- [ ] Micro-orchestrator weights update and improve accuracy
- [ ] No regression after weight updates
- [ ] Full restoration proof passes for all test runs

### Integration Checkpoint
- [ ] Zero impact on existing AGS operations
- [ ] All catalytic operations logged and auditable
- [ ] Critic.py enforces catalytic boundaries
- [ ] Tests pass in CI

---

## Autonomy and Delegation

This department is designed for **autonomous execution by small models** (e.g., Codex, 200M parameters):

1. **SOP**: Follow `CODEX_SOP.json` for step-by-step execution
2. **MCP Tools**: Call MCP tools for governance, validation, execution
3. **Logging**: Every decision goes to `task_log.jsonl`
4. **Fallback**: Escalate to user agent (Claude) if governance fails

---

## Integration Path

When Phase 1 PoC passes:

1. **Skills Migration**: Create AGS skills wrapper for each primitive
2. **CI Integration**: Add catalytic tests to `CONTRACTS/runner.py`
3. **Validator Upgrade**: Extend `TOOLS/critic.py` to enforce catalytic rules
4. **Lane F Integration**: Merge into main AGS roadmap as Phase 6

---

## References

- `CONTEXT/research/Catalytic Computing/CATALYTIC_ROADMAP.md` - Full theory and phases
- `CONTEXT/research/Catalytic Computing/MASTER_PLAN.md` - Implementation blueprint
- `CANON/CATALYTIC_COMPUTING.md` - Canonical definition (if exists)

---

## Status

- **Created**: 2025-12-23
- **Phase**: 0 (Contracts)
- **Token Budget**: Isolated from main system
- **Testability**: PoC before integration
- **Governance**: Sandboxed with clear boundaries
