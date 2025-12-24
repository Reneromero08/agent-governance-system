# CATALYTIC-DPT Index

**Status**: Phase 0 Ready for Codex
**Date**: 2025-12-23
**Budget**: Isolated R&D, no impact on main AGS

---

## Quick Navigation

### For Codex (Start Here)

1. **HANDOFF_TO_CODEX.md** ← START HERE
   - What you're building
   - Why it matters
   - Success criteria
   - Time estimate (1-2 hours)

2. **CODEX_SOP.json** ← DETAILED INSTRUCTIONS
   - Step-by-step execution plan
   - 8-phase lifecycle
   - Error handling
   - Monitoring and logging

3. **PHASE0_IMPLEMENTATION_GUIDE.md** ← DETAILED SPECIFICATION
   - Exact schema structures
   - Testing procedures
   - Example fixtures
   - Validation rules

### For Claude (Architecture & Context)

1. **README.md**
   - Department overview
   - Phase structure
   - Success criteria
   - Integration path

2. **ROADMAP.md**
   - All 7 phases
   - Build order and dependencies
   - Metrics and exit criteria
   - Non-goals

3. **TESTBENCH.md**
   - PoC testing strategy
   - Phase 0-2 test cases
   - Execution instructions
   - Report format

---

## Document Map

```
CATALYTIC-DPT/
│
├── INDEX.md                              (YOU ARE HERE)
├── README.md                             (Department overview)
├── ROADMAP.md                            (Phases 0-7 plan)
├── TESTBENCH.md                          (Testing strategy)
├── CODEX_SOP.json                        (Autonomous execution manual)
├── HANDOFF_TO_CODEX.md                   (Task description)
├── PHASE0_IMPLEMENTATION_GUIDE.md        (Detailed spec)
│
├── PRIMITIVES/
│   ├── __init__.py
│   ├── catalytic_store.py                (Phase 1 - TO DO)
│   ├── merkle.py                         (Phase 1 - TO DO)
│   ├── spectral_codec.py                 (Phase 1 - TO DO)
│   ├── ledger.py                         (Phase 1 - TO DO)
│   ├── validator.py                      (Phase 1 - TO DO)
│   └── micro_orchestrator.py             (Phase 1 - TO DO)
│
├── SCHEMAS/
│   ├── README.md                         (Schema spec)
│   ├── jobspec.schema.json               (Phase 0 - Codex creates)
│   ├── validation_error.schema.json      (Phase 0 - Codex creates)
│   └── ledger.schema.json                (Phase 0 - Codex creates)
│
├── TESTBENCH/
│   ├── run_poc.py                        (Test runner - TO DO)
│   ├── smoke_test.json                   (Minimal test - TO DO)
│   └── report.md                         (Test results - TO DO)
│
├── FIXTURES/
│   ├── phase0/
│   │   ├── valid/                        (Codex populates)
│   │   └── invalid/                      (Codex populates)
│   └── phase1/
│       ├── store_tests/                  (Phase 1)
│       ├── merkle_tests/
│       └── ledger_tests/
│
└── SKILLS/
    ├── catlab-jobspec-schema/            (Phase 1)
    ├── catlab-validator/                 (Phase 1)
    └── catlab-executor/                  (Phase 1)
```

---

## Current Status

### Phase 0: Freeze the Contract
- [x] README.md (overview)
- [x] ROADMAP.md (full phases)
- [x] TESTBENCH.md (testing)
- [x] CODEX_SOP.json (execution manual)
- [x] HANDOFF_TO_CODEX.md (task description)
- [x] PHASE0_IMPLEMENTATION_GUIDE.md (detailed spec)
- [x] SCHEMAS/README.md (schema specification)
- [ ] jobspec.schema.json (Codex creates)
- [ ] validation_error.schema.json (Codex creates)
- [ ] ledger.schema.json (Codex creates)
- [ ] FIXTURES/phase0/ examples (Codex creates)

### Phase 1: CATLAB (Waiting for Phase 0)
- [ ] catalytic_store.py
- [ ] merkle.py
- [ ] spectral_codec.py
- [ ] ledger.py
- [ ] validator.py
- [ ] micro_orchestrator.py
- [ ] Unit tests
- [ ] Fixtures

### Phase 2-7: Future
- Not started

---

## Codex's Checklist

**Before you start**:
1. [ ] Read HANDOFF_TO_CODEX.md
2. [ ] Read CODEX_SOP.json (all 8 phases)
3. [ ] Read PHASE0_IMPLEMENTATION_GUIDE.md (the spec)
4. [ ] Read SCHEMAS/README.md (domain knowledge)

**Your task**:
1. [ ] Create jobspec.schema.json (120 lines)
2. [ ] Create validation_error.schema.json (80 lines)
3. [ ] Create ledger.schema.json (140 lines)
4. [ ] Create valid fixtures in FIXTURES/phase0/valid/
5. [ ] Create invalid fixtures in FIXTURES/phase0/invalid/
6. [ ] Verify all schemas are valid JSON Schema Draft 7
7. [ ] Verify all schemas validate example data correctly
8. [ ] Run critic_run (pre and post)
9. [ ] Generate run ledger in CONTRACTS/_runs/<run_id>/
10. [ ] Report to Claude with success summary

**Estimated time**: 1-2 hours

**Success**: All three schemas created, documented, tested, and validated.

---

## For Integration (Later)

Once Phase 0 is done and Phase 1 passes:

1. **Skills Migration**
   - Wrap each primitive as an AGS skill
   - `SKILLS/catalytic-store/`
   - `SKILLS/catalytic-validator/`
   - etc.

2. **CI Integration**
   - Add catalytic tests to `CONTRACTS/runner.py`
   - Add catalytic checks to `.github/workflows/`

3. **Critic Upgrade**
   - Extend `TOOLS/critic.py` to enforce catalytic rules
   - Check forbidden overlaps
   - Check output existence
   - Check restoration proofs

4. **Lane F Integration**
   - Merge CATALYTIC-DPT into AGS_ROADMAP_MASTER.md Lane F
   - Document in CANON/CATALYTIC_COMPUTING.md
   - Update AGENTS.md with catalytic constraints

---

## Key References

### Theory & Vision
- `CONTEXT/research/Catalytic Computing/CATALYTIC_ROADMAP.md` - Full vision
- `CONTEXT/research/Catalytic Computing/MASTER_PLAN.md` - Implementation blueprint
- `CANON/CATALYTIC_COMPUTING.md` - Canonical definition

### System Context
- `AGS_ROADMAP_MASTER.md` - Full system roadmap (Lane F = catalytic)
- `AGENTS.md` - Agent governance rules
- `CANON/CONTRACT.md` - Execution contracts

### Implementation
- `TOOLS/catalytic_runtime.py` - Existing runtime (supports Phase 0-1)
- `TOOLS/catalytic_validator.py` - Existing validator (supports Phase 0-1)
- `SKILLS/swarm-governor/` - Token offloading skill
- `d:/CCC 2.0/AI/AGI/` - Swarm orchestration system

---

## Communication Channels

**Codex → Claude**:
- Report via return format in CODEX_SOP.json Section 7
- Include: run_id, status, paths, validation_report, decision_log_path
- Escalate on: governance failures, validation failures, timeouts, uncertain requirements

**Claude → Codex**:
- Handoff via HANDOFF_TO_CODEX.md
- Next task via Phase 1 JobSpec (after Phase 0 passes)

---

## Success Definition

**Phase 0 Complete When**:

```json
{
  "status": "success",
  "phase": 0,
  "deliverables": [
    "CATALYTIC-DPT/SCHEMAS/jobspec.schema.json",
    "CATALYTIC-DPT/SCHEMAS/validation_error.schema.json",
    "CATALYTIC-DPT/SCHEMAS/ledger.schema.json"
  ],
  "validation": {
    "all_schemas_valid_json": true,
    "all_schemas_valid_json_schema_draft7": true,
    "all_examples_pass": true,
    "restoration_verified": true,
    "no_governance_violations": true
  },
  "execution_time_seconds": 4200,
  "decisions_made": 12,
  "next_phase": 1
}
```

---

## Questions?

**For Codex**: If stuck, escalate to Claude. Refer to CODEX_SOP.json Section "error_handling.uncertain_requirement".

**For Claude**: This is a complete, isolated, testable R&D environment. Codex can execute Phase 0 autonomously. Phase 1 onwards depends on Phase 0 success.

**PoC First. Perfect Execution. Then Scale.**

---

**Generated**: 2025-12-23
**For**: Codex (200M orchestrator) + Claude (User Agent)
**Status**: Ready for Codex Phase 0 execution
