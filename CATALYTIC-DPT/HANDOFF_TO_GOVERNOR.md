# HANDOFF TO GOVERNOR: Catalytic Computing PoC

**From**: President (Orchestrator)
**To**: Governor (Manager / CLI Agent)
**Date**: 2025-12-24
**Goal**: Execute Phase 0 (define contracts) with no human intervention.

---

## What You Need to Know in 2 Minutes

1. **What**: Build Phase 0 of catalytic computing - three JSON schemas that everything else depends on
2. **Where**: `CATALYTIC-DPT/` folder (isolated R&D, not touching main AGS)
3. **How**: Follow the SOP in `CATALYTIC-DPT/GOVERNOR_SOP.json`
4. **Success**: All schemas defined, tested, documented
5. **Escalate**: If anything is uncertain, report back to President (Orchestrator)

---

## Your Task: Phase 0 - Freeze the Contract

### Deliverable 1: JobSpec Schema

**File**: `CATALYTIC-DPT/SCHEMAS/jobspec.schema.json`

**What it is**: A JSON Schema that defines "what is a valid catalytic job specification?"

**Read**: `CATALYTIC-DPT/SCHEMAS/README.md` Section "The Three Schemas > 1. JobSpec Schema"

**Requirements**:
- Must be valid JSON Schema Draft 7
- Must have: job_id, phase, task_type, intent, inputs, outputs, catalytic_domains
- Must validate itself (run the schema validator on the schema)
- Must come with examples (valid and invalid) in `CATALYTIC-DPT/FIXTURES/phase0/`

**Exit Criteria**:
- [ ] File exists and is valid JSON
- [ ] File is valid JSON Schema Draft 7
- [ ] File includes clear field descriptions
- [ ] Examples work (valid examples pass, invalid examples fail)

---

### Deliverable 2: Validation Error Vector Schema

**File**: `CATALYTIC-DPT/SCHEMAS/validation_error.schema.json`

**What it is**: A JSON Schema that defines "what does a deterministic error report look like?"

**Read**: `CATALYTIC-DPT/SCHEMAS/README.md` Section "The Three Schemas > 2. Validation Error Vector"

**Requirements**:
- Must define `valid` (boolean), `errors` (array), `warnings` (array), `timestamp`, `validator_version`
- Must define error codes: SCHEMA_INVALID, MISSING_REQUIRED_FIELD, INVALID_FIELD_TYPE, INVALID_ENUM_VALUE, etc.
- Error codes must be UPPERCASE_SNAKE_CASE (no ambiguity)
- Must be valid JSON Schema Draft 7
- Must include examples of error and warning objects

**Exit Criteria**:
- [ ] File exists and is valid JSON
- [ ] File is valid JSON Schema Draft 7
- [ ] All error codes documented
- [ ] Examples work

---

### Deliverable 3: Ledger Schema

**File**: `CATALYTIC-DPT/SCHEMAS/ledger.schema.json`

**What it is**: A JSON Schema that defines "what does a complete run audit trail look like?"

**Read**: `CATALYTIC-DPT/SCHEMAS/README.md` Section "The Three Schemas > 3. Ledger Schema"

**Requirements**:
- Must define structure of RUN_INFO, PRE_MANIFEST, POST_MANIFEST, RESTORE_DIFF, OUTPUTS, STATUS
- Must reference jobspec.schema.json (if a ledger contains a jobspec)
- Must be valid JSON Schema Draft 7
- Must include real examples

**Exit Criteria**:
- [ ] File exists and is valid JSON
- [ ] File is valid JSON Schema Draft 7
- [ ] All sub-schemas defined
- [ ] Examples work

---

## Files Already Created For You

These are your instruction manuals:

1. **CATALYTIC-DPT/README.md** - Overview of the department
2. **CATALYTIC-DPT/ROADMAP.md** - Phases 0-7 with deliverables
3. **CATALYTIC-DPT/TESTBENCH.md** - How to validate Phase 1+
4. **CATALYTIC-DPT/GOVERNOR_SOP.json** - Your operating manual (FOLLOW THIS)
5. **CATALYTIC-DPT/SCHEMAS/README.md** - Detailed spec for Phase 0 contracts

Directory structure:
```
CATALYTIC-DPT/
├── README.md                 # READ FIRST
├── ROADMAP.md
├── TESTBENCH.md
├── GOVERNOR_SOP.json            # FOLLOW THIS
├── PRIMITIVES/               # (Phase 1, not your job yet)
├── SCHEMAS/
│   ├── README.md             # Your detailed spec
│   ├── jobspec.schema.json   # YOU CREATE THIS
│   ├── validation_error.schema.json  # YOU CREATE THIS
│   └── ledger.schema.json    # YOU CREATE THIS
├── TESTBENCH/
├── FIXTURES/
│   └── phase0/
│       ├── valid/            # (You can put examples here)
│       └── invalid/          # (You can put examples here)
└── SKILLS/
```

---

## Execution Steps (From GOVERNOR_SOP.json)

**Follow this exactly:**

1. **Parse Task**: You receive this handoff (it's your JobSpec)
2. **Governance Check**: Run `critic_run` - verify governance
3. **Generate Run ID**: Create unique ID like `phase0-contracts-20251223-143022`
4. **Decision Point**: Phase 0 is sequential (not parallel)
5. **Execute**: Write the three schema files
   - Use CATALYTIC-DPT/SCHEMAS/README.md as the spec
   - Make schemas valid JSON Schema Draft 7
   - Include clear field descriptions
   - Test each schema against valid/invalid examples
6. **Validate**: Check that schemas are self-validating
7. **Post-Flight Governance**: Run `critic_run` again
8. **Report Back**: Send President a summary with:
   - run_id
   - status (success|failed)
   - paths of created files
   - validation report
   - decision log
   - execution time

**Total expected time**: 1-2 hours

---

## What You're NOT Doing Yet

- No implementation of primitives (catalytic_store.py, merkle.py, etc.)
- No micro-orchestrator with weight updates
- No fixtures beyond schema examples
- No integration into AGS

**Phase 0 only**: Define the contracts. Nothing else.

---

## Success Looks Like

When you finish Phase 0:

```
CATALYTIC-DPT/TESTBENCH/_runs/phase0-contracts-20251223-143022/
├── RUN_INFO.json              # Metadata
├── PRE_MANIFEST.json          # State before
├── POST_MANIFEST.json         # State after (should match PRE)
├── RESTORE_DIFF.json          # Empty = success
├── OUTPUTS.json               # Three schema files created
├── STATUS.json                # {status: "restored", restoration_verified: true}
└── task_log.jsonl             # Every decision logged

CATALYTIC-DPT/SCHEMAS/
├── jobspec.schema.json        # Valid JSON Schema Draft 7
├── validation_error.schema.json  # Valid JSON Schema Draft 7
└── ledger.schema.json         # Valid JSON Schema Draft 7
```

And you report to President:
> Phase 0 COMPLETE. All three schemas created and validated. 8 decisions made. No governance violations. Ready for Phase 1. Run ledger: CATALYTIC-DPT/TESTBENCH/_runs/phase0-contracts-20251223-143022/

---

## MCP Tools Available to You

You can call these via MCP:

1. `critic_run` - Governance check (must pass before + after)
2. `catalytic_execute` - Execute task with restoration proof
3. `decision_log` - Log a decision to task_log.jsonl
4. `jobspec_validate` - Validate a JobSpec JSON
5. `catalytic_validate` - Validate a run ledger

For Phase 0, you mainly use:
- `critic_run` (pre and post)
- `catalytic_execute` (run task)
- `decision_log` (log decisions)
- `catalytic_validate` (verify ledger)

---

## Key Constraints

**Must NOT**:
- Modify CANON/* or AGENTS.md
- Write outside CATALYTIC-DPT/
- Create files in BUILD/ or LOGS/
- Bypass governance checks
- Guess on requirements (escalate instead)

**Must DO**:
- Log every decision to task_log.jsonl
- Run critic_run before and after
- Verify schemas are self-validating
- Document error codes deterministically
- Include examples for clarity

---

## If You Get Stuck

1. Escalate to President immediately
2. Include: what you were trying to do, what failed, relevant logs
3. Do NOT continue without clarification
4. Do NOT skip governance checks

---

## Next: After Phase 0 Succeeds

Phase 1 is CATLAB (Catalytic Learning A-B Testing Lab):

- Build catalytic_store.py (content-addressable storage)
- Build merkle.py (root digests)
- Build spectral_codec.py (domain encoding)
- Build ledger.py (receipt storage)
- Build validator.py (schema validation)
- Build micro_orchestrator.py (tiny model with weight updates)

But that's Phase 1. President will send you new instructions when Phase 0 is done.

---

## You Are Ready

You have:
- Clear task definition (Phase 0: three schemas)
- Detailed specification (SCHEMAS/README.md)
- Operating manual (GOVERNOR_SOP.json)
- Test criteria (TESTBENCH.md)
- Isolated environment (CATALYTIC-DPT/)

**Start now. Follow the SOP. Report back.**

🚀
