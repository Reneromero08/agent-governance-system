# CATALYTIC-DPT Skills

**Purpose**: Distributed execution skills for multi-agent orchestration
**Architecture**: Gemini (analysis) → Grok (execution) → MCP (governance)

---

## Skill Directory

### Core Orchestration Skills

#### 1. **grok-executor/** - Distributed Task Executor
- **Purpose**: Execute individual tasks via Grok 1.5 Fast workers
- **Tasks**: File operations, code adaptation, validation, research
- **Model**: Grok 1.5 Fast (free, fast, parallel)
- **Integration**: MCP mediator, hash verification, immutable ledger
- **Status**: ✅ Implementation complete

**Key Features**:
- Copy files with SHA-256 verification
- Code find/replace adaptations
- Validation checking
- Immutable audit trail in CONTRACTS/_runs/

**Usage**:
```bash
python grok-executor/run.py task.json output.json
```

**Files**:
- `run.py` - Main executor implementation
- `SKILL.md` - Specification
- `schema.json` - Input/output schema
- `fixtures/` - Example task JSONs
- `test_grok_executor.py` - Test harness

---

#### 2. **conductor-task-builder/** - Task Spec Builder
- **Purpose**: Convert Claude's goals into Conductor task specs
- **Input**: Natural language goal
- **Output**: Task specs for Grok workers
- **Model**: Claude (planning) + Gemini (analysis)
- **Status**: 🔄 Design complete

**Key Features**:
- Goal analysis and decomposition
- Optimal worker distribution
- Dependency tracking
- Execution planning

**Usage**:
```bash
python conductor-task-builder/run.py goal.json output_tasks.json
```

---

#### 3. **Conductor (Gemini CLI)** - Parallel Task Distributor (Built-in)
- **Purpose**: Distribute parallel work across Grok workers
- **Task**: Phase 0/1/2 parallel schema validation
- **Model**: Gemini 2.0 with --experimental-acp (built-in Conductor)
- **Status**: ✅ Already available (swarm-governor IS gemini cli)

**Key Features**:
- Task analysis and decomposition (automatic)
- Parallel worker distribution (automatic)
- Progress monitoring (automatic)
- Result aggregation (automatic)
- Integrated with Grok via grok-executor skill

**Usage**:
```bash
# In YOUR terminal
gemini --experimental-acp

# Then ask Gemini
> Validate all Phase 0 schemas in parallel
```

**Note**: "Swarm governor is gemini cli" - Don't import/adapt, use Conductor directly

---

### Previous Integration Skills

#### 4. **gemini-file-analyzer/** - Repository Analysis
- **Purpose**: Analyze AGI repo structure to identify critical files
- **Model**: Gemini CLI
- **Tasks**: analyze_swarm, find_gemini_config, identify_dependencies, list_critical_files
- **Status**: ✅ Created (Phase 4)

---

#### 5. **gemini-executor/** - General-purpose Gemini Wrapper
- **Purpose**: Execute Gemini CLI prompts from Claude
- **Model**: Gemini 2.0
- **Integration**: Runs in YOUR VSCode terminal (not Claude's)
- **Status**: ✅ Created (Phase 4)

---

## Execution Flow

### Simple Task (Direct Grok)

```
Claude: "Copy file A to file B with verification"
    ↓
grok-executor (file_operation: copy)
    ├─ Reads: A
    ├─ Computes: SHA256(A) = hash1
    ├─ Copies: B
    ├─ Computes: SHA256(B) = hash2
    ├─ Verifies: hash1 == hash2
    └─ Logs: CONTRACTS/_runs/
```

### Complex Task (Via Conductor)

```
You (VSCode terminal): "Validate all Phase 0 schemas in parallel"
    ↓
gemini --experimental-acp (Conductor built-in to Gemini CLI)
    ├─ Analyzes: How many schemas? What checks needed?
    ├─ Decomposes: Creates N subtasks (one per schema)
    └─ Distributes: Sends to available Grok workers
    ↓
Grok workers execute (via grok-executor skill)
    ├─ Grok-1: grok-executor (validate schema-1)
    ├─ Grok-2: grok-executor (validate schema-2)
    ├─ Grok-3: grok-executor (validate schema-3)
    └─ All call MCP for governance + logging
    ↓
Conductor aggregates results
    ↓
Logged to CONTRACTS/_runs/
    ↓
Claude sees results via MCP (bidirectional monitoring)
```

**No need to:**
- Import swarm-governor (Conductor IS the swarm governor)
- Adapt code (Grok already speaks Gemini CLI)
- Build task distribution (Conductor does it automatically)


---

## Skill Governance

### Rule 1: Canonical Skill Definition

Each skill has:
- `SKILL.md` - Immutable specification
- `VERSION.json` - Hash of current version
- `schema.json` - Input/output validation
- `run.py` - Implementation

Before execution:
```python
loaded_hash = sha256(SKILL.md + run.py)
expected_hash = json.load(VERSION.json)
assert loaded_hash == expected_hash, "Skill changed! Restart agents."
```

### Rule 2: MCP-Mediated Execution

No direct file writes. All operations via MCP:
- `mcp.file_sync()` - Copy/verify files
- `mcp.skill_execute()` - Run skills atomically
- `mcp.terminal_log_command()` - Log to shared terminals

### Rule 3: Immutable Ledger

Every operation logged to `CONTRACTS/_runs/<task_id>/`:
- `TASK_SPEC.json` - What was requested (immutable)
- `RESULTS.json` - What happened (immutable)
- `FILES_MODIFIED.json` - Which files touched (immutable)
- `HASHES_VERIFIED.json` - SHA-256 proofs (immutable)

---

## Testing

### Unit Tests

```bash
# Test grok-executor
python grok-executor/test_grok_executor.py
```

Results saved to `grok-executor/test_results.json`

### Integration Tests

```bash
# Test MCP server
cd ../MCP
python server.py
```

Verify:
- Terminal registration works
- File sync with hash verification works
- Ledger creation works

### End-to-End Test

```bash
# Test full workflow: Copy → Adapt → Validate
python grok-executor/run.py fixtures/file_copy_task.json output1.json
python grok-executor/run.py fixtures/code_adapt_task.json output2.json
python grok-executor/run.py fixtures/validate_task.json output3.json
```

---

## Fixture Files

Located in `grok-executor/fixtures/`:

- `file_copy_task.json` - Copy swarm-governor files
- `code_adapt_task.json` - Replace Cline → Gemini
- `validate_task.json` - Test adapted code
- `research_task.json` - Analyze code structure

---

## Quick Reference

| Component | Purpose | Model | Status |
|-----------|---------|-------|--------|
| **grok-executor** | Execute individual tasks | Grok 1.5 Fast | ✅ Ready |
| **Conductor** | Parallel task distribution | Gemini 2.0 --experimental-acp | ✅ Built-in |
| **MCP server** | Governance + logging | Custom Python | ✅ Ready |
| conductor-task-builder | Create task specs (optional) | Claude + Gemini | 🔄 Design |
| gemini-file-analyzer | Repo analysis (optional) | Gemini | ✅ Created |
| gemini-executor | General Gemini wrapper (optional) | Gemini | ✅ Created |

**Key Insight**: "Swarm governor is gemini cli" - Conductor is built-in, no import needed

---

## Next Steps

### Immediate (Ready Now)

1. ✅ Test grok-executor locally
2. ✅ Verify MCP server works
3. ✅ Test all fixtures

### Short-term (Next)

4. 🔄 Setup Gemini Conductor (`gemini --experimental-acp` in YOUR terminal)
5. 🔄 Test Conductor task distribution to Grok workers
6. 🔄 Verify MCP logs all Conductor operations
7. 🔄 Run Phase 0 parallel schema validation via Conductor

### Medium-term (After validation)

8. Scale to Phase 1 CATLAB execution (parallel primitives)
9. Run Phase 2+ autonomous tests via Conductor
10. Setup continuous Conductor loops for adaptive learning

---

## File Structure

```
CATALYTIC-DPT/SKILLS/
├── grok-executor/             ← Worker skill (core execution engine)
│   ├── run.py                 ← Main executor (400+ lines)
│   ├── SKILL.md               ← Specification
│   ├── schema.json            ← Input/output schema
│   ├── test_grok_executor.py  ← Test harness (validates all features)
│   └── fixtures/              ← Example tasks for testing
│       ├── file_copy_task.json
│       ├── code_adapt_task.json
│       ├── validate_task.json
│       └── research_task.json
│
├── conductor-task-builder/    ← Optional task spec builder
│   └── SKILL.md               ← Design (run.py optional)
│
├── gemini-file-analyzer/      ← Optional repo analyzer
│   ├── run.py
│   └── SKILL.md
│
├── gemini-executor/           ← Optional Gemini wrapper
│   ├── run.py
│   └── SKILL.md
│
└── README.md                  ← This file
```

**Essential files for full system**:
- ✅ `grok-executor/` (required - workers need this)
- ✅ `MCP/server.py` (required - governance)
- 🔄 `conductor-task-builder/` (optional - Conductor can build specs too)

**Not needed**:
- ❌ `swarm-governor-adapted/` (Conductor IS the swarm governor)

---

**Status**: Core infrastructure ready, testing in progress
**Integration**: MCP + Conductor + Grok workers = Zero drift execution

