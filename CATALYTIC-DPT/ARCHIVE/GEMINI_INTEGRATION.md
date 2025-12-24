# Gemini Integration for CATALYTIC-DPT

**Date**: 2025-12-23
**Status**: Ready to implement
**Purpose**: Offload swarm analysis and file discovery to Gemini CLI in YOUR environment

---

## Overview

**Problem**: Claude is at 87% token budget. Need to analyze AGI swarm architecture without burning more tokens.

**Solution**:
1. Use Gemini CLI (installed in your environment)
2. Gemini reads AGI repo files
3. Gemini identifies critical swarm files
4. Results come back to Claude with minimal token cost

**Key Insight**: Gemini runs in YOUR VSCode terminal, not Claude's sandbox.

---

## What You Have

✅ **Gemini CLI installed** (`/c/Users/rene_/AppData/Roaming/npm/gemini`)
✅ **VSCode Antigravity Bridge** (port 4000, launch-terminal skill)
✅ **Swarm-governor** (D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor)
✅ **Token budget**: Saved by delegating to Gemini

---

## What We Just Created

### 1. **CATALYTIC-DPT/SKILLS/gemini-file-analyzer/**
Analyzes AGI repo to identify critical files.

**Task Types**:
- `analyze_swarm`: Find swarm-governor files
- `find_gemini_config`: Locate Gemini config
- `identify_dependencies`: Map file dependencies
- `list_critical_files`: Minimum set to understand integration

**Usage**:
```bash
python CATALYTIC-DPT/SKILLS/gemini-file-analyzer/run.py input.json output.json
```

**Example Input** (`input.json`):
```json
{
  "repo_path": "D:/CCC 2.0/AI/AGI",
  "task_type": "analyze_swarm",
  "focus_areas": [
    "SKILLS/swarm-governor",
    "SKILLS/launch-terminal",
    "EXTENSIONS/antigravity-bridge"
  ]
}
```

### 2. **CATALYTIC-DPT/SKILLS/gemini-executor/**
General-purpose skill to run Gemini prompts from Claude.

**Command Types**:
- `analyze`: Read and analyze files
- `execute`: Run a command
- `research`: Deep research
- `report`: Generate a report

**Usage**:
```bash
python CATALYTIC-DPT/SKILLS/gemini-executor/run.py input.json output.json
```

**Example**: Ask Gemini to find which files we need from AGI

```json
{
  "gemini_prompt": "In D:/CCC 2.0/AI/AGI, list all Python files in SKILLS/swarm-governor/ and explain each one's purpose",
  "task_id": "discover-swarm-files",
  "command_type": "analyze"
}
```

---

## Workflow: File Discovery

### Step 1: Claude Delegates to Gemini

```
Claude (87% budget): "Gemini, what files do we need from AGI swarm-governor?"
    ↓
gemini-executor skill
    ↓
Gemini CLI in YOUR VSCode terminal
    ↓
Gemini analyzes D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/
    ↓
Results: "run.py (orchestrator), validate.py (validator), ..."
    ↓
Back to Claude (minimal tokens used)
```

### Step 2: Gemini Finds Files

```bash
$ gemini "List all Python files in D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/ with descriptions"

Response:
- run.py: Core swarm orchestrator. Manages worker lifecycle, task distribution, result collection.
- validate.py: Output validator. Checks task results against expected schema.
- __init__.py: Module initialization
```

### Step 3: Claude Receives Results

Claude gets Gemini's analysis without reading the files himself (tokens saved!).

---

## Integration with CATALYTIC-DPT

### Phase 1: File Discovery (NOW)

Use `gemini-file-analyzer` to answer:
- "What files make up swarm-governor?"
- "How does Antigravity Bridge connect VSCode?"
- "Which files are essential?"

**Goal**: Identify which files to port to CATALYTIC-DPT

### Phase 2: Port Swarm to CATALYTIC-DPT (NEXT)

Once Gemini identifies critical files:
1. Copy essential files
2. Adapt swarm-governor to use Gemini CLI
3. Test with CATALYTIC-DPT tasks

### Phase 3: Parallel Catalytic Tasks (LATER)

Use swarm-governor in CATALYTIC-DPT with Gemini:

```json
{
  "tasks": [
    {
      "id": "phase0-schema-validation",
      "prompt": "Using gemini-executor, verify all Phase 0 schemas are valid JSON Schema Draft 7",
      "model": "gemini"
    },
    {
      "id": "phase1-fixture-generation",
      "prompt": "Generate 100 valid and 50 invalid JobSpec examples for testing",
      "model": "gemini"
    }
  ],
  "num_workers": 2,
  "timeout": 300
}
```

---

## Testing the Integration

### Test 1: Simple Gemini Call

```bash
cd D:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT/SKILLS/gemini-executor

cat > test_input.json << 'EOF'
{
  "gemini_prompt": "List the top 5 files I should read to understand Python swarm orchestration",
  "task_id": "learn-swarm",
  "command_type": "research"
}
EOF

python run.py test_input.json test_output.json

# Check output
cat test_output.json
```

### Test 2: Analyze AGI Swarm

```bash
cd D:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT/SKILLS/gemini-file-analyzer

cat > analyze_input.json << 'EOF'
{
  "repo_path": "D:/CCC 2.0/AI/AGI",
  "task_type": "analyze_swarm",
  "focus_areas": [
    "SKILLS/swarm-governor",
    "SKILLS/launch-terminal"
  ]
}
EOF

python run.py analyze_input.json analyze_output.json

# Check what files Gemini identified
cat analyze_output.json | grep -A5 "findings"
```

### Test 3: Integration via Swarm

```bash
cat > swarm_gemini_analysis.json << 'EOF'
{
  "tasks": [
    {
      "id": "gemini-discover-swarm-files",
      "prompt": "Use gemini-executor to analyze D:/CCC 2.0/AI/AGI and list all critical files for swarm-governor understanding",
      "model": "gemini"
    }
  ],
  "num_workers": 1,
  "timeout": 60
}
EOF

python D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/run.py swarm_gemini_analysis.json swarm_output.json
```

---

## Architecture Diagram

```
CATALYTIC-DPT/
├── SKILLS/
│   ├── gemini-file-analyzer/      # Analyzes AGI repo
│   │   ├── SKILL.md
│   │   └── run.py                 # Calls Gemini CLI
│   │
│   ├── gemini-executor/           # General Gemini wrapper
│   │   ├── SKILL.md
│   │   └── run.py                 # Runs Gemini prompts
│   │
│   └── [Other CATLAB skills]
│
└── PRIMITIVES/
    └── [catalytic_store, merkle, spectral_codec, etc.]

↓

AGI/
├── SKILLS/
│   ├── swarm-governor/            # Orchestrates parallel workers
│   ├── launch-terminal/           # Sends commands to YOUR VSCode
│   └── ...
│
└── EXTENSIONS/
    └── antigravity-bridge/        # Port 4000, connects to VSCode
```

---

## Next Steps

### Immediate (Next 1-2 hours)

1. **Test gemini-executor**
   ```bash
   # Run a simple Gemini prompt to verify it works
   ```

2. **Run gemini-file-analyzer**
   ```bash
   # Analyze AGI swarm-governor to find critical files
   ```

3. **Review findings**
   - What files are essential?
   - What can be ported to CATALYTIC-DPT?
   - What's AGI-specific?

### Short-term (1-3 days)

4. **Port swarm-governor to CATALYTIC-DPT**
   - Copy essential files
   - Adapt to use Gemini CLI
   - Test with CATLAB tasks

5. **Create swarm-based catalytic tasks**
   - Phase 0 schema validation (parallel)
   - Phase 1 fixture generation (parallel)
   - Phase 2 testing (parallel)

### Medium-term (1 week)

6. **Full integration**
   - Swarm + Catalytic = parallel R&D
   - Gemini + Swarm = autonomous analysis
   - Claude + Swarm + Gemini = complete token offloading

---

## Governance Rules

1. **Gemini runs in YOUR terminal** - not Claude's sandbox
2. **Token budget preserved** - offload mechanical work
3. **All results logged** - audit trail maintained
4. **Deterministic** - same input → same Gemini response
5. **No surprises** - Claude reviews Gemini output before acting

---

## Key Files

| File | Purpose |
|------|---------|
| `CATALYTIC-DPT/SKILLS/gemini-file-analyzer/run.py` | Analyzes AGI repo |
| `CATALYTIC-DPT/SKILLS/gemini-executor/run.py` | General Gemini wrapper |
| `D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/run.py` | Parallel task orchestrator |
| `D:/CCC 2.0/AI/AGI/EXTENSIONS/antigravity-bridge/` | VSCode integration |

---

## Quick Reference

**Test Gemini**:
```bash
gemini --help
gemini -o json "What is catalytic computing?"
```

**Test gemini-executor**:
```bash
cd CATALYTIC-DPT/SKILLS/gemini-executor
echo '{"gemini_prompt": "Hello world"}' | ... run.py
```

**Integrate with swarm**:
```bash
python AGI/SKILLS/swarm-governor/run.py swarm_config.json output.json
```

---

**Status**: Ready for testing
**Owner**: You (user), with Claude guiding
**Token Impact**: ~90% reduction for file analysis tasks
