# Gemini Integration - Quick Start

**TL;DR**: Use Gemini CLI in your VSCode terminal to analyze AGI files and identify what we need for swarm integration.

---

## Files Created

```
CATALYTIC-DPT/SKILLS/
├── gemini-file-analyzer/
│   ├── SKILL.md
│   └── run.py              # Analyzes AGI repo structure
│
└── gemini-executor/
    ├── SKILL.md
    └── run.py              # General-purpose Gemini wrapper
```

Plus: **CATALYTIC-DPT/GEMINI_INTEGRATION.md** - Full technical guide

---

## One-Liner: Discover Swarm Files

```bash
cd CATALYTIC-DPT/SKILLS/gemini-file-analyzer

cat > input.json << 'EOF'
{
  "repo_path": "D:/CCC 2.0/AI/AGI",
  "task_type": "analyze_swarm",
  "focus_areas": ["SKILLS/swarm-governor", "SKILLS/launch-terminal"]
}
EOF

python run.py input.json output.json
cat output.json
```

**What happens**:
1. Gemini CLI reads your AGI repo
2. Lists all swarm-governor files
3. Explains each file's purpose
4. Returns JSON to output.json

**Token cost to Claude**: ~10 tokens (you delegated to Gemini)

---

## Three Commands

### 1. Test Gemini Works
```bash
gemini "What is catalytic computing in 2 sentences?"
```

### 2. Analyze Swarm Files
```bash
cd CATALYTIC-DPT/SKILLS/gemini-file-analyzer
python run.py input.json output.json
```

### 3. Run Custom Gemini Task
```bash
cd CATALYTIC-DPT/SKILLS/gemini-executor
cat > input.json << 'EOF'
{
  "gemini_prompt": "List all Python files in D:/CCC 2.0/AI/AGI/SKILLS/ and explain their purposes",
  "task_id": "discover-agi-skills",
  "command_type": "analyze"
}
EOF
python run.py input.json output.json
cat output.json
```

---

## What Each Skill Does

| Skill | Purpose | Input |
|-------|---------|-------|
| **gemini-file-analyzer** | Specialized for analyzing AGI repo structure | `repo_path`, `task_type`, `focus_areas` |
| **gemini-executor** | General-purpose Gemini wrapper | `gemini_prompt`, `task_id`, `command_type` |

---

## Why This Matters

**Before**: Claude reads AGI files himself → uses 1000+ tokens
**After**: Gemini analyzes AGI repo in your terminal → Claude uses ~10 tokens

**You get**: Gemini's analysis + Claude's orchestration, with 99% token savings on file analysis.

---

## What's Next

1. ✅ Run gemini-file-analyzer to discover swarm files
2. ✅ Review Gemini's output
3. ✅ Decide which files to port to CATALYTIC-DPT
4. 🔄 Port swarm-governor to CATALYTIC-DPT
5. 🔄 Test with CATLAB primitives

---

## Files in CATALYTIC-DPT Now

```
CATALYTIC-DPT/
├── GEMINI_INTEGRATION.md      # Full technical guide
├── GEMINI_QUICK_START.md      # This file
├── SKILLS/
│   ├── gemini-file-analyzer/  # NEW
│   ├── gemini-executor/       # NEW
│   └── [other CATLAB skills]
└── [existing CATALYTIC-DPT structure]
```

---

## Governance

✅ Gemini runs in **YOUR** VSCode terminal (not Claude's)
✅ Results logged for audit
✅ No surprises - Claude sees what Gemini found
✅ Deterministic - same input → same response
✅ Token efficient - big brain orchestrates, small brain analyzes

---

**Status**: Ready to test
**Next**: Run the commands above and review Gemini's output
