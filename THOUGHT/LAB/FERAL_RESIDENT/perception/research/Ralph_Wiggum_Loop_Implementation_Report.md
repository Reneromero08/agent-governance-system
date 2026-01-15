# Ralph Wiggum Loop Implementation Report

## Executive Summary

The Ralph Wiggum Loop is an autonomous iteration technique for Claude Code that transforms single-pass AI interactions into persistent, self-correcting development loops. Core principle: **"Iteration > Perfection"**—failures become data, the loop refuses to quit until success criteria are met.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Claude receives prompt                                   │
│ 2. Claude works (tool calls, file edits, tests)             │
│ 3. Claude attempts to exit                                  │
│ 4. Stop hook intercepts → checks completion promise         │
│ 5. Promise missing? Re-inject prompt, increment counter     │
│ 6. Claude sees modified filesystem, continues working       │
│ 7. Repeat until: promise found OR max-iterations reached    │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: File system + Git history = external memory. Each iteration sees prior work without context window bloat.

---

## Implementation Options

### Option 1: Official Plugin (Recommended)

```bash
# Install via Claude Code plugin system
/install ralph-wiggum@claude-plugins-official
```

**Usage:**
```bash
/ralph-loop "<prompt>" --max-iterations 20 --completion-promise "<promise>DONE</promise>"
```

**How it works:**
- Stop hook in `hooks/stop-hook.sh` intercepts exit
- Scans transcript for completion promise string
- Exit code 2 = continue loop, Exit code 0 = success
- Uses `jq` for JSON parsing (dependency)

### Option 2: Primitive Bash Loop

```bash
while :; do cat PROMPT.md | claude ; done
```

**WARNING**: No safety controls—infinite loop risk, token burn hazard.

### Option 3: Community Implementation (frankbria/ralph-claude-code)

```bash
git clone https://github.com/frankbria/ralph-claude-code.git
cd ralph-claude-code
./install.sh

# Usage
ralph -p PROMPT.md -c 100 -t 15 --max-iterations 20
```

**Features:**
- Rate limiting (100 calls/hour default)
- Circuit breaker with error detection
- tmux monitoring dashboard
- Session continuity
- Response analyzer

---

## Stop Hook Mechanism

The stop hook is the core innovation—it intercepts session termination and enforces completion criteria.

**hooks.json configuration:**
```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "${CLAUDE_PLUGIN_ROOT}/hooks/stop-hook.sh"
      }]
    }]
  }
}
```

**Exit Code Logic:**
| Code | Meaning | Action |
|------|---------|--------|
| 0 | Promise found | Loop terminates |
| 2 | Promise missing | Re-inject prompt, continue |

**Completion Promise Pattern:**
```
<promise>COMPLETE</promise>
```
- Exact string match (case-sensitive)
- Must appear in Claude's output
- Use unique, tamper-proof strings

---

## Prompt Engineering for Loops

### Template Structure

```markdown
# Task: [FEATURE_NAME]

## Requirements
- [Requirement 1]
- [Requirement 2]

## Success Criteria
- All tests passing
- No linter errors
- Documentation updated

## Process
1. Write failing tests
2. Implement feature
3. Run tests
4. Fix failures
5. Repeat until green

## Completion
Output <promise>COMPLETE</promise> when ALL criteria met.

## Stuck Protocol
After 15 iterations without progress:
- Document blockers
- List attempted approaches
- Output <promise>BLOCKED</promise>
```

### PROMPT.md for File-Based Loops

```markdown
# Ralph Demo: [PROJECT NAME]

## Tasks
- [ ] 1. Initialize project
- [ ] 2. Configure tooling
- [ ] 3. Set up structure
- [ ] 4. Implement core feature
- [ ] 5. **HARD STOP** - Checkpoint: verify functionality
- [ ] 6. Add tests
- [ ] 7. Final verification

## Success Criteria
- All tests pass
- Build completes without errors

## Instructions
Go through tasks step-by-step, check off each when complete.
When encountering HARD STOP, use AskUserQuestion for confirmation.
Output <promise>COMPLETE</promise> when finished.
```

---

## Ideal Use Cases

| Good Fit | Poor Fit |
|----------|----------|
| Test suite migration (Jest → Vitest) | Design decisions requiring judgment |
| Lint/formatting fixes | Novel algorithm design |
| Documentation generation | Tasks requiring long contiguous context |
| Framework migrations | Ambiguous specifications |
| Boilerplate implementation | Creative/exploratory work |
| TDD cycles | High human oversight needs |

---

## Safety Controls

### CRITICAL: Always set max-iterations

```bash
# Prevent runaway loops
/ralph-loop "Task..." --max-iterations 20
```

### Token Cost Awareness
- 50-iteration loop on large codebase: $50-100+ in API credits
- Monitor usage actively
- Start small, scale up

### Circuit Breaker Pattern
```bash
# Community implementation
ralph --circuit-status    # Check status
ralph --reset-circuit     # Reset after failures
```

---

## Windows Compatibility

**Known Issues:**
1. WSL bash resolution conflicts
2. Missing `jq` dependency
3. Path format mismatches (backslash vs forward slash)
4. Git Bash PATH issues (`cat: command not found`)

**Fixes:**

```json
// hooks.json - Force Git Bash
{
  "command": "\"C:\\Program Files\\Git\\usr\\bin\\bash.exe\" ${CLAUDE_PLUGIN_ROOT}/hooks/stop-hook.sh"
}
```

```bash
# Add to stop-hook.sh beginning
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH"
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

```bash
# Install jq for Windows
mkdir -p ~/bin
curl -L -o ~/bin/jq.exe "https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-windows-amd64.exe"
chmod +x ~/bin/jq.exe
```

**Best Solution**: Run Claude Code inside WSL for full compatibility.

---

## Permissions Configuration

```json
// .claude/settings.local.json
{
  "permissions": {
    "allow": [
      "Bash(**/ralph-wiggum/**)"
    ]
  }
}
```

---

## Multi-Phase Development

```bash
# Phase 1: Data models
/ralph-loop "Phase 1: Build data models with validation.
Output <promise>PHASE1_DONE</promise>" --max-iterations 20

# Phase 2: API layer
/ralph-loop "Phase 2: Build API endpoints for existing models.
Output <promise>PHASE2_DONE</promise>" --max-iterations 25

# Phase 3: Frontend
/ralph-loop "Phase 3: Build UI components.
Output <promise>PHASE3_DONE</promise>" --max-iterations 30
```

### Overnight Batch Script

```bash
#!/bin/bash
# overnight-work.sh

cd /path/to/project1
claude -p "/ralph-loop 'Task 1...' --max-iterations 50"

cd /path/to/project2
claude -p "/ralph-loop 'Task 2...' --max-iterations 50"
```

---

## Verification Loop Requirements

Success depends on **objective truth functions**:
- Test suites (pytest, jest, vitest)
- Linters (eslint, ruff)
- Type checkers (mypy, tsc)
- Build systems

**Without verification loops, the AI cannot determine success.**

---

## Context Management

| Strategy | Benefit |
|----------|---------|
| Fresh context per iteration | Prevents context rot |
| File system as memory | Persists across iterations |
| Git history | Rollback capability |
| Minimal prompt injection | Focused context |

---

## Implementation Checklist

- [ ] Install plugin: `/install ralph-wiggum@claude-plugins-official`
- [ ] Verify jq installed (Linux/Mac) or configure for Windows
- [ ] Create PROMPT.md with clear success criteria
- [ ] Define unique completion promise
- [ ] Set appropriate --max-iterations
- [ ] Configure permissions in settings.local.json
- [ ] Ensure test suite or verification mechanism exists
- [ ] Test with small task before overnight runs

---

## References

- Official Plugin: `github.com/anthropics/claude-code/tree/main/plugins/ralph-wiggum`
- Community Fork: `github.com/frankbria/ralph-claude-code`
- Documentation: `code.claude.com/docs/en/hooks`
