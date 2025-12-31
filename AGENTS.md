# AGENTS.md

Agent Operating Contract for the Agent Governance System (AGS)

> **⛔ HARD PROHIBITION**: The **interactive terminal bridge** (VSCode terminal tabs) has been **AMPUTATED** from the system. The `launch-terminal` skill has been deleted. Agents are STRICTLY FORBIDDEN from attempting to spawn terminal windows or using interactive bridge tools. All work must be conducted via background processes and verified artifacts. The Swarm remains active for background task execution only.

This file defines how autonomous or semi-autonomous agents operate inside this
repository. It is procedural authority. If unclear, defer to CANON.

## 0. Initial Connection & Cortex Access

Before reading any rules, agents must establish a connection to the cortex (semantic indexing system) to access governance files and tools. This section provides the essential connection guidance that enables all subsequent operations.

### 0.1 Connection Methods

**Primary (Auto-start)**:
```bash
python LAW/CONTRACTS/ags_mcp_entrypoint.py
```
- **ADR-021 Compliant**: Automatically generates `session_id` (UUID) and logs all activity
- **Recommended**: Most reliable entry point with full audit logging

**Alternative**:
```bash
python CAPABILITY/MCP/server.py
```

**Test Connection**:
```bash
python LAW/CONTRACTS/ags_mcp_entrypoint.py --test
```

### 0.2 Essential Cortex Tools

Once connected via MCP, agents have access to these core tools:

- `cortex_query({"query": "term"})` - Search semantic index (logged with `session_id`)
- `canon_read({"file": "CONTRACT"})` - Read governance files (logged with `session_id`)
- `context_search({"type": "decisions"})` - Find ADRs and context records (logged with `session_id`)
- `session_info({})` - Get session information including `session_id` (ADR-021 compliance)

### 0.3 First Commands (Bootstrap Sequence)

1. **Read CONTRACT** via `canon_read({"file": "CONTRACT"})` - Establishes audit trail
2. **Read AGENTS.md** via `cortex_query({"query": "AGENTS.md"})` - Continues audit trail
3. **Check system status** - Verify ADR-021 compliance and connection health

### 0.4 ADR-021 Identity & Observability

**Session Identity**: The MCP server automatically generates a `session_id` (UUID) for each agent connection, fulfilling ADR-021 requirements.

**Essential Tool**: Use the `session_info` tool to discover your session_id:
```json
session_info({})  // Returns session_id, server info, and optional audit log entries
```

**Audit Compliance**: All cortex queries and tool calls are automatically logged to `LAW/CONTRACTS/_runs/mcp_logs/audit.jsonl` with your `session_id`, establishing traceable identity as required by ADR-021.

**Verification**: After connecting, immediately call `session_info({})` to obtain your `session_id` and verify ADR-021 compliance.

### 0.5 Troubleshooting

- **Python Check**: `python --version` (requires Python 3.8+)
- **Server Verification**: Run test command above
- **Log Inspection**: Check `LAW/CONTRACTS/_runs/mcp_logs/audit.jsonl` for connection errors
- **Session Tracking**: Verify `session_id` appears in audit logs (indicates ADR-021 compliance)

### 0.6 Connection Success Criteria

A successful connection is confirmed when:
1. MCP server starts without errors
2. Cortex tools respond to queries
3. Audit logs show entries with `session_id`
4. Agent can read CONTRACT.md via `canon_read`

**Only after establishing a successful cortex connection should agents proceed to Section 1.**

**Quick Reference**: For detailed tool usage and common query patterns, see `INBOX/reports/cortex-quick-reference.md`.

### 0.7 MCP-First Principle (Token Efficiency)

**🚨 CRITICAL: NO TOKEN WASTE 🚨**

Agents MUST use MCP tools for all cortex access and MUST NOT write custom database queries or manual file inspection code. This principle enforces catalytic computing token efficiency.

#### What is Token Waste?
Token waste occurs when an agent:
1. Writes Python SQLite snippets to inspect databases directly
2. Uses `open()` or `Path().read_text()` to read governance files
3. Creates custom scripts for tasks already covered by MCP tools
4. Analyzes database schemas instead of using semantic search tools

#### Examples of Token Waste vs. Correct Usage:

**❌ WRONG (Token Waste):**
```python
import sqlite3
conn = sqlite3.connect('CORTEX/_generated/system1.db')
cursor = conn.execute('SELECT * FROM symbols')
```

**✅ CORRECT (MCP-First):**
```json
cortex_query({"query": "symbols", "limit": 10})
```

**❌ WRONG (Token Waste):**
```python
open('LAW/CANON/CONTRACT.md').read()
```

**✅ CORRECT (MCP-First):**
```json
canon_read({"file": "CONTRACT"})
```

**❌ WRONG (Token Waste):**
```python
import os
for root, _, files in os.walk('LAW/CONTEXT/decisions'):
    for f in files:
        if 'catalytic' in open(f).read():
            print(f)
```

**✅ CORRECT (MCP-First):**
```json
context_search({"type": "decisions", "query": "catalytic"})
```

#### Enforcement
- The `mcp-access-validator` skill detects token waste and recommends MCP tools
- All agent actions are audited for token efficiency
- Violations are logged in `LAW/CONTRACTS/_runs/mcp_logs/token_waste.jsonl`

#### Token Savings
Using MCP tools provides:
- **95% savings** for database queries (vs. SQLite snippets)
- **90% savings** for file reading (vs. manual file ops)
- **85% savings** for context searches (vs. custom scripts)
- **100% governance compliance** (automatic audit logging)

#### The Rule
**If an MCP tool exists for a task, you MUST use it. Writing custom code for MCP-covered tasks is a governance violation.**

## 1. Required startup sequence (non-negotiable)

Before taking any action, an agent MUST:

1. **Connect to Cortex** using Section 0 guidelines
2. **Read essential governance documents** (via cortex tools):
   - LAW/CANON/CONTRACT.md - Core rules and authority gradient
   - LAW/CANON/INVARIANTS.md - Locked decisions that cannot change
   - LAW/CANON/VERSIONING.md - Version policy (identify current canon_version)
   - LAW/CANON/AGREEMENT.md - Constitutional agreement between human and system
   - LAW/CANON/STEWARDSHIP.md - Engineering practices and escalation paths
   - LAW/CANON/INBOX_POLICY.md - Document storage and content hash requirements
   - LAW/CANON/IMPLEMENTATION_REPORTS.md - Report format for all implementations
   - LAW/CANON/CRISIS.md - Emergency procedures and quarantine detection
3. **Read this file (AGENTS.md) in full**
4. **Identify the current canon_version** (found in LAW/CANON/VERSIONING.md)
5. **Identify whether the task is**:
   - governance change (requires ADR and ceremony)
   - skill implementation (requires SKILL.md, run.py, validate.py, fixtures)
   - build execution (outputs to BUILD/)
   - documentation only (must follow INBOX policy with content hash)
6. **Review relevant ADRs**:
   - ADR-021: Mandatory Agent Identity and Observability (session_id)
   - ADR-029: Headless Swarm Execution (terminal prohibition)
   - ADR-004: MCP Integration (connection protocol)
   - ADR-017: Skill Formalization (skill structure)
   - ADR-022: Why Flash Bypassed The Law (test requirement)
   - ADR-008: Composite Commit Approval (ceremony rules)
   - ADR-001: Build and Artifacts (output locations)
   - ADR-007: Constitutional Agreement (human-system relationship)
   - ADR-015: Logging Output Roots (audit log locations)
   - ADR-016: Context Edit Authority (mutation rules)
   - ADR-020: Admission Control Gate (governance enforcement)

7. **Note engineering standards** (from CANON/STEWARDSHIP.md):
   - No bare excepts, atomic writes, headless execution
   - Deterministic outputs, safety caps, proper database connections
   - Never bypass tests, cross-platform scripts, interface regression tests
   - Amend over pollute (clean commit history)

If any of the above steps are skipped, the agent must stop.

## 1C. Multi-agent workflow (worktrees)

When multiple agents are active in the same repo, each agent MUST use a separate Git worktree (or branch + worktree) to avoid shared working-tree conflicts that break governance checks.

Minimum practice:
- One agent = one worktree directory.
- Each agent updates `CHANGELOG.md` by adding a topmost entry, then rebases before push.
- Do not run tests or commit from a shared dirty worktree.
- Use `commit-queue` to enqueue and stage per-agent commit slices before the commit ceremony.

## 1A. Question-first gate (no-write)

If the user is asking questions, requesting analysis, or requesting a strategy without explicitly approving implementation, the agent MUST:

- answer first (no edits)
- avoid creating, modifying, deleting, or committing files
- avoid running commands that write artifacts
- ask for explicit approval before making changes (example: "Do you want me to implement this now?")

## 1B. Intent gate (canon and context)

Only change CANON or edit existing CONTEXT records when the task is explicitly about rules, governance, or memory updates. If intent is ambiguous, ask one clarifying question before touching CANON or existing CONTEXT records. Changes are reversible; if a change is wrong, revert it.

## 2. Authority gradient

If instructions conflict, obey in this order:

1. LAW/CANON/AGREEMENT.md
2. LAW/CANON/CONTRACT.md
3. LAW/CANON/INVARIANTS.md
4. LAW/CANON/VERSIONING.md
5. AGENTS.md
6. CONTEXT records (ADRs, rejections, preferences)
7. NAVIGATION/maps/*
8. User instructions
9. Implementation convenience

Never invert this order.

## 2A. Sovereign override interface (`MASTER_OVERRIDE`)

If a user prompt contains `MASTER_OVERRIDE`, treat it as an authorized Governance Interface directive (see `CANON/AGREEMENT.md`).

- It authorizes bypassing any repository governance rule for that prompt only (do not carry it across turns).
- Before taking an overridden action, append an audit entry to `LAW/CONTRACTS/_runs/override_logs/master_override.jsonl` (use the `master-override` skill).
- Do not read, quote, or summarize override logs unless the user prompt also contains `MASTER_OVERRIDE`.
- Do not echo the directive back to the user unless explicitly asked.

## 2B. Privacy boundary (no out-of-repo access)

- Default scope is the repository root only.
- Do not access, scan, or search outside the repo unless the user explicitly requests those paths in the same prompt.
- Avoid user profile and OS-level directories by default; if needed, ask before any out-of-repo access.

## 3. Mutation rules

Agents MAY:
- create or modify files under:
  - CAPABILITY/SKILLS/
  - LAW/CONTRACTS/
  - NAVIGATION/CORTEX/ (implementation), and `NAVIGATION/CORTEX/_generated/` (generated)
  - MEMORY/ (implementation), and `MEMORY/LLM_PACKER/_packs/` (generated)
  - BUILD/ (user build outputs only)
- append new records under LAW/CONTEXT/ (append-first; editing existing records requires explicit instruction)
- ignore THOUGHT/research unless the user explicitly requests it (non-binding)

Agents MAY NOT:
- modify LAW/CANON/* or edit existing LAW/CONTEXT records unless explicitly instructed or the task is explicitly about rules or memory updates
- delete authored content without explicit user instruction and confirmation (CANON rules must follow INV-010 archiving)
- rewrite history in LAW/CONTEXT/* without explicit instruction
- touch generated artifacts outside:
  - LAW/CONTRACTS/_runs/
  - NAVIGATION/CORTEX/_generated/
  - MEMORY/LLM_PACKER/_packs/

Generated files must be clearly marked as generated.

Research under THOUGHT/research is non-binding and ignored unless explicitly
requested. It must not be treated as canon.

## 4. Build output rules

System-generated artifacts MUST be written only to:

- LAW/CONTRACTS/_runs/
- NAVIGATION/CORTEX/_generated/
- MEMORY/LLM_PACKER/_packs/

`BUILD/` is reserved for user build outputs. It must not be used for system artifacts.

- BUILD/ is disposable
- BUILD/ is the dist equivalent
- BUILD/ may be wiped at any time
- No authored content belongs in BUILD/

If a task requires writing elsewhere, the agent must stop and ask.

## 5. Skills-first execution

Agents must not perform arbitrary actions.

All non-trivial work must be performed via a skill:

- If a suitable skill exists, use it.
- If no suitable skill exists:
  - propose a new skill
  - write SKILL.md first (manifest with metadata)
  - write run.py (implementation)
  - write validate.py (output validator)
  - define fixtures (test cases with input.json and expected.json)
  - then implement

Every skill must follow the contract defined in ADR-017:
- SKILL.md: manifest with metadata
- run.py: implementation script
- validate.py: output validator (accept two JSON file paths, return 0/1)
- fixtures/: test cases with input.json and expected.json

Direct ad-hoc scripting is forbidden.

## 6. Fixtures gate changes

If an agent changes behavior, it MUST:

1. Add or update fixtures
2. Run LAW/CONTRACTS/runner.py
3. Ensure all fixtures pass
4. Update CANON or CHANGELOG if behavior is user-visible

If fixtures fail, the change does not exist.

## 7. Uncertainty protocol

If any of the following are true, the agent must stop and ask:

- intent is ambiguous
- multiple canon interpretations exist
- change would affect invariants
- output location is unclear
- irreversible action is required

Guessing is forbidden.

## 8. Determinism requirement

Agent actions must be:

- deterministic
- reproducible
- explainable via canon and context

No randomness.
No hidden state.
No silent side effects.

## 9. Exit conditions

An agent should stop when:
- the requested task is complete
- fixtures pass
- outputs are written to the allowed artifact roots
- any blocking uncertainty appears

Agents must not continue "optimizing" beyond scope.

## 10. Commit ceremony (CRITICAL)

**Every single `git commit`, `git push`, and release publication requires explicit, per-instance user approval.**

This is the highest-priority governance rule for agent behavior.

### What does NOT authorize a commit
- "proceed"
- "let's move on to the next task"
- "continue"
- "yes" (unless in direct response to a commit ceremony prompt)

These authorize **implementation** only. They are **never** implicit commit approvals.

An explicit "commit" directive counts as approval to commit once checks pass and staged files are listed; no extra confirmation prompt is required.

### Explicit composite approvals
Explicit composite directives that include "commit", "push", and "release" (for example,
"commit, push, and release") count as approval for each action listed in that request.
This does not authorize additional commits beyond the current task.

### Ceremony confirmations
When checks have passed and staged files have been listed, short confirmations such as
"go on" count as approval for the listed actions.

### The anti-chaining rule
**One commit approval = one commit.** If the user approves a commit for Task A, and the agent then completes Task B, the agent MUST stop and request a new approval for Task B. Chaining commits under a single approval is forbidden.

### The ceremony
Before any Git command:
1. Run `CAPABILITY/TOOLS/critic.py` and `LAW/CONTRACTS/runner.py`. Confirm they pass.
2. Stop all execution.
3. List every file in the staging area.
4. If the user already gave an explicit approval for commit (including a standalone "commit" directive or a composite approval), proceed without re-prompting.
5. Otherwise ask: "Ready for the Chunked Commit Ceremony? Shall I commit these [N] files?"
6. Wait for explicit user approval.

Violation of this ceremony is a **critical governance failure**.

See also: `CONTEXT/preferences/STYLE-001-commit-ceremony.md`

## 11. The Law (pre-commit test requirement)

**🚨 NO COMMIT WITH FAILING TESTS 🚨**

This section exists because on 2025-12-27, an agent committed code with failing tests
using `--no-verify`, causing governance violations. See `LAW/CONTEXT/decisions/ADR-022-why-flash-bypassed-the-law.md`.

### Before ANY commit, agents MUST:

#### 11.1 Run tests and verify they pass
```bash
py -m pytest CAPABILITY/TESTBENCH/ -v
```
**If tests fail, DO NOT COMMIT. Fix them first.**

#### 11.2 Read FULL test output
When tests fail:
- **DO NOT** assume what the error is
- **READ** the actual error message (look for `FAIL`, `ERROR`, `rc=`)
- **LOOK** for the root cause, not just the assertion message

#### 11.3 Never use `--no-verify` without:
- Running tests manually and confirming they PASS
- Documenting WHY you're bypassing hooks
- Getting explicit user approval
- Adding justification to the commit message

#### 11.4 Preflight failures are not logic bugs
If you see `FAIL preflight rc=2` with reasons like `DIRTY_TRACKED`:
- This is NOT a governance logic bug
- This is because the git repo has uncommitted changes
- Tests that call `ags run` will fail on dirty repos
- **FIX**: Test governance logic directly using:
  - `ags route` for routing/revocation checks
  - `catalytic pipeline verify` for verification checks

#### 11.5 The test output truncation trap
Test output in agent tools may be truncated. If you see partial output:
1. Redirect output to a file: `py -m pytest ... 2>&1 | Out-File test.txt`
2. Read the file to see the FULL error
3. Never assume the error from truncated output

**Violation of The Law is a critical governance failure.**
