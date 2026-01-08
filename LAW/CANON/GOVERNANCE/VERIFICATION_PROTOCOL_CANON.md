# Verification Protocol
**Status:** CANON  
**Applies to:** Any task that modifies production code, enforcement primitives, receipts, fixtures, tests, schemas, or governance gates.  
**Exemptions:** Documentation-only changes (see `CANON/DOCUMENT_POLICY.md` for exempt paths: `LAW/CANON/*`, `LAW/CONTEXT/*`, `INBOX/*`, etc.) do not require full verification protocol unless they modify enforcement logic.  
**Goal:** Make correctness mechanical. Prevent “looks done” work. Ensure every completion claim is reproducible from command outputs and artifacts.

## Core principle
A catalytic system runs on **mechanical truth**, not intent.

- Narratives are cheap.  
- Proof is reproducible.  
- Verification is a loop, not a sentence.

If a claim cannot be reproduced from **commands, outputs, and receipts**, it is not verified.

## Relationship to existing invariants

This protocol reinforces and operationalizes several core invariants:

- **[INV-007] Change ceremony** (`CANON/INVARIANTS.md`) - Verification ensures that behavior changes include updated fixtures, changelog entries, and proof of correctness in the same commit.
- **[INV-013] Declared Truth** (`CANON/INVARIANTS.md`) - Verification outputs must be declared in hash manifests; if not hashed, it's not truth.
- **[INV-015] Narrative Independence** (`CANON/INVARIANTS.md`) - Verification success is bound to artifact integrity (receipts, test outputs, git status), not to reasoning traces or chat history.

The Verification Protocol makes these invariants **mechanically enforceable** by requiring verbatim proof and hard gates.

## Definitions
- **Verified:** All required checks executed and passed, with proof recorded.
- **Proof:** Verbatim command outputs captured in canonical artifacts (and optionally pasted inline).
- **Fail-closed:** Any ambiguity or failed check stops the task.
- **Clean state:** No unrelated diffs in the scoped paths before verification.

## Canonical invariants
### INV-VP-001: No verification without execution
An agent may not claim a task is complete unless it executed the required verification commands for that task.

### INV-VP-002: Proof must be recorded verbatim
A claim is not verified unless proof is recorded verbatim for:
- `git status`
- every required test command
- every required audit command (linters, formatters, schema checks, rg/grep gates), if any

**Summaries are not proof.**

### INV-VP-003: Tests are hard gates
A “test” that detects violations while still passing is invalid as a completion gate.

If a forbidden condition exists, the gate **must fail**.  
If the gate passes, it means **no violations remain in scope**.

Scanner-only “pass with findings” gates are forbidden.

### INV-VP-004: Deterministic stop conditions
If any mandatory verification step fails, the agent must:
1) fix (within scope), then  
2) re-run the same command(s), then  
3) record the new outputs, then  
4) repeat until pass

If the agent cannot fix within scope, it must stop and report **BLOCKED** with a precise reason.

### INV-VP-005: Clean-state discipline
Verification must be run from a clean state for the scoped paths.

If unrelated diffs exist, the agent must do one:
- STOP and report the diffs, or
- revert them, or
- explicitly scope them into the task (and record that change)

No verification on a polluted tree.

## Definition of Done
A task is **VERIFIED COMPLETE** only when all are true:

1) **Scope respected**
- No files changed outside allowed paths.

2) **Clean-state check passed**
- `git status` is clean except for explicitly scoped changes.

3) **All required tests pass**
- Includes task-specific tests and any broader suite required by the task contract.

4) **All required audits pass**
- Includes rg/grep enforcement gates, schema validations, linting, formatting, and any task-specific audits.

5) **Proof recorded**
- Outputs for each command are recorded verbatim in canonical artifacts.

Otherwise, the status is **PARTIAL** or **BLOCKED**.

## Where proof must live
All proof must live under canonical run artifacts, not INBOX.

Recommended structure (use existing repo conventions where applicable):
- `LAW/CONTRACTS/_runs/REPORTS/<phase>/<task>_report.md`
- `LAW/CONTRACTS/_runs/RECEIPTS/<phase>/<task>_receipt.json`
- `LAW/CONTRACTS/_runs/LOGS/<phase>/<task>/` (optional but recommended)

### Token-sustainable rule for large outputs
If outputs are too large to paste inline in a report:
- Save the full verbatim output to a log file under `_runs/LOGS/...`
- In the report, include:
  - log path
  - byte size
  - sha256 of the log file
  - first 100 lines and last 100 lines (verbatim)

This preserves mechanical truth without wasting tokens.

## Mandatory Verification Contract (copy into every task prompt)
> **VERIFICATION CONTRACT (NON-NEGOTIABLE)**  
> You must follow this loop until completion.

### STEP 0: CLEAN STATE
Run:
- `git status`

Rules:
- If changes exist outside allowed scope, STOP and report.

Record:
- verbatim `git status` output (inline or log + hash).

### STEP 1: RUN REQUIRED TESTS
Run the exact test commands listed in the task.

Rules:
- Record the full outputs verbatim (inline or log + hash).
- Exit codes must be recorded.
- If any command is skipped, the task cannot be VERIFIED COMPLETE.

### STEP 2: IF ANY FAILURES
If any test or audit fails:
- fix code (within scope only)
- re-run the same command(s)
- record the new outputs verbatim
Repeat until all required checks pass or BLOCKED is reached.

### STEP 3: RUN REQUIRED AUDITS (IF ANY)
If the task includes audits (rg/grep, schema checks, lint, formatting):
- run them exactly
- record outputs verbatim
- audits must be hard gates (no “warn but pass”)

**Standard audit commands available:**
- `python CAPABILITY/AUDIT/root_audit.py --verbose` - Verifies INV-006 (output roots compliance)
- `python CAPABILITY/TOOLS/governance/critic.py` - Checks canon compliance (INV-009, INV-011)
- `python LAW/CONTRACTS/runner.py` - Runs all fixtures (INV-004: fixtures gate merges)
- `rg` or `grep` enforcement gates - Custom pattern searches for forbidden constructs
- Schema validation - JSON schema checks for governance objects

### STEP 4: FINAL REPORT (STRICT)
Final report must include:
- `git status` output (or log reference + hash)
- list of files changed
- exact commands executed
- full outputs for tests and audits (or log references + hashes)
- final status: **VERIFIED COMPLETE | PARTIAL | BLOCKED**
- if PARTIAL: remaining violations with file:line
- if BLOCKED: precise constraint and the minimal change needed to unblock

You are forbidden from using “complete/done/verified” language unless status is **VERIFIED COMPLETE**.

## Gate test requirements
If a task involves enforcement (firewalls, receipts, invariants, pruning, domain safety), it must include at least one gate test with this semantic:

- It must **fail** if the forbidden condition exists.
- It must **pass only** when the forbidden condition is absent.

Examples:
- “No raw writes remain in scope”
- “No forbidden domain writes”
- “Commit gate blocks durable writes pre-commit”
- “PRUNED never appears when emit_pruned=false”
- “Repo digest unchanged on failure”

## Forbidden anti-patterns
These are disallowed in CANON tasks:

- A scanner prints violations but exits 0
- A test exists but is not wired into pytest or the gate
- An agent reports “tests pass” without recorded outputs
- An agent runs a different test set than requested
- An agent changes scope to make tests pass
- An agent declares success while any mandatory command exits nonzero

## Standard final report template (required)
Use this structure exactly:

### FINAL REPORT

#### 1) Scope + clean state
- Allowed scope:
- `git status`:
- (paste verbatim, or provide LOG path + sha256 + size + head and tail excerpts)

#### 2) Files changed
- (list)

#### 3) Commands executed
- (exact commands)

#### 4) Test outputs
- (paste verbatim, or LOG references + hashes)

#### 5) Audit outputs
- (paste verbatim, or LOG references + hashes)

#### 6) Result
- Status: VERIFIED COMPLETE | PARTIAL | BLOCKED
- If PARTIAL: remaining violations (file:line)
- If BLOCKED: precise blocker and minimal change needed to unblock

## Recommended integration points
To make this automatic, require the Verification Contract block in:
- every agent prompt template
- every task prompt that touches invariants or production behavior
- every roadmap phase that can change production behavior

If a prompt lacks the contract, it is malformed for execution work.
